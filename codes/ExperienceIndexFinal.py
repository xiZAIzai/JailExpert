from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from tqdm import tqdm
import faiss
import pickle
from utils import normalize, query_to_vector, query_to_vector_batch
from utils import compute_similarity
from utils import JailbreakState, make_hashable
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(threadName)s: %(message)s')

random.seed(2025)
def state_to_vector_version_4(state):

    harmfulness_score_norm = normalize(state.harmfulness_score, min_val=0, max_val=5)
    scalar_vector = np.array([harmfulness_score_norm], dtype=np.float32)
    return scalar_vector

class ExperienceIndexVersion4:
    def __init__(self, scalar_weight=0.1, semantic_weight=0.9, max_clusters=10):
        """
        ExperienceIndex with enhanced clustering and jailbreak pattern extraction.
        :param scalar_weight: Weight of scalar features in similarity calculations.
        :param semantic_weight: Weight of semantic features in similarity calculations.
        :param max_clusters: Maximum number of clusters to consider for optimization.
        """
        self.scalar_index = None
        self.semantic_index = None
        self.diff_semantic_index = None  # Index for difference embeddings
        self.state_map = []  
        self.scalar_weight = scalar_weight
        self.semantic_weight = semantic_weight
        self.max_clusters = max_clusters 
        self.cluster_centers = None
        self.cluster_stats = {}
        self.clustered_state_map = defaultdict(list)  # 每个簇的状态映射
        self.best_num_clusters = None
        self.cluster_patterns = None  # Patterns per cluster

    def build_index(self, past_states, strategy=None):
        """
        Build indices for scalar, semantic vectors, and perform KMeans clustering based on difference embeddings.
        :param past_states: Experience pool containing all states.
        """
        self.state_map = past_states
        scalar_vectors = []
        semantic_vectors = []
        diff_semantic_vectors = []

        # 将所有需要查询的文本批量化（减少API调用次数）
        all_texts = []
        for state in past_states:
            all_texts.append(state.pre_query)  # 添加 pre_query
            all_texts.append(state.full_query)  # 添加 full_query

        # 调用 query_to_vector 一次性获取所有嵌入
        all_embeddings = query_to_vector_batch(all_texts)  # 假设 query_to_vector 支持批量输入

        # 按照 pre_query 和 full_query 的顺序拆分嵌入
        embeddings_iter = iter(all_embeddings)
        for state in tqdm(past_states, desc="Building FAISS indices for experience pool"):
            # Normalize scalar features
            harmfulness_score_norm = normalize(state.harmfulness_score, min_val=0, max_val=5)
            success_rate = state.success_times / (state.false_times + state.success_times)

            # Combine scalar features -- 有害分数和成功率（都是越高越好）
            scalar_vector = np.array([harmfulness_score_norm, success_rate], dtype=np.float32)

            # Process semantic features
            semantic_vector = np.array(next(embeddings_iter), dtype=np.float32)  # 获取 pre_query 的向量
            full_jail_embedding = np.array(next(embeddings_iter), dtype=np.float32)  # 获取 full_query 的向量

            # Compute the difference embedding
            diff_embedding = full_jail_embedding - semantic_vector

            # Store vectors
            scalar_vectors.append(scalar_vector)
            semantic_vectors.append(semantic_vector)
            diff_semantic_vectors.append(diff_embedding)

        # Convert to NumPy arrays
        semantic_vectors = np.array(semantic_vectors, dtype=np.float32)
        scalar_vectors = np.array(scalar_vectors, dtype=np.float32)
        diff_semantic_vectors = np.array(diff_semantic_vectors, dtype=np.float32)

        # Build indices
        # Difference embedding index
        self.diff_semantic_index = faiss.IndexFlatL2(diff_semantic_vectors.shape[1])
        self.diff_semantic_index.add(diff_semantic_vectors)

        # Semantic index for pre_query
        self.semantic_index = faiss.IndexFlatL2(semantic_vectors.shape[1])
        self.semantic_index.add(semantic_vectors)

        # Scalar index
        self.scalar_index = faiss.IndexFlatL2(scalar_vectors.shape[1])
        self.scalar_index.add(scalar_vectors)

        # Perform KMeans clustering with optimization on difference embeddings
        self.best_num_clusters = self.optimize_clusters(diff_semantic_vectors)
        logging.info(f"====Best number of clusters: {self.best_num_clusters}====")
        kmeans = KMeans(n_clusters=self.best_num_clusters, n_init=10, random_state=42)
        cluster_ids = kmeans.fit_predict(diff_semantic_vectors)

        self.cluster_centers = kmeans.cluster_centers_

        # Assign states to clusters
        self.clustered_state_map = defaultdict(list)
        # self.cluster_stats = {}
        # for i, cluster_id in enumerate(cluster_ids):
        #     # Store (scalar, difference semantic, state object) in cluster
        #     self.clustered_state_map[cluster_id].append(
        #         (scalar_vectors[i], semantic_vectors[i], diff_semantic_vectors[i], self.state_map[i])
        #     )
        #     if cluster_id not in self.cluster_stats:
        #         self.cluster_stats[cluster_id] = {"attempts": 0, "successes": 0}
        #     self.cluster_stats[cluster_id]["attempts"] += 1
        #     self.cluster_stats[cluster_id]["successes"] += 1

        # # Compute initial success rates for clusters
        # for cluster_id in self.cluster_stats:
        #     attempts = self.cluster_stats[cluster_id]["attempts"]
        #     successes = self.cluster_stats[cluster_id]["successes"]
        #     success_rate = successes / attempts  # Avoid division by zero
        #     self.cluster_stats[cluster_id]["success_rate"] = success_rate
        self.cluster_stats = {}
        # 遍历所有簇
        for i, cluster_id in enumerate(cluster_ids):
            # Store (scalar, difference semantic, state object) in cluster
            mutation_prompt_pair = (make_hashable(self.state_map[i].mutation), self.state_map[i].method)

            # 在簇内追加对应的信息
            self.clustered_state_map[cluster_id].append(
                (semantic_vectors[i], scalar_vectors[i], diff_semantic_vectors[i], mutation_prompt_pair, self.state_map[i])
            )

            # 如果当前簇的统计信息还不存在，则初始化
            if cluster_id not in self.cluster_stats:
                self.cluster_stats[cluster_id] = {}

            # 如果当前簇下的策略 (mutation_prompt_pair) 不存在，则初始化
            if mutation_prompt_pair not in self.cluster_stats[cluster_id]:
                self.cluster_stats[cluster_id][mutation_prompt_pair] = {
                    "attempts": 0,
                    "successes": 0,
                    "experience_idx": []
                }

            # 获取当前簇的所有经验
            cluster_experiences = self.clustered_state_map[cluster_id]

            # 获取当前簇内的索引 (local_idx)
            local_idx = len(cluster_experiences) - 1  # 当前簇中的新加入经验的索引

            # 更新尝试次数，包括成功和失败的尝试
            self.cluster_stats[cluster_id][mutation_prompt_pair]["attempts"] += (self.state_map[i].success_times + self.state_map[i].false_times)

            # 更新成功次数
            self.cluster_stats[cluster_id][mutation_prompt_pair]["successes"] += self.state_map[i].success_times

            # 更新当前策略所属经验的idx（在当前簇内的顺序），用于后续动态更新代表策略的历史分数
            self.cluster_stats[cluster_id][mutation_prompt_pair]["experience_idx"].append(local_idx)

        # 初始化簇的代表策略---
        self.cluster_patterns = self.extract_cluster_patterns(strategy=strategy)

    def update_index(self, update_info=None, recluster=False, strategy=None):
        """
        Build indices for scalar, semantic vectors, and perform KMeans clustering based on difference embeddings.
        :param past_states: Experience pool containing all states.
        """
        # 如果有新的经验加入时，才需要重新聚簇
        if recluster:
            self.state_map = update_info["states"]
            scalar_vectors = update_info["scalars"]
            semantic_vectors = update_info["semantics"]
            diff_semantic_vectors = update_info["diff_semantics"]

            # Build indices
            # Difference embedding index
            self.diff_semantic_index = faiss.IndexFlatL2(diff_semantic_vectors.shape[1])
            self.diff_semantic_index.add(diff_semantic_vectors)

            # Semantic index for pre_query
            self.semantic_index = faiss.IndexFlatL2(semantic_vectors.shape[1])
            self.semantic_index.add(semantic_vectors)

            # Scalar index
            self.scalar_index = faiss.IndexFlatL2(scalar_vectors.shape[1])
            self.scalar_index.add(scalar_vectors)

            # Perform KMeans clustering with optimization on difference embeddings
            self.best_num_clusters = self.optimize_clusters(diff_semantic_vectors)
            logging.info(f"====Best number of clusters: {self.best_num_clusters}====")
            kmeans = KMeans(n_clusters=self.best_num_clusters, n_init=10, random_state=42)
            cluster_ids = kmeans.fit_predict(diff_semantic_vectors)

            self.cluster_centers = kmeans.cluster_centers_

            # Assign states to cluster, 当更新时，需要重新分配簇吗？
            self.clustered_state_map = defaultdict(list)
            self.cluster_stats = {}
            # 遍历所有簇
            for i, cluster_id in enumerate(cluster_ids):
                # Store (scalar, difference semantic, state object) in cluster
                mutation_prompt_pair = (make_hashable(self.state_map[i].mutation), self.state_map[i].method)

                # 在簇内追加对应的信息
                self.clustered_state_map[cluster_id].append(
                    (semantic_vectors[i], scalar_vectors[i], diff_semantic_vectors[i], mutation_prompt_pair, self.state_map[i])
                )

                # 如果当前簇的统计信息还不存在，则初始化
                if cluster_id not in self.cluster_stats:
                    self.cluster_stats[cluster_id] = {}

                # 如果当前簇下的策略 (mutation_prompt_pair) 不存在，则初始化
                if mutation_prompt_pair not in self.cluster_stats[cluster_id]:
                    self.cluster_stats[cluster_id][mutation_prompt_pair] = {
                        "attempts": 0,
                        "successes": 0,
                        "experience_idx": []
                    }

                # 获取当前簇的所有经验
                cluster_experiences = self.clustered_state_map[cluster_id]

                # 获取当前簇内的索引 (local_idx)
                local_idx = len(cluster_experiences) - 1  # 当前簇中的新加入经验的索引

                # 更新尝试次数，包括成功和失败的尝试
                self.cluster_stats[cluster_id][mutation_prompt_pair]["attempts"] += (self.state_map[i].success_times + self.state_map[i].false_times)

                # 更新成功次数
                self.cluster_stats[cluster_id][mutation_prompt_pair]["successes"] += self.state_map[i].success_times

                # 更新当前策略所属经验的idx（在当前簇内的顺序），用于后续动态更新代表策略的历史分数
                self.cluster_stats[cluster_id][mutation_prompt_pair]["experience_idx"].append(local_idx)
            
        # 更新每个簇的代表策略（用更新的历史经验）
        self.cluster_patterns = self.extract_cluster_patterns(strategy=strategy)

    def optimize_clusters(self, data):
        """
        Optimize the number of clusters using Silhouette Score.
        :param data: Data to cluster.
        :return: Optimal number of clusters.
        """
        pca = PCA(n_components=0.4, random_state=42)  # 保留 95% 的方差信息
        reduced_data = pca.fit_transform(data)
        best_score = -1
        best_k = 2
        for k in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(reduced_data)
            score = silhouette_score(reduced_data, labels)
            logging.info(score)
            if score > best_score:
                best_score = score
                best_k = k
        return best_k

    # def extract_cluster_patterns(self):
    #     """
    #     Extract high-frequency mutations and jailbreak prompts for each cluster, including mutation length distribution.
    #     :return: Patterns per cluster.
    #     """
    #     cluster_patterns = {}
    #     for cluster_id, states in self.clustered_state_map.items():
    #         mutations = []
    #         jail_prompts = []
    #         mutation_lengths = []
    #
    #         for _, _, state in states:
    #             # If mutation is a list, flatten it and record lengths
    #             if isinstance(state.mutation, list):
    #                 mutations.extend(state.mutation)
    #                 mutation_lengths.append(len(state.mutation))
    #             else:
    #                 mutations.append(state.mutation)
    #                 mutation_lengths.append(1)  # Single mutation has length 1
    #
    #             jail_prompts.append(state.method)
    #
    #         # Use term frequency to extract common patterns
    #         mutation_patterns = self._extract_high_frequency_patterns(mutations)
    #         jail_prompt_patterns = self._extract_high_frequency_patterns(jail_prompts)
    #
    #         # Calculate representative mutation length (e.g., median or mode)
    #         representative_length = self._calculate_representative_length(mutation_lengths)
    #
    #         cluster_patterns[cluster_id] = {
    #             "mutation_patterns": mutation_patterns,
    #             "jail_prompt_patterns": jail_prompt_patterns,
    #             "representative_mutation_length": representative_length,
    #         }
    #     return cluster_patterns
    #
    # def _extract_high_frequency_patterns(self, items):
    #     """
    #     Extract high-frequency patterns using term frequency.
    #     :param items: List of text items.
    #     :return: High-frequency patterns.
    #     """
    #     from collections import Counter
    #
    #     counter = Counter(items)
    #     most_common = counter.most_common(5)  # Top 5 patterns
    #     return [item for item, _ in most_common]
    #
    # def _calculate_representative_length(self, lengths):
    #     """
    #     Calculate the representative length of mutations in a cluster.
    #     :param lengths: List of lengths of mutations.
    #     :return: Representative length (e.g., median or mode).
    #     """
    #     from statistics import median
    #     from collections import Counter
    #
    #     # Option 1: Use median as the representative length
    #     median_length = median(lengths)
    #
    #     # Option 2: Use mode (most common length) as the representative length
    #     length_counter = Counter(lengths)
    #     mode_length = length_counter.most_common(1)[0][0]
    #
    #     # You can choose either median or mode; here we return median
    #     return int(mode_length)
    # def extract_cluster_patterns(self):
    #     """
    #     Extract high-frequency mutations and jailbreak prompts for each cluster, integrating diversity evaluation.
    #     :return: Patterns per cluster.
    #     """
    #     cluster_patterns = {}
    #     for cluster_id, states in self.clustered_state_map.items():
    #         mutations = []
    #         jail_prompts = []
    #         mutation_lengths = []
    #
    #         for _, _, _, state in states:
    #             # If mutation is a list, flatten it and record lengths
    #             if isinstance(state.mutation, list):
    #                 mutations.append(state.mutation)
    #                 mutation_lengths.append(len(state.mutation))
    #             elif isinstance(state.mutation, str):
    #                 mutations.append([state.mutation])
    #                 mutation_lengths.append(1)  # Single mutation has length 1
    #
    #             jail_prompts.append(state.method)
    #         mutations = [tuple(item) for item in mutations]
    #         # Use term frequency to extract common patterns
    #         mutation_patterns = self._extract_high_frequency_patterns_with_diversity(mutations, top_k=1)
    #         print(mutation_patterns)
    #         jail_prompt_patterns = self._extract_high_frequency_patterns_with_diversity(jail_prompts, top_k=1)
    #
    #         # Calculate representative mutation length (e.g., median or mode)
    #         representative_length = self._calculate_representative_length(mutation_lengths)
    #
    #         cluster_patterns[cluster_id] = {
    #             "mutation_patterns": mutation_patterns,
    #             "jail_prompt_patterns": jail_prompt_patterns,
    #             "representative_mutation_length": representative_length,
    #         }
    #     return cluster_patterns

    def extract_cluster_patterns(self, strategy=None):
        """
        Extract high-frequency mutation and jailbreak prompt combinations for each cluster, integrating diversity evaluation.
        :return: Patterns per cluster.
        """
        cluster_patterns = {}
        for cluster_id, states in self.clustered_state_map.items():
            mutation_jail_prompt_pairs = []

            for _, _, _, _, state in states:
                if isinstance(state.mutation, list):
                    mutation = make_hashable(state.mutation)  # 转换为可哈希的 tuple
                elif isinstance(state.mutation, str):
                    mutation = (state.mutation,)
                else:
                    logging.info(f"Unexpected mutation type: {type(state.mutation)}, skipping.")
                    continue

                pair = (mutation, state.method)
                mutation_jail_prompt_pairs.append(pair)

            if strategy != "random":
                # Use term frequency to extract common patterns for mutation + jailbreak prompt pairs
                combined_patterns = self._extract_high_frequency_patterns_with_diversity(mutation_jail_prompt_pairs, cluster_id, top_k=5)
            else:
                # random choose from cluster
                combined_patterns = self._extract_random_strategy(mutation_jail_prompt_pairs, cluster_id)

            cluster_patterns[cluster_id] = {
                "combined_patterns": combined_patterns
            }

        return cluster_patterns

    def _extract_high_frequency_patterns_with_diversity(self, items, cluster_id, top_k=5):
        """
        Extract high-frequency patterns for combined items using term frequency, 
        and integrate diversity evaluation with success rate and preference based on attempts.
        :param items: List of combined items (e.g., tuples of mutation and jailbreak prompt).
        :param top_k: Number of top patterns to extract.
        :return: High-frequency patterns integrated with success rate and diversity scores.
        """
        from collections import Counter
        from scipy.stats import entropy
        import numpy as np

        # 计算每个策略的频率
        counter = Counter(items)
        most_common = counter.most_common(top_k)  # 获取前 top_k 个策略

        # 获取策略的列表（用于后续计算）
        top_items = [item for item, _ in most_common]

        # 计算每个策略的成功率
        # 并且计算每个策略在簇中的占比 (attempts_tmp / attempts_all)
        success_rates = {}
        # preferences = {}
        # total_attempts = sum(self.cluster_stats[cluster_id]["attempts"].values())

        for item, _ in most_common:
            # attempts_tmp = self.cluster_stats[cluster_id]["attempts"].get(item, 0)
            attempts_tmp = self.cluster_stats[cluster_id].get(item, {}).get("attempts", 0)
            # successes = self.cluster_stats[cluster_id]["successes"].get(item, 0)
            successes = self.cluster_stats[cluster_id].get(item, {}).get("successes", 0)

            # 成功率
            success_rate = successes / attempts_tmp if attempts_tmp > 0 else 0
            success_rates[item] = success_rate

            # # 占比
            # preference = attempts_tmp / total_attempts if total_attempts > 0 else 0
            # preferences[item] = preference

        # 结合成功率和占比的分数来计算每个策略的优先级
        pattern_scores = []
        for item in top_items:
            # score = success_rates[item] * preferences[item]  # 联合评分
            score = success_rates[item]
            pattern_scores.append((item, score))

        # 根据评分排序
        sorted_patterns = sorted(pattern_scores, key=lambda x: x[1], reverse=True)

        # 提取高频模式和它们的多样性得分（熵）
        top_patterns = [item for item, _ in sorted_patterns]
        

        return {
            "patterns": top_patterns
            # "diversity_score": normalized_H
        }

    def _extract_random_strategy(self, items, cluster_id):
        """
        Extract a random strategy from the given list of items for a cluster.
        :param items: List of combined items (e.g., tuples of mutation and jailbreak prompt).
        :return: A randomly selected pattern from the cluster.
        """
        if not items:
            logging.info(f"No items found for cluster {cluster_id}.")
            return None

        # Randomly pick one item from the list
        random_item = random.choice(items)

        # Return the random strategy along with its success rate (optional)
        return {
            "patterns": [random_item]
        }



    # def _extract_high_frequency_patterns_with_diversity(self, items, top_k=5):
    #     """
    #     Extract high-frequency patterns for combined items using term frequency and integrate diversity evaluation.
    #     :param items: List of combined items (e.g., tuples of mutation and jailbreak prompt).
    #     :param top_k: Number of top patterns to extract.
    #     :return: High-frequency patterns integrated with diversity scores.
    #     """
    #     from collections import Counter
    #     from scipy.stats import entropy
    #     import numpy as np

    #     # Calculate term frequency
    #     counter = Counter(items)
    #     most_common = counter.most_common(top_k)  # Top k patterns

    #     top_items = [item for item, _ in most_common]
    #     # for item, count in most_common:
    #     #     print(item, count)

    #     # Calculate probabilities for diversity (entropy) calculation
    #     probabilities = [count / sum(counter.values()) for _, count in most_common]
    #     diversity_score = entropy(probabilities, base=np.e)

    #     # Maximum entropy (assuming uniform distribution)
    #     max_H = np.log(len(probabilities)) if probabilities else 0

    #     # Normalized entropy (diversity score)
    #     normalized_H = diversity_score / max_H if max_H > 0 else 0

    #     # Combine patterns with diversity score
    #     return {
    #         "patterns": top_items,
    #         "diversity_score": normalized_H
    #     }

    def search_within_cluster(self, query_vector, cluster_id, top_k=10):
        """
        搜索聚类中的状态，根据 query_state 的轮次决定相似度计算方式。
        :param query_state: 查询的状态
        :param cluster_id: 聚类 ID
        :param top_k: 返回的最近邻数量
        """
        cluster_vectors = self.clustered_state_map[cluster_id]  # 获取指定簇的存储向量
        semantic_vectors, scalar_vectors,  _, _, state_objects = zip(*cluster_vectors)
        # scalar_vectors = np.array(scalar_vectors, dtype=np.float32)
        # semantic_vectors = np.array(semantic_vectors, dtype=np.float32)

        # 判断是否是初始状态（round_num == 0）,初始状态才是一个状态，后续在树搜索中的都是使用经验池中的tuple 向量。
        if query_vector is not None:
            # if query_state.round_num == 0:
            semantic_similarities = [compute_similarity(query_vector, semantic_vec) * scalar_vector[1] for semantic_vec, scalar_vector in
                                     zip(semantic_vectors, scalar_vectors)]
            combined_similarities = semantic_similarities  # 只考虑语义相似度
            # else:
            #     query_semantic = np.array(query_to_vector(query_state.pre_query), dtype=np.float32)
            #     query_scalar = np.array(state_to_vector_version_4(query_state), dtype=np.float32)
            #     scalar_distances = np.linalg.norm(scalar_vectors - query_scalar, axis=1)
            #     semantic_similarities = [compute_similarity(query_semantic, semantic_vec) for semantic_vec in
            #                              semantic_vectors]
            #     combined_similarities = [-self.scalar_weight * scalar_dist + self.semantic_weight * semantic_sim for
            #                              scalar_dist, semantic_sim in zip(scalar_distances, semantic_similarities)]
        # else:
        #     query_scalar, query_semantic = query_state[0], query_state[1]
        #     scalar_distances = np.linalg.norm(scalar_vectors - query_scalar, axis=1)
        #     semantic_similarities = [compute_similarity(query_semantic, semantic_vec) for semantic_vec in
        #                              semantic_vectors]
        #     combined_similarities = [-self.scalar_weight * scalar_dist + self.semantic_weight * semantic_sim for
        #                              scalar_dist, semantic_sim in zip(scalar_distances, semantic_similarities)]

        # 按相似度排序
        ranked_indices = np.argsort(combined_similarities)[::-1]  # 降序排序，越高的相似度排在前面

        # 返回 top_k 的状态及相似度
        return [(i, state_objects[i], combined_similarities[i]) for i in ranked_indices[:top_k]]

    def random_sample_experience(self, query_state, top_k=5):
        # 获取 self.state_map 的索引列表
        indices = list(range(len(self.state_map)))
        
        # 从索引列表中采样
        sample_indices = random.sample(indices, top_k)
        
        # 根据采样的索引获取对应的元素
        sample_states = [self.state_map[i] for i in sample_indices]
        
        # 返回采样的元素和它们的下标
        return sample_states, sample_indices
    

    # def insert_new_experience(self, new_states):
    #     """
    #     Insert new attack experiences and update clusters.
    #     :param new_states: List of new states to insert.
    #     """
    #     scalar_vectors = []
    #     diff_semantic_vectors = []
    #
    #     # Step 1: Process new states
    #     for state in tqdm(new_states, desc="Processing new states"):
    #         # Normalize scalar features
    #         harmfulness_score_norm = normalize(state.harmfulness_score, min_val=0, max_val=5)
    #         scalar_vector = np.array([harmfulness_score_norm], dtype=np.float32)
    #
    #         # Compute difference embedding
    #         query_embedding = np.array(query_to_vector(state.pre_query), dtype=np.float32)
    #         full_jail_embedding = np.array(query_to_vector(state.full_query), dtype=np.float32)
    #         diff_embedding = full_jail_embedding - query_embedding
    #
    #         scalar_vectors.append(scalar_vector)
    #         diff_semantic_vectors.append(diff_embedding)
    #
    #     # Convert to NumPy arrays
    #     scalar_vectors = np.array(scalar_vectors, dtype=np.float32)
    #     diff_semantic_vectors = np.array(diff_semantic_vectors, dtype=np.float32)
    #
    #     # Step 2: Add new data to FAISS indices
    #     self.scalar_index.add(scalar_vectors)
    #     self.diff_semantic_index.add(diff_semantic_vectors)
    #
    #     # Step 3: Assign new data to clusters
    #     for i, diff_embedding in enumerate(diff_semantic_vectors):
    #         best_cluster = None
    #         best_distance = float('inf')
    #
    #         # Find the closest cluster center
    #         for cluster_id, center in enumerate(self.cluster_centers):
    #             diff_vector = center - diff_embedding
    #             distance = np.linalg.norm(diff_vector)
    #             if distance < best_distance:
    #                 best_distance = distance
    #                 best_cluster = cluster_id
    #
    #         # Update cluster assignments
    #         self.clustered_state_map[best_cluster].append(
    #             (scalar_vectors[i], diff_embedding, new_states[i])
    #         )
    #         self.cluster_stats[best_cluster]["attempts"] += 1
    #
    #     # Step 4: Optionally recompute cluster centers
    #     self.recompute_cluster_centers()

    # def recompute_cluster_centers(self):
    #     """
    #     Recompute cluster centers based on current assignments.
    #     """
    #     new_centers = []
    #     for cluster_id, states in self.clustered_state_map.items():
    #         embeddings = [item[1] for item in states]  # Collect difference embeddings
    #         new_center = np.mean(embeddings, axis=0)
    #         new_centers.append(new_center)

    #     self.cluster_centers = np.array(new_centers, dtype=np.float32)

    def save_index(self, semantic_index_path, scalar_index_path, state_map_path, cluster_info_path, cluster_patterns_path):
        """
        Save indices, state map, and cluster information, and cluster patterns.
        :param semantic_index_path: Path to save the semantic index.
        :param scalar_index_path: Path to save the scalar index.
        :param state_map_path: Path to save the state map.
        :param cluster_info_path: Path to save the cluster information.
        :param cluster_patterns_path: Path to save the cluster patterns.
        """
        if self.semantic_index:
            faiss.write_index(self.semantic_index, semantic_index_path)
        if self.scalar_index:
            faiss.write_index(self.scalar_index, scalar_index_path)

        logging.info(len(self.state_map))

        with open(state_map_path, 'wb') as f:
            pickle.dump(self.state_map, f)
        cluster_info = {
            "cluster_centers": self.cluster_centers,
            "clustered_state_map": self.clustered_state_map,
            "cluster_stats": self.cluster_stats,
        }
        with open(cluster_info_path, 'wb') as f:
            pickle.dump(cluster_info, f)
        # Save cluster patterns
        with open(cluster_patterns_path, 'wb') as f:
            pickle.dump(self.cluster_patterns, f)

    def load_index(self, semantic_index_path, scalar_index_path, state_map_path, cluster_info_path, cluster_patterns_path):
        """
        Load indices, state map, and cluster information, and cluster patterns.
        :param semantic_index_path: Path to load the semantic index.
        :param scalar_index_path: Path to load the scalar index.
        :param state_map_path: Path to load the state map.
        :param cluster_info_path: Path to load the cluster information.
        :param cluster_patterns_path: Path to load the cluster patterns.
        """
        # self.diff_semantic_index = faiss.read_index(semantic_index_path)
        self.semantic_index = faiss.read_index(semantic_index_path)
        self.scalar_index = faiss.read_index(scalar_index_path)
        with open(state_map_path, 'rb') as f:
            self.state_map = pickle.load(f)
        with open(cluster_info_path, 'rb') as f:
            cluster_info = pickle.load(f)
            self.cluster_centers = cluster_info["cluster_centers"]
            self.clustered_state_map = cluster_info["clustered_state_map"]
            self.cluster_stats = cluster_info["cluster_stats"]
        # Load cluster patterns
        with open(cluster_patterns_path, 'rb') as f:
            self.cluster_patterns = pickle.load(f)

