from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from tqdm import tqdm
import faiss
import pickle
import random
import logging

from attack_utils import normalize, query_to_vector_batch, compute_similarity

from utils import make_hashable

logging.basicConfig(level=logging.INFO, format='%(threadName)s: %(message)s')
random.seed(2025)

class experienceIndex:
    def __init__(self, scalar_weight=0.1, semantic_weight=0.9, max_clusters=10):
        """
        Experience index with clustering and pattern extraction.
        :param scalar_weight: Weight for scalar features.
        :param semantic_weight: Weight for semantic features.
        :param max_clusters: Maximum clusters for KMeans.
        """
        self.scalar_index = None
        self.semantic_index = None
        self.diff_semantic_index = None
        self.state_map = []
        self.scalar_weight = scalar_weight
        self.semantic_weight = semantic_weight
        self.max_clusters = max_clusters
        self.cluster_centers = None
        self.cluster_stats = {}
        self.clustered_state_map = defaultdict(list)
        self.best_num_clusters = None
        self.cluster_patterns = None

    def build_index(self, past_states, strategy=None):
        """
        Build indices for scalar and semantic vectors, and perform clustering using difference embeddings.
        :param past_states: List of experience states.
        """
        self.state_map = past_states
        scalar_vectors = []
        semantic_vectors = []
        diff_semantic_vectors = []
        
        # Batch query for embeddings to reduce API calls.
        all_texts = [state.pre_query for state in past_states] + [state.full_query for state in past_states]
        all_embeddings = query_to_vector_batch(all_texts)
        embeddings_iter = iter(all_embeddings)
        
        for state in tqdm(past_states, desc="Building FAISS indices"):
            norm_score = normalize(state.harmfulness_score, min_val=0, max_val=5)
            success_rate = state.success_times / (state.false_times + state.success_times)
            scalar_vector = np.array([norm_score, success_rate], dtype=np.float32)
            semantic_vector = np.array(next(embeddings_iter), dtype=np.float32)
            full_embedding = np.array(next(embeddings_iter), dtype=np.float32)
            diff_embedding = full_embedding - semantic_vector
            
            scalar_vectors.append(scalar_vector)
            semantic_vectors.append(semantic_vector)
            diff_semantic_vectors.append(diff_embedding)
        
        semantic_vectors = np.array(semantic_vectors, dtype=np.float32)
        scalar_vectors = np.array(scalar_vectors, dtype=np.float32)
        diff_semantic_vectors = np.array(diff_semantic_vectors, dtype=np.float32)
        
        # Build FAISS indices.
        self.diff_semantic_index = faiss.IndexFlatL2(diff_semantic_vectors.shape[1])
        self.diff_semantic_index.add(diff_semantic_vectors)
        self.semantic_index = faiss.IndexFlatL2(semantic_vectors.shape[1])
        self.semantic_index.add(semantic_vectors)
        self.scalar_index = faiss.IndexFlatL2(scalar_vectors.shape[1])
        self.scalar_index.add(scalar_vectors)
        
        # Optimize cluster number using silhouette score.
        self.best_num_clusters = self.optimize_clusters(diff_semantic_vectors)
        logging.info(f"Best number of clusters: {self.best_num_clusters}")
        kmeans = KMeans(n_clusters=self.best_num_clusters, n_init=10, random_state=42)
        cluster_ids = kmeans.fit_predict(diff_semantic_vectors)
        self.cluster_centers = kmeans.cluster_centers_
        
        # Assign experiences to clusters.
        self.clustered_state_map = defaultdict(list)
        self.cluster_stats = {}
        for i, cluster_id in enumerate(cluster_ids):
            ap = (make_hashable(self.state_map[i].mutation), self.state_map[i].method)
            self.clustered_state_map[cluster_id].append((semantic_vectors[i],
                                                          scalar_vectors[i],
                                                          diff_semantic_vectors[i],
                                                          ap,
                                                          self.state_map[i]))
            if cluster_id not in self.cluster_stats:
                self.cluster_stats[cluster_id] = {}
            if ap not in self.cluster_stats[cluster_id]:
                self.cluster_stats[cluster_id][ap] = {"attempts": 0, "successes": 0, "experience_idx": []}
            local_idx = len(self.clustered_state_map[cluster_id]) - 1
            total_attempts = self.state_map[i].success_times + self.state_map[i].false_times
            self.cluster_stats[cluster_id][ap]["attempts"] += total_attempts
            self.cluster_stats[cluster_id][ap]["successes"] += self.state_map[i].success_times
            self.cluster_stats[cluster_id][ap]["experience_idx"].append(local_idx)
        
        # Extract representative patterns per cluster.
        self.cluster_patterns = self.extract_cluster_patterns(strategy=strategy)

    def update_index(self, update_info=None, recluster=False, strategy=None):
        """
        Update indices with new experiences. If recluster is True, update clustering.
        :param update_info: Dict containing new scaler, semantic, diff vectors and states.
        """
        if recluster and update_info is not None:
            self.state_map = update_info["states"]
            scalar_vectors = update_info["scalars"]
            semantic_vectors = update_info["semantics"]
            diff_semantic_vectors = update_info["diff_semantics"]
            
            self.diff_semantic_index = faiss.IndexFlatL2(diff_semantic_vectors.shape[1])
            self.diff_semantic_index.add(diff_semantic_vectors)
            self.semantic_index = faiss.IndexFlatL2(semantic_vectors.shape[1])
            self.semantic_index.add(semantic_vectors)
            self.scalar_index = faiss.IndexFlatL2(scalar_vectors.shape[1])
            self.scalar_index.add(scalar_vectors)
            
            self.best_num_clusters = self.optimize_clusters(diff_semantic_vectors)
            logging.info(f"Updated best clusters: {self.best_num_clusters}")
            kmeans = KMeans(n_clusters=self.best_num_clusters, n_init=10, random_state=42)
            cluster_ids = kmeans.fit_predict(diff_semantic_vectors)
            self.cluster_centers = kmeans.cluster_centers_
            
            self.clustered_state_map = defaultdict(list)
            self.cluster_stats = {}
            for i, cluster_id in enumerate(cluster_ids):
                ap = (make_hashable(self.state_map[i].mutation), self.state_map[i].method)
                self.clustered_state_map[cluster_id].append((semantic_vectors[i],
                                                              scalar_vectors[i],
                                                              diff_semantic_vectors[i],
                                                              ap,
                                                              self.state_map[i]))
                if cluster_id not in self.cluster_stats:
                    self.cluster_stats[cluster_id] = {}
                if ap not in self.cluster_stats[cluster_id]:
                    self.cluster_stats[cluster_id][ap] = {"attempts": 0, "successes": 0, "experience_idx": []}
                local_idx = len(self.clustered_state_map[cluster_id]) - 1
                total_attempts = self.state_map[i].success_times + self.state_map[i].false_times
                self.cluster_stats[cluster_id][ap]["attempts"] += total_attempts
                self.cluster_stats[cluster_id][ap]["successes"] += self.state_map[i].success_times
                self.cluster_stats[cluster_id][ap]["experience_idx"].append(local_idx)
        # Always update representative patterns.
        self.cluster_patterns = self.extract_cluster_patterns(strategy=strategy)

    def optimize_clusters(self, data):
        """Determine optimal number of clusters using Silhouette Score."""
        pca = PCA(n_components=0.4, random_state=42)
        reduced_data = pca.fit_transform(data)
        best_score, best_k = -1, 2
        for k in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(reduced_data)
            score = silhouette_score(reduced_data, labels)
            logging.info(f"Silhouette score for k={k}: {score}")
            if score > best_score:
                best_score = score
                best_k = k
        return best_k

    def extract_cluster_patterns(self, strategy=None):
        """
        Extract representative mutation and prompt patterns for each cluster.
        :param strategy: Strategy for pattern extraction ('random' for random selection).
        :return: Dict of patterns per cluster.
        """
        cluster_patterns = {}
        for cluster_id, states in self.clustered_state_map.items():
            pairs = []
            for _, _, _, _, state in states:
                if isinstance(state.mutation, list):
                    mutation = make_hashable(state.mutation)
                elif isinstance(state.mutation, str):
                    mutation = (state.mutation,)
                else:
                    logging.info(f"Unexpected mutation type: {type(state.mutation)}")
                    continue
                pairs.append((mutation, state.method))
            if strategy != "random":
                combined = self._extract_high_frequency_patterns_with_diversity(pairs, cluster_id, top_k=5)
            else:
                combined = self._extract_random_strategy(pairs, cluster_id)
            cluster_patterns[cluster_id] = {"combined_patterns": combined}
        return cluster_patterns

    def _extract_high_frequency_patterns_with_diversity(self, items, cluster_id, top_k=5):
        """
        Extract high frequency patterns with integrated success rate.
        :param items: List of (mutation, method) pairs.
        :param cluster_id: Cluster identifier.
        :param top_k: Number of patterns to extract.
        :return: Dict containing top patterns.
        """
        from collections import Counter
        counter = Counter(items)
        most_common = counter.most_common(top_k)
        top_items = [item for item, _ in most_common]
        success_rates = {}
        for item, _ in most_common:
            attempts = self.cluster_stats[cluster_id].get(item, {}).get("attempts", 0)
            successes = self.cluster_stats[cluster_id].get(item, {}).get("successes", 0)
            success_rates[item] = successes / attempts if attempts > 0 else 0
        pattern_scores = [(item, success_rates[item]) for item in top_items]
        sorted_patterns = sorted(pattern_scores, key=lambda x: x[1], reverse=True)
        top_patterns = [item for item, _ in sorted_patterns]
        return {"patterns": top_patterns}

    def _extract_random_strategy(self, items, cluster_id):
        """
        Randomly select a strategy from the given items.
        :param items: List of (mutation, method) pairs.
        :param cluster_id: Cluster identifier.
        :return: Dict containing a random pattern.
        """
        import random
        if not items:
            logging.info(f"No items found for cluster {cluster_id}.")
            return None
        random_item = random.choice(items)
        return {"patterns": [random_item]}

    def search_within_cluster(self, query_vector, cluster_id, top_k=10):
        """
        Search for most similar states in a given cluster.
        :param query_vector: Query embedding vector.
        :param cluster_id: Target cluster id.
        :param top_k: Number of nearest neighbors to return.
        :return: List of tuples (index, state, similarity).
        """
        cluster_data = self.clustered_state_map[cluster_id]
        semantic_vectors, scalar_vectors, _, _, state_objects = zip(*cluster_data)
        semantic_similarities = [compute_similarity(query_vector, s_vec) * s_scalar[1] for s_vec, s_scalar in zip(semantic_vectors, scalar_vectors)]
        ranked_indices = np.argsort(semantic_similarities)[::-1]
        return [(i, state_objects[i], semantic_similarities[i]) for i in ranked_indices[:top_k]]

    def save_index(self, semantic_index_path, scalar_index_path, state_map_path, cluster_info_path, cluster_patterns_path):
        """
        Save FAISS indices, state map and cluster info.
        """
        if self.semantic_index:
            faiss.write_index(self.semantic_index, semantic_index_path)
        if self.scalar_index:
            faiss.write_index(self.scalar_index, scalar_index_path)
        with open(state_map_path, 'wb') as f:
            pickle.dump(self.state_map, f)
        cluster_info = {"cluster_centers": self.cluster_centers,
                        "clustered_state_map": self.clustered_state_map,
                        "cluster_stats": self.cluster_stats}
        with open(cluster_info_path, 'wb') as f:
            pickle.dump(cluster_info, f)
        with open(cluster_patterns_path, 'wb') as f:
            pickle.dump(self.cluster_patterns, f)

    def load_index(self, semantic_index_path, scalar_index_path, state_map_path, cluster_info_path, cluster_patterns_path):
        """
        Load FAISS indices, state map and cluster info.
        """
        self.semantic_index = faiss.read_index(semantic_index_path)
        self.scalar_index = faiss.read_index(scalar_index_path)
        with open(state_map_path, 'rb') as f:
            self.state_map = pickle.load(f)
        with open(cluster_info_path, 'rb') as f:
            cluster_info = pickle.load(f)
            self.cluster_centers = cluster_info["cluster_centers"]
            self.clustered_state_map = cluster_info["clustered_state_map"]
            self.cluster_stats = cluster_info["cluster_stats"]
        with open(cluster_patterns_path, 'rb') as f:
            self.cluster_patterns = pickle.load(f)