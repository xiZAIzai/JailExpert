
from fastchat.conversation import get_conv_template
from openai import OpenAI
import logging
from typing import Optional, Dict, List, Any
import time
import numpy as np
from typing import List
from mutation import *
from model_utils import OpenAIClient
from experience_state import JailbreakState
from tqdm import tqdm

def normalize(value, min_val=0, max_val=5):
    if min_val == max_val:
        raise ValueError("min_val and max_val cannot be the same")
    normalized_value = (value - min_val) / (max_val - min_val)
    return np.clip(normalized_value, 0, 1)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def query_to_vector(text: str, model="text-embedding-3-small", retries=30, delay=1, **kwargs) -> List[float]:
    """
    将文本转换为向量，支持超时重试机制。
    
    参数:
    - text: 输入文本
    - model: 使用的嵌入模型
    - retries: 最大重试次数
    - delay: 每次重试之间的延迟时间（秒）
    - **kwargs: 其他参数传递给 API 客户端

    返回:
    - 向量列表
    """
    client = OpenAIClient(api_key="", base_url="")
    text = text.replace("\n", " ")

    for attempt in range(retries):
        try:
            response = client.embeddings.create(input=[text], model=model, **kwargs)
            return response.data[0].embedding
        except Exception as e:
            logging.info(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1: 
                time.sleep(delay)
            else:
                raise e
            
def query_to_vector_batch(texts: List[str], model="text-embedding-3-small", retries=30, delay=1, **kwargs) -> List[List[float]]:
    """
    批量转换文本为嵌入向量，内部调用 query_to_vector。
    
    参数:
    - texts: 文本列表
    - model: 使用的嵌入模型
    - retries: 最大重试次数
    - delay: 每次重试间隔
    - **kwargs: 传递给 API 客户端的其他参数

    返回:
    - 嵌入向量列表
    """
    return [query_to_vector(text, model=model, retries=retries, delay=delay, **kwargs) for text in texts]


def get_mutated_text(mutation, attack_model, input):
    mutation_type_mapping = {
        'Base64': Base64(),
        'Base64_raw': Base64(),
        'Base64_input_only': Base64(),
        'Artificial':Artificial(),
        'Disemvowel': Disemvowel(),
        'Leetspeak': Leetspeak(),
        'Rot13': Rot13(),
        'Combination_1': Combination_1(),
        'Combination_2': Combination_2(),
        'Combination_3': Combination_3(),
        'Auto_payload_splitting': Auto_payload_splitting(attack_model),
        'Auto_obfuscation':Auto_obfuscation(attack_model),
        'AlterSentenceStructure': AlterSentenceStructure(attack_model),
        'ChangeStyle': ChangeStyle(attack_model),
        'Rephrase': Rephrase(attack_model),
        'InsertMeaninglessCharacters': InsertMeaninglessCharacters(attack_model),
        'MisspellSensitiveWords': MisspellSensitiveWords(attack_model),
        'Translation': Translation(attack_model),
        'BinaryTree': BinaryTree(attack_model),
        'OddEven': OddEven(attack_model),
        'Reverse': Reverse(attack_model),
        'Length': Length(attack_model),
    }
    max_try = 0
    method = mutation_type_mapping[mutation]
    output = None
    while max_try < 30:
        max_try += 1
        output = method(input)
        if output is not None:
            break
        time.sleep(1)
    return output
    

def compute_similarity(vector_a, vector_b):
    """
    计算两个向量的余弦相似度。
    """
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def convert_to_jailbreak_state(experience_pool):
    jailbreak_states = []
    for entry in tqdm(experience_pool, desc="Converting experiences"):
        try:
            state = JailbreakState(
                    pre_query=entry.get("pre_query", ""),
                    full_query=entry.get("full_query", ""),
                    response=entry.get("response", ""),
                    harmfulness_score=entry.get("harmfulness_score", 0),
                    mutation=entry.get("mutation", []),
                    method=entry.get("method", ""),
                    success_times=entry.get("success_times", 0),
                    false_times=entry.get("false_times", 0)
            )
            jailbreak_states.append(state)
        except Exception as e:
            logging.info(f"Error processing entry: {entry}. Error: {e}")
    return jailbreak_states


