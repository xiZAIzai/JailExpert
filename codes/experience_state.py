import pandas as pd
import os
import json


class JailbreakState:
    def __init__(self, 
                 mutation, # 
                 method, # method应该用jailbreak-prompt来替代
                 full_query,  # 完整 query
                 pre_query, # 应用方法前的 query
                 response,  # 模型生成的响应
                 harmfulness_score,  # 模型响应的有害性评估（1-5）
                 success_times, # 成功次数
                 false_times, # 失败次数
                 ):  
        self.mutation = mutation
        self.full_query = full_query
        self.pre_query = pre_query
        self.response = response
        self.harmfulness_score = harmfulness_score
        self.method = method
        self.success_times = success_times
        self.false_times = false_times

    def to_dict(self):
        return {
            "mutation": self.mutation,
            "full_query": self.full_query,
            "pre_query": self.pre_query,
            "response": self.response,
            "harmfulness_score": self.harmfulness_score,
            "method": self.method,
            "success_times": self.success_times,
            "false_times": self.false_times,
        }

    def __repr__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4)