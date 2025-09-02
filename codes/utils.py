import pandas as pd
import json
import pickle
import os

# 读取 JSON 文件的内容
def read_json_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        # 如果文件不存在，则返回一个空列表
        return []

# 读取 PICKLE 文件的内容
def read_pickle_file(filename):
    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        # 如果文件不存在，则返回 None
        return None

# 读取 CSV 文件的内容
def read_csv_file(filename):
    try:
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        # 如果文件不存在，则返回 None
        return None

# 创建空json文件
def create_empty_json(full_path):
    dir_name = os.path.dirname(full_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        if not os.path.exists(full_path):
            with open(full_path, 'w', encoding='utf-8') as json_file:
                json.dump([], json_file, indent=4)
        else:
            print("File is already Exists")

# 创建json文件
def write_json_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def make_hashable(obj):
    if isinstance(obj, list):
        return tuple(make_hashable(i) for i in obj) 
    return obj