import os
import sys
import json
import pandas as pd

from utils import read_csv_file, read_json_file, read_pickle_file
from experience_state import JailbreakState

def convert_from_csv(input_path, save_path):
    df = read_csv_file(filename=input_path)
    states = []
    if df is None or df.empty:
        print("CSV文件为空或不存在")
        return
    for _, row in df.iterrows():
        mutation = row.get("mutation", [])
        if isinstance(mutation, str):
            mutation = [mutation]
        score = row.get("gpt_score", 1)
        if score == 5:
            success_times = 1
            false_times = 0
        else:
            success_times = 0
            false_times = 1
        state = JailbreakState(
            pre_query=row.get("question", ""),
            full_query=row.get("full_jailquery", ""),
            response=row.get("response", ""),
            harmfulness_score=score,
            mutation=mutation,
            method=row.get("jailbreak_prompt", ""),
            success_times=success_times,
            false_times=false_times
        )
        states.append(state)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump([s.to_dict() for s in states], f, ensure_ascii=False, indent=2)
    print(f"转换完成，结果保存在 {save_path}")

def convert_from_json(input_path, save_path):
    data = read_json_file(filename=input_path)
    states = []
    if isinstance(data, list):
        for d in data:
            mutation = d.get("mutation", [])
            if isinstance(mutation, str):
                mutation = [mutation]
            score = d.get("gpt_score", 1)
            if score == 5:
                success_times, false_times = 1, 0
            else:
                success_times, false_times = 0, 1
            state = JailbreakState(
                pre_query=d.get("question", ""),
                full_query=d.get("full_jailquery", ""),
                response=d.get("response", ""),
                harmfulness_score=score,
                mutation=mutation,
                method=d.get("jailbreak_prompt", ""),
                success_times=success_times,
                false_times=false_times,
            )
            states.append(state)
    elif isinstance(data, dict):
        mutation = data.get("mutation", [])
        if isinstance(mutation, str):
            mutation = [mutation]
        score = data.get("gpt_score", 1)
        if score == 5:
            success_times, false_times = 1, 0
        else:
            success_times, false_times = 0, 1
        state = JailbreakState(
            pre_query=data.get("question", ""),
            full_query=data.get("full_jailquery", ""),
            response=data.get("response", ""),
            harmfulness_score=score,
            mutation=mutation,
            method=data.get("jailbreak_prompt", ""),
            success_times=success_times,
            false_times=false_times,
        )
        states.append(state)
    else:
        print("JSON文件格式不识别")
        return

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump([s.to_dict() for s in states], f, ensure_ascii=False, indent=2)
    print(f"转换完成，结果保存在 {save_path}")

def convert_from_pickle(input_path, save_path):
    data = read_pickle_file(filename=input_path)
    states = []
    if isinstance(data, list):
        for d in data:
            mutation = d.get("mutation", [])
            if isinstance(mutation, str):
                mutation = [mutation]
            score = d.get("gpt_score", 1)
            if score == 5:
                success_times, false_times = 1, 0
            else:
                success_times, false_times = 0, 1
            state = JailbreakState(
                pre_query=d.get("question", ""),
                full_query=d.get("full_jailquery", ""),
                response=d.get("response", ""),
                harmfulness_score=score,
                mutation=mutation,
                method=d.get("jailbreak_prompt", ""),
                success_times=success_times,
                false_times=false_times
            )
            states.append(state)
    elif isinstance(data, dict):
        mutation = data.get("mutation", [])
        if isinstance(mutation, str):
            mutation = [mutation]
        score = data.get("gpt_score", 1)
        if score == 5:
            success_times, false_times = 1, 0
        else:
            success_times, false_times = 0, 1
        state = JailbreakState(
            pre_query=data.get("question", ""),
            full_query=data.get("full_jailquery", ""),
            response=data.get("response", ""),
            harmfulness_score=score,
            mutation=mutation,
            method=data.get("jailbreak_prompt", ""),
            success_times=success_times,
            false_times=false_times
        )
        states.append(state)
    else:
        print("Pickle文件格式不识别")
        return

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump([s.to_dict() for s in states], f, ensure_ascii=False, indent=2)
    print(f"转换完成，结果保存在 {save_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("请按照用法: python transfer.py input_file output_file")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    ext = os.path.splitext(input_path)[1].lower()

    if ext in ['.csv', '.xlsx']:
        convert_from_csv(input_path, output_path)
    elif ext == '.json':
        convert_from_json(input_path, output_path)
    elif ext in ['.pkl', '.pickle']:
        convert_from_pickle(input_path, output_path)
    else:
        print("不支持的文件格式")
        sys.exit(1)