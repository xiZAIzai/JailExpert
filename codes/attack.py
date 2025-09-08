import json
import random
import logging

import pandas as pd

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from attack_utils import convert_to_jailbreak_state
from utils import create_empty_json
from model_utils import OpenaiModel, OpenaiModel_Attack

from parser import attack_parser
from jailExpert import JailExpertWithCluster

logging.basicConfig(level=logging.INFO, format='%(threadName)s: %(message)s')
logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
random.seed(2025)

if __name__ == '__main__':
    args = attack_parser().parse_args()
    print("Attack Configuration:")
    print(args)

    device_index = int(args.device.split(":")[-1])
    device = args.device
    torch.cuda.set_device(device_index)
    print("Current CUDA device index:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(device_index))
    random.seed(2025)

    experience_name = args.experience_name
    if "llama" in args.target_model.lower() or "oss" in args.target_model.lower():
        target_model_path = f'/science/llms/{args.target_model}'
        target_model_name = target_model_path.split('/')[-1]
        target_model = AutoModelForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.bfloat16)
        target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)
        generation_config = {"max_new_tokens": 256, "do_sample": False}
        target_model.to(device)
        from model_utils import HuggingfaceModel
        target_model = HuggingfaceModel(target_model, target_tokenizer, is_attack=False, model_name=target_model_name, generation_config=generation_config)
    else:
        target_model_name = args.target_model
        generation_config_1 = {"max_tokens": 600, "temperature": 1.0}
        target_model = OpenaiModel_Attack(model_name=target_model_name, api_keys=args.targe_api, generation_config=generation_config_1, url=args.base_url)

    attack_model_name = 'gpt-3.5-turbo'
    generation_config_1 = {"max_tokens": 4000, "temperature": 1.0}
    attack_model = OpenaiModel(model_name=attack_model_name, api_keys=args.attack_api, generation_config=generation_config_1, url=args.base_url)
    
    eval_model_name = 'gpt-4-turbo'
    generation_config_1 = {"max_tokens": 600, "temperature": 0}
    eval_model = OpenaiModel(model_name=eval_model_name, api_keys=args.eval_api, generation_config=generation_config_1, url=args.base_url)

    experience_pool = json.load(open(f"../experience/{experience_name}/{args.experience_type}_{args.strategy}.json"))
    experience_pool = convert_to_jailbreak_state(experience_pool)
    logging.info(f"{args.experience_type} Experiences Converted Successfully!")
    manager = JailExpertWithCluster(model=target_model, tokenizer=None, attack_model=attack_model, eval_model=eval_model,
                                    experience_pool=experience_pool, target_model_name=target_model_name)
    
    if args.experiment == "ablation":
        attack_data = pd.read_csv(f"../data/harmful_behaviors_subset.csv").goal.tolist()
    else:
        attack_data = pd.read_csv(f"../data/strongreject_small_dataset.csv").forbidden_prompt.tolist() + pd.read_csv(f"../data/harmful_behaviors_subset.csv").goal.tolist()
    save_path = args.result_dir
    create_empty_json(save_path)
    manager.run(args, attack_data, save_path=save_path, experience_name=experience_name, target_name=target_model_name, top_k=args.top_k, strategy=args.strategy, multi_turn=False)