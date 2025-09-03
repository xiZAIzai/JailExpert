import argparse

def attack_parser():
    """
    Parser for attack module parameters.
    """
    parser = argparse.ArgumentParser(
        description="JailExpert Attack Configuration"
    )
    
    # Experiment settings related to attack
    exp_group = parser.add_argument_group("Experiment Settings", "Attack experiment options")
    exp_group.add_argument("--experiment", type=str, default="main", choices=["main", "ablation"],
                           help="Type of experiment (e.g., main, ablation)")
    exp_group.add_argument("--experience_name", type=str, default="llama-2",
                           choices=["llama-2", "llama-3", "gpt-4-turbo", "gpt-4-0613", "gemini-1.5-pro",
                                    "llama-2-13b", "gpt-3.5-turbo-1106", "llama-2-unlearned-Full"],
                           help="Name of the experience pool")
    exp_group.add_argument("--experience_type", type=str, default="renellm",
                           choices=["full", "renellm", "GPTFuzzer", "codeChameleon", "jailbroken"],
                           help="Type of experience used")
    exp_group.add_argument("--strategy", type=str, default="single",
                           choices=["baseline", "random", "no_dynamic", "no_simappend", "single"],
                           help="Attack strategy selection")
    exp_group.add_argument("--top_k", type=float, default=1.0,
                           help="Top K value for retrieval")
    
    # Model parameters for attack
    model_group = parser.add_argument_group("Model Parameters", "Configuration for target, attack, and evaluation models")
    model_group.add_argument("--target_model", type=str, default="Llama-2-7b-chat-hf",
                             choices=["Llama-2-13b-chat-hf", "Llama-2-7b-chat-hf", "Llama-2-7b-chat-hf-unlearned",
                                      "Meta-Llama-3-8B-Instruct", "gemini-1.5-pro", "gpt-3.5-turbo-1106",
                                      "gpt-4-turbo", "gpt-4", "gpt-oss-20b"],
                             help="Target LLM for jailbreak attack")
    model_group.add_argument("--target_api", type=str, default="sk-",
                             help="API key for accessing the target model")
    model_group.add_argument("--attack_api", type=str, default="sk-",
                             help="API key for the attack model")
    model_group.add_argument("--eval_api", type=str, default="sk-",
                             help="API key for the evaluation model")
    
    # Device configuration
    device_group = parser.add_argument_group("Device Settings", "Hardware configuration")
    device_group.add_argument("--device", type=str, default="cuda:2",
                              help="CUDA device identifier (e.g., cuda:0)")
    
    # File paths for attack results
    path_group = parser.add_argument_group("File Paths", "Directories and file paths for attack")
    path_group.add_argument("--target_model_path", type=str, default="/science/llms",
                            help="Base directory for target models")
    path_group.add_argument("--experience_data_path", type=str,
                            default="/science/wx/research/JailExpert/experiments/data/experience",
                            help="Path to experience pool data")
    path_group.add_argument("--result_dir", type=str, default="JailExpert_results",
                            help="Directory to store experiment results")
    
    return parser


def index_parser():
    """
    Parser for ExperienceIndex module parameters.
    """
    parser = argparse.ArgumentParser(
        description="JailExpert Experience Index Configuration"
    )
    
    index_group = parser.add_argument_group("Index Settings", "Parameters for building experience index")
    index_group.add_argument("--scalar_weight", type=float, default=0.1,
                             help="Weight for scalar features in index")
    index_group.add_argument("--semantic_weight", type=float, default=0.9,
                             help="Weight for semantic features in index")
    index_group.add_argument("--max_clusters", type=int, default=10,
                             help="Maximum clusters for KMeans clustering")
    index_group.add_argument("--experience_data_path", type=str,
                             default="/science/wx/research/JailExpert/experiments/data/experience",
                             help="Path to experience pool data used for index building")
                             
    return parser


if __name__ == "__main__":
    # Example usage: choose one parser to test.
    # For attack module configuration:
    attack_args = attack_parser().parse_args()
    print("Attack Configuration:")
    print(attack_args)
    
    # For index configuration:
    index_args = index_parser().parse_args()
    print("\nIndex Configuration:")
    print(index_args)