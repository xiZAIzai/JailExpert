# JailExpert

![method](./imgs/framework.png)

This is the official repository for our paper [Stand on The Shoulders of Giants: Building JailExpert from Previous Attack Experience]().

we propose the **JailExpert**, an automated jailbreak framework, which is the first to achieve a formal representation of experience structure, group experiences based on semantic drift, and support the dynamic up dating of the experience pool. Extensive experiments demonstrate that JailExpert significantly improves both attack effectiveness and efficiency. Compared to the current state-of the-art black-box jailbreak methods, JailExpert achieves an average increase of **17%** in attack success rate and **2.7 times** improvement in attack efficiency. Please refer to our [paper]() for more details.

## News

- 🎉🎉🎉 21 / 08 / 2025, Our Paper `Stand on The Shoulders of Giants: Building JailExpert from Previous Attack Experience` is accepted by EMNLP 2025 Main (oral) !

## Quick Start

### Setup

The attack implementation of JailExpert require dependencies can be installed by:

```shell
pip install -r requirements.txt
```

### Experience Pool Initialization

We require [EasyJailbreak](https://github.com/EasyJailbreak/EasyJailbreak) for the experience pool initialization, you guys can follow up the implementation of it and execute the methods including: ReNeLLM, GPTFuzzer, JailBroken and CodeChameleon. And use the use the **transfer.py** to transfer all attack results to the pre-defined jailbreak experience pattern:

```shell
   python transfer.py input_file output_file

   # which the inputed attack results file support xlsx, csv, pkl, pickle, json
   # single attack obj in each type of attack results should contains keys include: 
   '''
        "mutation": [], # sampled mutation list
        "full_query": "", # final jailbreak prompt
        "pre_query": "", # original question(query)
        "response": "", # LLM response
        "harmfulness_score": 5, # harmful score evaluated by Judge LLM
        "method": "", # jailbreak prompt
        "success_times": 1, # default to 1
        "false_times": 0,
   '''
```


#### Running the attack Script

We build upon the GCG attack framework and integrate our method. Use the following commands to run the search:

```shell
bash 
```

Remember to change the running config in `./scripts/configs`

#### Post Processing

We provide scripts to extract adversarial prompts from log files and combine them with test questions and model chat templates:

```shell
python
```

### Generation

After searching for the adversarial attack prompt, we provide a script supporting batch generation to get the response of the model. Remember to change the model/tokenizer path and the input/output path.

```shell
cd
```

### Evaluation

We support evaluation using `AISafetyLab`. Run the evaluation script as follows:

```shell
cd
```

## Citation
