# LLM DPO Training

Welcome to the DPO Training Module. Here you can train any Huggingaface DPO dataset with any Huggingface Model using Direct Preference Optimization. The purpose of this module is to serve as a template for DPO Training

## Table of Contents

- Iroduction
- Installation
- Selecting LLM
- Selecting Dataset
- Hyper-Parameter Tuning
- DPO Training
- Training Monitoring
- Inference

## Introduction

Direct Preference Optimization (DPO) is an advanced technique used in machine learning and optimization to directly optimize for the preferences or objectives specified by a user or system. Unlike traditional methods that may rely on surrogate objectives or intermediate steps, DPO focuses on optimizing the final performance measure or preference directly. 

## Installation

To get started, you need to set up the Conda environment.

### Step 1: Install Conda

If you haven't already, install Conda from the [Official Anaconda Website](https://www.anaconda.com/products/distribution) and follow the installation instructions.

### Step 2: Create the Conda environment

Once Conda is installed, create a new environment named `llm_module` using the provided `.yml` file and activate that environment:

```bash
conda env create -f environment.yml
conda activate llm_module
```
## Selecting LLM

At first you need to select an Huggingface LLM / local LLM and put it's path in *config.py* file

```python
model_name = "Sharathhebbar24/SSH_355M"
```
Here a finetuned version of GPT2 is used. You can use any model here but you should do DPO training on a model after finetuning it with custom dataset (Supervised Finetuning). So any base model (only pre-trained) shouldn't be used here

## Selecting Dataset

Just like model, you can select any huggingface dataset here. But it have to follow the dataset format of DPO Trainer. DPO Trainer expects 3 fields, 'prompt', 'chosen', 'rejected'. Depending on your dataset you may need to do some pre-processing before the training. The pre-processing done here is specifically for this dataset 'Intel/orca_dpo_pairs'. You may need to change the 'data_preprocess' function in 'train_dpo.py' so it follows the expected format of DPO Trainer

```python
def data_preprocess(example):
    example['prompt'] = example['system'] + '\n' + example['question']
    return example
```

## Hyper-Parameter Tuning
You can change any hyper-parameter of add any new one depending on your use case. Just change the values in 'config.py' according to your need. You can also change the output directory if you want
```python
padding_side = "left"
lr_scheduler_type="cosine"
learning_rate=2e-5
gradient_accumulation_steps = 2
per_device_train_batch_size = 1
epoch = 1
max_steps = 100
logging_steps=10
save_strategy='steps'
save_steps= 10
max_prompt_length = 512
max_length = 1024
max_new_token=512
beta=0.1
output_dir="./models/dpo_v4/"
report_to = 'tensorboard'
```

## DPO Training
After the necessary chnages in model, dataset and hyper-parameters, you can just run the 'train_dpo.py' file
```bash
python train_dpo.py
```
After training the model will be saved in the 'output_dir' specified in the 'config.py' file

## Training Monitoring
You can monitor your models training using tensorboard using this command,
```bash
tensorboard --logdir <your_output_dir>
```
To check my training log, put 'models/dpo_v2' instead of <your_output_dir> in the above command 

## Inference

To Infer your DPO trained model you need to set the path of your trained model, set a system prompt and write your query in the 'config.py' file
```python
dpo_trained_model_path = 'models/dpo_v2/checkpoint-500'
sys_prompt = 'You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.'
query = 'Tell me a funny Joke'
```
After that you can just run 'inference_dpo.py' file.
```bash
python inference_dpo.py
```

