import torch
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from datasets import load_dataset, load_from_disk
from trl import DPOTrainer, DPOConfig
import config as ctg


def clean():
    gc.collect()
    torch.cuda.empty_cache()
clean()

def data_preprocess(example):
    example['prompt'] = example['system'] + '\n' + example['question']
    return example


model = AutoModelForCausalLM.from_pretrained(ctg.model_name, device_map='auto')
tokenizer =AutoTokenizer.from_pretrained(ctg.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = ctg.padding_side

dataset = load_dataset(ctg.dataset_name, split='train')
dataset = dataset.map(data_preprocess)
dataset = dataset.remove_columns(['system', 'question'])
# dataset.save_to_disk('data')
# dataset = load_from_disk('data')
print(dataset.to_pandas().head(5))

training_args = DPOConfig(
    # auto_find_batch_size=True,
    per_device_train_batch_size=ctg.per_device_train_batch_size,
    gradient_accumulation_steps=ctg.gradient_accumulation_steps,
    gradient_checkpointing=True,
    learning_rate=ctg.learning_rate,
    # num_train_epochs=ctg.epoch,
    max_steps=ctg.max_steps,
    save_strategy=ctg.save_strategy,
    # save_steps=ctg.save_steps,
    logging_steps=ctg.logging_steps,
    output_dir=ctg.output_dir,
    warmup_steps=ctg.max_steps//4,
    fp16=True,
    report_to=ctg.report_to
)

dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    beta=ctg.beta,
    max_prompt_length=ctg.max_prompt_length,
    max_length=ctg.max_length
)

dpo_trainer.train()

