model_name = "Sharathhebbar24/SSH_355M"
padding_side = "left"
dataset_name = "Intel/orca_dpo_pairs"
lr_scheduler_type="cosine"
learning_rate=2e-5
gradient_accumulation_steps = 2
per_device_train_batch_size = 1
epoch = 1
max_steps = 100
logging_steps=10
output_dir="./models/dpo_v4/"
report_to = 'tensorboard'

save_strategy='steps'
save_steps= 10

max_prompt_length = 512
max_length = 1024
max_new_token=512

beta=0.1

dpo_trained_model_path = '/media/nsl3090-3/hdd1/hujaifa/DPO_V3/models/dpo_v2/checkpoint-500'
sys_prompt = 'You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.'
query = 'Tell me a funny Joke'
