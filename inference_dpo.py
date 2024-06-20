from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import config as ctg

def generate_prompt(query, sys_prompt = None):
    if sys_prompt is None:
        sys_prompt = ctg.sys_prompt
    return sys_prompt+'\n'+query
    

def llm_initialization():
    # model = AutoModelForCausalLM.from_pretrained(ctg.dpo_trained_model_path,device_map='auto')
    # model.eval()
    # tokenizer = AutoTokenizer.from_pretrained(ctg.dpo_trained_model_path)
    model = AutoModelForCausalLM.from_pretrained(ctg.model_name,device_map='auto')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ctg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = ctg.padding_side
    return model, tokenizer

def generate(model, tokenizer, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = tokenizer(prompt, truncation=True, max_length=ctg.max_length, padding=False, return_tensors='pt').to(device)
    input_ids = {
        'input_ids' : input_ids['input_ids'],
        'attention_mask' : input_ids['attention_mask'],
    }

    outputs = model.generate(
        max_new_tokens=ctg.max_new_token, 
        do_sample=True,
        temperature=0.9,#0.7 
        top_p=0.75, 
        top_k=40,         
        no_repeat_ngram_size=2,
        **input_ids
    )
    outputs = outputs[0].tolist()

    decoded = tokenizer.decode(outputs)
    sentinel = prompt[-10:]
    sentinelLoc = decoded.find(sentinel)
    if sentinelLoc >= 0:
        response = decoded[sentinelLoc+len(sentinel):]
        return response
    else:
        return 'Warning: Expected prompt template to be emitted.  Ignoring output.'

if __name__ == '__main__':
    model, tokenizer = llm_initialization()
    
    prompt=generate_prompt(ctg.query)
    print('PROMPT::::', prompt)
    response = generate(model, tokenizer, prompt)
    print(prompt)
