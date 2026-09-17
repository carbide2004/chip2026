import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

model_path = r"C:\Users\HITSZ\.cache\modelscope\models\Qwen--Qwen3-8B\snapshots\master"
lora_path = "saves/qwen3_8b_lora"

tok = AutoTokenizer.from_pretrained(model_path)
m = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
m = PeftModel.from_pretrained(m, lora_path)

with open("data/processed/sft_association.json", "r", encoding="utf-8") as f:
    sft = json.load(f)[1]

instruction = sft["instruction"]
user_input = sft["input"]

conv = [
    {"role": "system", "content": instruction},
    {"role": "user", "content": user_input},
]
text_in = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
inputs = tok([text_in], return_tensors="pt").to("cuda")

out = m.generate(**inputs, max_new_tokens=128, do_sample=False)
resp = tok.decode(out[0][inputs.input_ids.shape[1]:])
print("Generated response:")
print(resp)
