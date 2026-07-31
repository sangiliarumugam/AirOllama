from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float16, device_map="cpu")

msgs = [
    {"role": "user", "content": "My secret code is Alpha-777. Please remember it!"},
    {"role": "assistant", "content": "Got it! I will remember your secret code: Alpha-777."},
    {"role": "user", "content": "What is my secret code?"}
]

prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("Response:", response)
