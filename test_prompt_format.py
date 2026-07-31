from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
msgs = [
    {"role": "user", "content": "My secret code is Alpha-777. Please remember it!"},
    {"role": "assistant", "content": "Understood! Your secret code is Alpha-777."},
    {"role": "user", "content": "What is my secret code?"}
]

prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("Formatted Prompt:\n", repr(prompt))
