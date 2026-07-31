from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
for tok in ["<|im_end|>", "<|end_of_text|>", "<end_of_turn>", "<|im_start|>"]:
    tid1 = tokenizer.convert_tokens_to_ids(tok)
    tid2 = tokenizer.encode(tok, add_special_tokens=False)
    print(f"Token '{tok}': convert_tokens_to_ids={tid1}, encode={tid2}")
