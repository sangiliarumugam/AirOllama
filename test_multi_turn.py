import requests
import json

def chat_turn(messages):
    res = requests.post(
        "http://127.0.0.1:11211/api/chat",
        json={
            "model": "qwen2.5:0.5b",
            "messages": messages,
            "stream": True
        },
        stream=True
    )
    full_resp = ""
    for line in res.iter_lines():
        if line:
            data = json.loads(line.decode())
            c = data.get("message", {}).get("content", "")
            if not c.startswith("⌛ Preparing model"):
                full_resp += c
    return full_resp.strip()

history = []

# Turn 1
prompt1 = "My secret code is Alpha-777. Please remember it!"
print(f"User: {prompt1}")
history.append({"role": "user", "content": prompt1})
resp1 = chat_turn(history)
print(f"Assistant: {resp1}\n")
history.append({"role": "assistant", "content": resp1})

# Turn 2
prompt2 = "What is my secret code?"
print(f"User: {prompt2}")
history.append({"role": "user", "content": prompt2})
resp2 = chat_turn(history)
print(f"Assistant: {resp2}\n")
