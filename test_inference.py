import time
import requests
import json

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"1. Pulling model '{model_name}' into AirOllama model store...")
r_pull = requests.post("http://127.0.0.1:11211/api/pull", json={"name": model_name, "stream": False})
print("Pull result:", r_pull.json())

print("\n2. Checking updated model list (/api/tags)...")
r_tags = requests.get("http://127.0.0.1:11211/api/tags")
print("Available models:", r_tags.json())

print(f"\n3. Testing layer-by-layer chat streaming on port 11211 with model '{model_name}'...")
payload = {
    "model": model_name,
    "messages": [
        {"role": "user", "content": "What is 15 + 27? Answer in one short sentence."}
    ],
    "stream": True,
    "options": {
        "num_predict": 64,
        "temperature": 0.1
    }
}

r_chat = requests.post("http://127.0.0.1:11211/api/chat", json=payload, stream=True)

print("Streaming token response:")
full_text = ""
for line in r_chat.iter_lines():
    if line:
        data = json.loads(line.decode())
        token = data.get("message", {}).get("content", "")
        full_text += token
        print(token, end="", flush=True)

print("\n\nFull output received:", full_text)

print("\n4. Checking memory & layer monitor status (/api/ps)...")
r_ps = requests.get("http://127.0.0.1:11211/api/ps")
print("Process memory & layer state:", r_ps.json())
