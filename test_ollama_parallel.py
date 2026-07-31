import requests
import json

target_model = "qwen2.5:0.5b"

print(f"Testing direct Ollama multi-threaded layer pull for '{target_model}'...")
res = requests.post("http://127.0.0.1:11211/api/pull", json={"name": target_model, "stream": True}, stream=True)

for line in res.iter_lines():
    if line:
        data = json.loads(line.decode())
        print("Progress ->", data.get("status", data))
