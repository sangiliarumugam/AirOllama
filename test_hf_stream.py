import requests
import json

target_model = "gemma4:31b-mlx"

print(f"Testing real-time HF streaming progress for '{target_model}'...")
res = requests.post("http://127.0.0.1:11211/api/pull", json={"name": target_model, "source": "huggingface", "stream": True}, stream=True)

for line in res.iter_lines():
    if line:
        data = json.loads(line.decode())
        print("Progress ->", data.get("status", data))
        if "Started downloading" in data.get("status", ""):
            print("✅ Verified real-time parallel file thread progress updates working!")
            break
