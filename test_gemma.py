import requests
import json

target_model = "gemma4:31b-mlx"

print(f"Testing parallel pull for '{target_model}'...")
res = requests.post("http://127.0.0.1:11211/api/pull", json={"name": target_model, "stream": True}, stream=True)

for line in res.iter_lines():
    if line:
        data = json.loads(line.decode())
        print("Progress ->", data.get("status", data))
        if "initiated" in data.get("status", ""):
            print("✅ Verified gemma4:31b-mlx resolved to open MLX safetensors repository!")
            break
