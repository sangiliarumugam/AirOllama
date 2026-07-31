import requests
import json

target_model = "aratan/qwen3.6-claude-coder-35b-A3b-mlx-Q4KM-abliterated"

print(f"Testing direct Ollama Registry pull for '{target_model}'...")
res = requests.post("http://127.0.0.1:11211/api/pull", json={"name": target_model, "stream": True}, stream=True)

for line in res.iter_lines():
    if line:
        data = json.loads(line.decode())
        print("Progress:", data)
        # Break after testing initial layer download response to avoid downloading 23GB fully in test
        if "Downloading layer" in data.get("status", ""):
            print("✅ Verified layer blob download successfully initiated from registry.ollama.ai!")
            break
