import requests
import json

target_model = "smollm:135m"

print(f"Testing Docker-style multi-threaded parallel pull for '{target_model}'...")
res = requests.post("http://127.0.0.1:11211/api/pull", json={"name": target_model, "stream": True}, stream=True)

for line in res.iter_lines():
    if line:
        data = json.loads(line.decode())
        print("Progress ->", data.get("status", data))
