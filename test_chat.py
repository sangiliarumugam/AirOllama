import requests
import json

print("Testing chat generation stream...")
res = requests.post(
    "http://127.0.0.1:11211/api/chat",
    json={
        "model": "qwen2.5:0.5b",
        "messages": [{"role": "user", "content": "Say hello in one word"}],
        "stream": True
    },
    stream=True
)

for line in res.iter_lines():
    if line:
        data = json.loads(line.decode())
        content = data.get("message", {}).get("content", "")
        if content:
            print(content, end="", flush=True)

print("\nDone testing chat!")
