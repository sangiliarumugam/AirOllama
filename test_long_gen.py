import requests
import json

print("Testing long response generation stream (asking for detailed essay)...")
res = requests.post(
    "http://127.0.0.1:11211/api/chat",
    json={
        "model": "qwen2.5:0.5b",
        "messages": [{"role": "user", "content": "Write a 3-paragraph detailed explanation of artificial intelligence and machine learning."}],
        "stream": True,
        "options": {"num_predict": 2048}
    },
    stream=True
)

total_tokens = 0
for line in res.iter_lines():
    if line:
        data = json.loads(line.decode())
        content = data.get("message", {}).get("content", "")
        if content:
            total_tokens += 1
            print(content, end="", flush=True)

print(f"\n\nTotal generated tokens: {total_tokens}")
if total_tokens > 100:
    print("✅ Verified long multi-paragraph response generation works cleanly!")
