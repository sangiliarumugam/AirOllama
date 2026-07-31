import requests
import json

def test_ram_cap():
    print("Testing chat with 8GB RAM Cap...")
    res = requests.post(
        "http://127.0.0.1:11211/api/chat",
        json={
            "model": "qwen2.5:0.5b",
            "messages": [{"role": "user", "content": "Tell me a short story about a magic clock."}],
            "stream": True,
            "options": {"max_ram_gb": 8}
        },
        stream=True
    )
    full = ""
    for line in res.iter_lines():
        if line:
            data = json.loads(line.decode())
            full += data.get("message", {}).get("content", "")
    print("\nModel Output:\n", full.strip())

test_ram_cap()
