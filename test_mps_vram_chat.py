import requests
import json
import time

def test_mps():
    print("Testing chat with Metal GPU transfer & VRAM tracking...")
    res = requests.post(
        "http://127.0.0.1:11211/api/chat",
        json={
            "model": "qwen2.5:0.5b",
            "messages": [{"role": "user", "content": "Tell me a story about Apple Metal GPU."}],
            "stream": True
        },
        stream=True
    )
    full = ""
    for line in res.iter_lines():
        if line:
            data = json.loads(line.decode())
            full += data.get("message", {}).get("content", "")
    
    ps = requests.get("http://127.0.0.1:11211/api/ps").json()
    print("\nModel Output:\n", full.strip()[:200], "...")
    print("\nServer Memory Stats:\n", json.dumps(ps, indent=2))

test_mps()
