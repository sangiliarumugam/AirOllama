import requests
import json

def test_hi():
    print("Testing 'hi' prompt with 32GB RAM cap...")
    res = requests.post(
        "http://127.0.0.1:11211/api/chat",
        json={
            "model": "gemma4:31b-mlx",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "options": {"max_ram_gb": 32}
        },
        stream=True
    )
    full = ""
    for line in res.iter_lines():
        if line:
            data = json.loads(line.decode())
            c = data.get("message", {}).get("content", "")
            print(c, end="", flush=True)
            full += c
    print("\n--- Done ---")

if __name__ == "__main__":
    test_hi()
