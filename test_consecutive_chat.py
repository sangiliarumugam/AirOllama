import requests
import json

def send_chat(prompt):
    print(f"\nUser: {prompt}")
    res = requests.post(
        "http://127.0.0.1:11211/api/chat",
        json={
            "model": "qwen2.5:0.5b",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        },
        stream=True
    )

    is_first = True
    for line in res.iter_lines():
        if line:
            data = json.loads(line.decode())
            content = data.get("message", {}).get("content", "")
            if content:
                if is_first:
                    print("Stream Output -> ", end="")
                    is_first = False
                print(content, end="", flush=True)
    print()

print("Testing consecutive chat requests with same model...")
send_chat("Prompt 1: What color is the sky?")
send_chat("Prompt 2: What color is grass?")
