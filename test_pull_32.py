import requests
import json

def test_pull():
    print("Testing model pull with 32 parallel workers...")
    res = requests.post(
        "http://127.0.0.1:11211/api/pull",
        json={"name": "qwen2.5:0.5b", "stream": True},
        stream=True
    )
    for line in res.iter_lines():
        if line:
            data = json.loads(line.decode())
            status = data.get("status", "")
            if status and "32" in status or "completed" in status.lower():
                print("Pull status:", status)

test_pull()
