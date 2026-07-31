import requests
import json

def test_web_chat():
    print("Testing Web Search & System Location Grounding...")
    res = requests.post(
        "http://127.0.0.1:11211/api/chat",
        json={
            "model": "qwen2.5:0.5b",
            "messages": [
                {"role": "user", "content": "What is the latest Python 3 release in 2026 and what is my system location?"}
            ],
            "web_search": True,
            "location": "London, Greater London, United Kingdom (Lat: 51.5074, Lon: -0.1278)",
            "stream": True
        },
        stream=True
    )
    
    full_output = ""
    for line in res.iter_lines():
        if line:
            data = json.loads(line.decode())
            c = data.get("message", {}).get("content", "")
            full_output += c
    print("\nModel Response:\n", full_output.strip())

test_web_chat()
