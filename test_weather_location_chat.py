import requests
import json

def test_weather():
    print("Testing 'what is the weather today?' with location & live weather API...")
    res = requests.post(
        "http://127.0.0.1:11211/api/chat",
        json={
            "model": "qwen2.5:0.5b",
            "messages": [
                {"role": "user", "content": "what is the weather today?"}
            ],
            "web_search": True,
            "location": "San Francisco, California, US (Lat: 37.7749, Lon: -122.4194)",
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

test_weather()
