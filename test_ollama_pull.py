import requests
import json

def test_ollama_registry_pull():
    print("Testing direct pull from Ollama Registry (registry.ollama.ai)...")
    res = requests.post(
        "http://127.0.0.1:11211/api/pull",
        json={"name": "qwen2.5:0.5b", "source": "ollama", "stream": True},
        stream=True
    )
    for line in res.iter_lines():
        if line:
            data = json.loads(line.decode())
            print("Ollama Registry Status:", data.get("status", data.get("error", "")))
            if "success" in data.get("status", "").lower() or "completed" in data.get("status", "").lower():
                break

test_ollama_registry_pull()
