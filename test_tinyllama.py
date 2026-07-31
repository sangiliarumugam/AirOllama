import requests

model_tag = "tinyllama"
print(f"1. Pulling Ollama repo tag '{model_tag}'...")
r_pull = requests.post("http://127.0.0.1:11211/api/pull", json={"name": model_tag, "stream": False})
print("Pull result:", r_pull.json())

print(f"\n2. Running Chat completion on '{model_tag}'...")
r_chat = requests.post("http://127.0.0.1:11211/api/chat", json={
    "model": model_tag,
    "messages": [{"role": "user", "content": "Write a 1-sentence joke about a programmer."}],
    "stream": False
})
print("Response:", r_chat.json())
