import requests
import json
import time

def get_ram():
    r = requests.get("http://127.0.0.1:11211/api/ps").json()
    return r["process_ram_mb"]

print("Initial RAM:", get_ram(), "MB")

# Trigger chat load
print("Sending chat request...")
res = requests.post(
    "http://127.0.0.1:11211/api/chat",
    json={"model": "qwen2.5:0.5b", "messages": [{"role": "user", "content": "Hi"}], "stream": False}
)

print("RAM while loaded:", get_ram(), "MB")

# Trigger unload
print("Sending /api/unload request...")
unload_res = requests.post("http://127.0.0.1:11211/api/unload").json()
print("Unload status:", unload_res)

time.sleep(1)
print("RAM after unload:", get_ram(), "MB")
