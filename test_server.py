import time
import requests

url_version = "http://127.0.0.1:11211/api/version"
url_ps = "http://127.0.0.1:11211/api/ps"
url_tags = "http://127.0.0.1:11211/api/tags"

print("Testing connection to AirOllama server on port 11211...")

for i in range(5):
    try:
        r_ver = requests.get(url_version, timeout=2)
        r_ps = requests.get(url_ps, timeout=2)
        r_tags = requests.get(url_tags, timeout=2)
        
        print("✅ AirOllama Version Response:", r_ver.json())
        print("✅ AirOllama Process Memory Status:", r_ps.json())
        print("✅ AirOllama Local Models List:", r_tags.json())
        print("\n🎉 ALL ENDPOINTS WORKING PERFECTLY ON PORT 11211!")
        break
    except Exception as e:
        print(f"Waiting for server... ({e})")
        time.sleep(1)
