import requests

def get_live_weather(location_str):
    try:
        # Clean location_str to city name (e.g. "London, UK (Lat...)" -> "London")
        city = location_str.split(",")[0].split("(")[0].strip()
        url = f"https://wttr.in/{requests.utils.quote(city)}?format=%C+%t+%w+%h"
        resp = requests.get(url, headers={"User-Agent": "curl/7.68.0"}, timeout=5)
        if resp.status_code == 200:
            return f"Current live weather in {city}: {resp.text.strip()}"
    except Exception as e:
        print("Weather API error:", e)
    return ""

print(get_live_weather("London, United Kingdom"))
print(get_live_weather("San Francisco, CA"))
