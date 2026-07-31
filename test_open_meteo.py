import requests

def get_open_meteo_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        r = requests.get(url, timeout=5).json()
        cw = r.get("current_weather", {})
        temp = cw.get("temperature")
        wind = cw.get("windspeed")
        code = cw.get("weathercode")
        return f"Exact GPS Current Weather (Lat {lat}, Lon {lon}): Temp: {temp}°C, Wind Speed: {wind} km/h, Weather Code: {code}"
    except Exception as e:
        return ""

print(get_open_meteo_weather(51.5074, -0.1278))
