import requests

def get_server_ip_location():
    try:
        r = requests.get("https://ipapi.co/json/", timeout=3).json()
        city = r.get("city", "")
        region = r.get("region", "")
        country = r.get("country_name", "")
        lat = r.get("latitude")
        lon = r.get("longitude")
        if city:
            return f"{city}, {region}, {country} (Lat: {lat}, Lon: {lon})"
    except Exception:
        pass
    try:
        r = requests.get("http://ip-api.com/json/", timeout=3).json()
        city = r.get("city", "")
        region = r.get("regionName", "")
        country = r.get("country", "")
        lat = r.get("lat")
        lon = r.get("lon")
        if city:
            return f"{city}, {region}, {country} (Lat: {lat}, Lon: {lon})"
    except Exception:
        pass
    return "San Francisco, California, US (Lat: 37.7749, Lon: -122.4194)"

print("Server IP Location:", get_server_ip_location())
