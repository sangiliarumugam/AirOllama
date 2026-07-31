import requests
import re
import html
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("AirOllama.WebSearch")

def get_server_ip_location() -> str:
    """Fetch location based on server IP address as fallback."""
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

def get_live_weather(location_str: str) -> str:
    """
    Fetch real-time live weather using Open-Meteo GPS coordinates and wttr.in.
    """
    if not location_str or location_str == "Location unavailable":
        location_str = get_server_ip_location()


    weather_text = ""
    
    # Try parsing lat/lon coordinates from location string
    lat_match = re.search(r"Lat:\s*([-\d.]+)", location_str)
    lon_match = re.search(r"Lon:\s*([-\d.]+)", location_str)
    
    if lat_match and lon_match:
        try:
            lat = float(lat_match.group(1))
            lon = float(lon_match.group(1))
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            res = requests.get(url, timeout=4).json()
            cw = res.get("current_weather", {})
            if "temperature" in cw:
                temp_c = cw["temperature"]
                temp_f = round(temp_c * 9/5 + 32, 1)
                wind = cw.get("windspeed", 0)
                code = cw.get("weathercode", 0)
                code_map = {
                    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                    45: "Foggy", 51: "Light drizzle", 61: "Slight rain", 63: "Moderate rain",
                    65: "Heavy rain", 71: "Slight snow", 80: "Rain showers", 95: "Thunderstorm"
                }
                desc = code_map.get(code, "Clear/Cloudy")
                weather_text += f"🌡️ Live GPS Weather ({location_str}): {desc}, {temp_c}°C ({temp_f}°F), Wind: {wind} km/h.\n"
        except Exception as e:
            logger.warning(f"Open-Meteo weather check error: {e}")

    # Fallback/Additional wttr.in check
    try:
        city = location_str.split(",")[0].split("(")[0].strip()
        if city and city != "Location unavailable":
            w_url = f"https://wttr.in/{requests.utils.quote(city)}?format=%C+%t+%w+%h"
            w_res = requests.get(w_url, headers={"User-Agent": "curl/7.68.0"}, timeout=4)
            if w_res.status_code == 200 and w_res.text.strip():
                weather_text += f"📍 City Forecast ({city}): {w_res.text.strip()}\n"
    except Exception as e:
        logger.warning(f"wttr.in check error: {e}")

    return weather_text.strip()


def perform_web_search(query: str, location_str: Optional[str] = None, max_results: int = 4) -> List[Dict[str, str]]:
    """
    Perform a real-time web search. If location_str is provided and relevant,
    augment the query with user location.
    """
    if not query or len(query.strip()) < 2:
        return []

    if not location_str or location_str == "Location unavailable":
        location_str = get_server_ip_location()

    search_query = query.strip()


    # Augment location into query if query doesn't specify a place
    if location_str and location_str.strip() and location_str != "Location unavailable":
        city_name = location_str.split(",")[0].split("(")[0].strip()
        keywords_requiring_loc = ["weather", "temperature", "forecast", "restaurant", "hotel", "news", "event", "near me", "today", "place", "store", "shop", "traffic"]
        if any(k in query.lower() for k in keywords_requiring_loc) and city_name.lower() not in query.lower():
            search_query = f"{query} in {city_name}"

    logger.info(f"Executing web search query: '{search_query}' (Original: '{query}')")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(search_query)}"
    try:
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code != 200:
            return []

        html_content = resp.text
        result_blocks = re.findall(r'<div class="result__body">(.*?)</div>\s*</div>', html_content, re.DOTALL)
        
        results = []
        for block in result_blocks[:max_results]:
            title_match = re.search(r'<a class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', block, re.DOTALL)
            snippet_match = re.search(r'<a class="result__snippet[^>]*>(.*?)</a>', block, re.DOTALL)

            if snippet_match:
                snip_text = html.unescape(re.sub(r'<[^>]+>', '', snippet_match.group(1))).strip()
                title_text = ""
                link_url = ""
                if title_match:
                    link_url = title_match.group(1)
                    title_text = html.unescape(re.sub(r'<[^>]+>', '', title_match.group(2))).strip()

                if snip_text:
                    results.append({
                        "title": title_text or "Web Result",
                        "snippet": snip_text,
                        "url": link_url
                    })

        if not results:
            snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html_content, re.DOTALL)
            for s in snippets[:max_results]:
                clean_s = html.unescape(re.sub(r'<[^>]+>', '', s)).strip()
                if clean_s:
                    results.append({"title": "Web Search Snippet", "snippet": clean_s, "url": ""})

        return results

    except Exception as e:
        logger.error(f"Web search execution error: {e}")
        return []


def format_search_context(query: str, results: List[Dict[str, str]], weather_info: str = "") -> str:
    """Format search & weather results into system prompt context."""
    context = ""

    if weather_info:
        context += f"\n[🌤️ Real-Time Live Weather Data]\n{weather_info}\n[End of Weather Data]\n"

    if results:
        context += f"\n[🌐 Live Web Search Results for: '{query}']\n"
        for i, r in enumerate(results, 1):
            context += f"{i}. {r['title']}\n   Snippet: {r['snippet']}\n"
            if r.get('url'):
                context += f"   Source: {r['url']}\n"
        context += "[End of Web Search Results]\n"

    return context
