import requests
import re
import html

def web_search(query: str, max_results: int = 5):
    """Perform real-time web search without external dependencies."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    try:
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code != 200:
            return []
        
        snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', resp.text, re.DOTALL)
        titles = re.findall(r'<a class="result__url[^>]*>(.*?)</a>', resp.text, re.DOTALL)
        
        results = []
        for s in snippets[:max_results]:
            clean_s = re.sub(r'<[^>]+>', '', s)
            clean_s = html.unescape(clean_s).strip()
            if clean_s:
                results.append(clean_s)
        return results
    except Exception as e:
        print("Web search error:", e)
        return []

results = web_search("latest Python release version 2026")
print("Web Search Results:\n", results)
