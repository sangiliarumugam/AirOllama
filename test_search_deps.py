import requests

def search_ddg(query):
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    res = requests.get(url, headers=headers, timeout=5)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(res.text, "html.parser")
    results = []
    for a in soup.find_all("a", class_="result__snippet"):
        results.append(a.get_text().strip())
        if len(results) >= 4:
            break
    return results

try:
    res = search_ddg("latest Apple news")
    print("Search Results:", res)
except Exception as e:
    print("Search Error:", e)
