import requests
import time
import logging
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import quote

log = logging.getLogger(__name__)

def list_allpages(api: str, limit: int = 100, namespace: int = 0):
    cont = None
    while True:
        params = {
            "action": "query",
            "format":"json",
            "list":"allpages",
            "apnamespace":str(namespace),
            "aplimit":str(limit)
        }
        if cont:
            params.update(cont)

        r = requests.get(api, params=params, timeout=60)
        r.raise_for_status()
        data =r.json()

        pages = data.get("query", {}).get("allpages", [])
        for p in pages:
            yield p["title"]

        cont = data.get("continue")
        if not cont:
            break

def fandom_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.select("script, style, noscript, .reference, .mw-editsection"):
        tag.decompose()
    main = soup.select_one(".mw-parser-output") or soup
    return md(str(main))

def fetch_page_html(api: str, title: str) -> str | None:
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text",
        "disabletoc": "1",
        "disablelimitreport": "1",
        "redirects": "1",
    }
    r = requests.get(api, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    html = data.get("parse", {}).get("text", {}).get("*")
    if not html:
        return None
    return html

def load_fandom_docs(source_cfg: dict, rate_limit_s: float = 1.0, max_pages: int | None = None):
    api = source_cfg["api"]
    ns = int(source_cfg.get("namespace", 0))
    count = 0

    for title in list_allpages(api, namespace=ns):
        html = fetch_page_html(api, title)
        if html:
            text = fandom_html_to_text(html)
            url = f"{api}?title={quote(title)}"
            yield url, title, text

        count += 1
        if max_pages is not None and count >= max_pages:
            break
        time.sleep(rate_limit_s)