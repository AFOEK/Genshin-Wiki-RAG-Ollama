import time
import requests
import logging
from collections import deque
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from markdownify import markdownify as md

log = logging.getLogger(__name__)

def same_site(url: str, base_url: str) -> bool:
    return urlparse(url).netloc == urlparse(base_url).netloc

def extract_links(html: str, base: str):
    soup = BeautifulSoup(html, "lxml")
    for a in soup.select("a[href]"):
        href = a.get("href")
        if not href:
            continue
        u = urljoin(base, href)
        u = u.split("#", 1)[0]
        yield u

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    main = (
        soup.select_one("main")
        or soup.select_one("article")
        or soup.select_one("#content")
        or soup.select_one(".mw-parser-output")
        or soup.body
        or soup
    )
    return md(str(main))

def crawl_site(base_url: str, seeds: list[str], deny_url, rate_limit_s: float=1.0, max_pages: int =2000):
    q = deque(seeds)
    seen = set()

    while q and (max_pages is None or len(seen) < max_pages):
        url = q.popleft()
        if url in seen:
            continue
        seen.add(url)

        if not same_site(url, base_url):
            continue
        
        if deny_url and deny_url.search(url):
            continue

        try:
            r = requests.get(url, timeout=60, headers={"User-Agent":"GenshinRAG/1.0 (personal research)"})
            if r.status_code >= 400:
                time.sleep(rate_limit_s)
                continue
            html = r.text
        except Exception:
            time.sleep(rate_limit_s)
            continue

        for link in extract_links(html, url):
            if link not in seen and same_site(link, base_url):
                q.append(link)

        text = html_to_text(html)
        title = url
        try:
            soup = BeautifulSoup(html, "lxml")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
        except Exception:
            log.exception(f"Failed to extract {title}, skipping")
            pass

        yield url, title, text
        time.sleep(rate_limit_s)
