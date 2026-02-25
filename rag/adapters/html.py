import time
import requests
import logging
import random
from collections import deque
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from requests.exceptions import RequestException, Timeout, ConnectionError

log = logging.getLogger(__name__)

_RETRY_STATUSES = {429, 500, 502, 503, 504}

def sleep_backoff(attempt: int, base: float = 1.0, cap: float = 60.0) -> None:
    delay = min(cap, base * (2 ** attempt))
    delay *= (0.7 + random.random() * 0.6)
    time.sleep(delay)

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

def crawl_site(base_url: str, seeds: list[str], deny_url, rate_limit_s: float = 1.0, max_pages: int | None = 2000):
    q = deque(seeds)
    seen: set[str] = set()
    retries: dict[str, int] = {}
    session = requests.Session()

    while q and (max_pages is None or len(seen) < max_pages):
        url = q.popleft()
        if url in seen:
            continue

        if not same_site(url, base_url):
            continue
        if deny_url and deny_url.search(url):
            continue

        attempt = retries.get(url, 0)

        try:
            r = session.get(
                url,
                timeout=60,
                headers={"User-Agent": "GenshinRAG/1.0 (personal research)"},
            )

            if r.status_code in _RETRY_STATUSES:
                retries[url] = attempt + 1
                if retries[url] <= 6:
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        time.sleep(min(int(ra), 120))
                    else:
                        sleep_backoff(attempt)
                    q.append(url)
                    continue
                else:
                    log.warning("Giving up url=%s after %d retries status=%d", url, retries[url], r.status_code)
                    seen.add(url)
                    time.sleep(rate_limit_s)
                    continue

            if r.status_code >= 400:
                log.warning("HTTP %d url=%s", r.status_code, url)
                seen.add(url)
                time.sleep(rate_limit_s)
                continue

            html = r.text

        except (Timeout, ConnectionError) as e:
            retries[url] = attempt + 1
            if retries[url] <= 10:
                log.warning("Network error (%s) url=%s retry=%d/10", type(e).__name__, url, retries[url])
                sleep_backoff(attempt)
                q.append(url)
                continue
            log.warning("Giving up url=%s after %d retries (%s)", url, retries[url], type(e).__name__)
            seen.add(url)
            time.sleep(rate_limit_s)
            continue

        except RequestException as e:
            retries[url] = attempt + 1
            if retries[url] <= 10:
                log.warning("RequestException url=%s retry=%d/10 err=%s", url, retries[url], e)
                sleep_backoff(attempt)
                q.append(url)
                continue
            log.warning("Giving up url=%s after %d retries (RequestException)", url, retries[url])
            seen.add(url)
            time.sleep(rate_limit_s)
            continue

        seen.add(url)

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
            log.exception("Failed to extract title url=%s", url)

        yield url, title, text
        time.sleep(rate_limit_s)