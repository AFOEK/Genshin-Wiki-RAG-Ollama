import time
import requests
import logging
import random
from urllib.parse import urlsplit, urlunsplit
from collections import deque
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from requests.exceptions import RequestException, Timeout, ConnectionError

log = logging.getLogger(__name__)

_RETRY_STATUSES = {429, 500, 502, 503, 504}

def normalize_url(u: str) -> str:
    parts = urlsplit(u)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

def sleep_backoff(attempt: int, base: float = 1.0, cap: float = 60.0) -> None:
    delay = min(cap, base * (2 ** attempt))
    delay *= (0.7 + random.random() * 0.6)
    time.sleep(delay)

def same_site(url: str, base_url: str) -> bool:
    return urlparse(url).netloc == urlparse(base_url).netloc

def soup_text_fallback(node) -> str:
    try:
        txt = node.get_text("\n", strip=True)
    except Exception:
        txt = str(node)

    lines = [l.strip() for l in (txt or "").splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)

def extract_links(html: str, base: str):
    soup = BeautifulSoup(html, "lxml")
    for a in soup.select("a[href]"):
        href = a.get("href")
        if not href:
            continue
        u = urljoin(base, href)
        parts = urlsplit(u)
        if parts.scheme not in ("http", "https"):
            continue
        u = normalize_url(u)
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
    try:
        return md(str(main))
    except RecursionError:
        log.warning("[WARN] html_to_text: RecursionError in markdownify; falling back to get_text()")
        return soup_text_fallback(main)
    except Exception as e:
        log.warning("[WARN] html_to_text: markdownify failed (%s); falling back to get_text()", type(e).__name__)
        return soup_text_fallback(main)

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

        SKIP_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp",
                    ".svg", ".pdf", ".zip", ".mp4", ".mp3",
                    ".ico", ".css", ".js", ".woff", ".woff2")
        
        url = normalize_url(url)
        path = urlsplit(url).path.lower()
        if path.endswith(SKIP_EXT):
            seen.add(url)
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
                if retries[url] <= 10:
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
            ct = (r.headers.get("Content-Type") or "").lower()
            if ("text/html" not in ct) and ("application/xhtml+xml" not in ct):
                log.info("Skip non-HTML content-type=%s url=%s", ct, url)
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
            link = normalize_url(link)
            path = urlsplit(link).path.lower()
            if path.endswith(SKIP_EXT):
                continue
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