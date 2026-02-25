import requests
import time
import random
import logging
from typing import Any, Optional
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import quote
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError

log = logging.getLogger(__name__)
_RETRY_STATUSES = {429, 500, 502, 503, 504}

def sleep_backoff(attemp: int, base: float = 1.0, cap: float = 60.0) -> None:
    delay = min(cap, base * (2 ** attemp))
    delay *= (0.75 + random.random() * 0.6)
    time.sleep(delay)

def get_json_with_retry(
        session: requests.Session,
        url: str,
        *,
        params: dict[str, Any],
        timeout: float = 60.0,
        max_retries: int = 10
) -> Optional[dict[str, Any]]:
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code in _RETRY_STATUSES:
                # honor Retry-After if present (common on 429)
                ra = r.headers.get("Retry-After")
                if ra and ra.isdigit():
                    time.sleep(min(int(ra), 120))
                else:
                    sleep_backoff(attempt)
                continue

            if r.status_code >= 400:
                log.warning("HTTP %s for %s params=%s", r.status_code, url, params)
                return None

            return r.json()

        except (Timeout, ConnectionError) as e:
            log.warning("Network error (%s) url=%s attempt=%d/%d", type(e).__name__, url, attempt+1, max_retries)
            sleep_backoff(attempt)
            continue
        except RequestException as e:
            log.warning("RequestException url=%s attempt=%d/%d err=%s", url, attempt+1, max_retries, e)
            sleep_backoff(attempt)
            continue
        except ValueError as e:
            log.warning("Bad JSON url=%s err=%s", url, e)
            return None

    log.error("Giving up after %d retries url=%s", max_retries, url)
    return None

def list_allpages(api: str, limit: int = 100, namespace: int = 0):
    session = requests.Session()
    cont = None

    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "allpages",
            "apnamespace": str(namespace),
            "aplimit": str(limit),
        }
        if cont:
            params.update(cont)

        data = get_json_with_retry(session, api, params=params, timeout=60, max_retries=10)
        if not data:
            log.warning("allpages failed; sleeping and retrying")
            time.sleep(10)
            continue

        pages = data.get("query", {}).get("allpages", [])
        for p in pages:
            t = p.get("title")
            if t:
                yield t

        cont = data.get("continue")
        if not cont:
            break

def fandom_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.select("script, style, noscript, .reference, .mw-editsection"):
        tag.decompose()
    main = soup.select_one(".mw-parser-output") or soup
    return md(str(main))

def fetch_page_html(session: requests.Session, api: str, title: str) -> str | None:
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text",
        "disabletoc": "1",
        "disablelimitreport": "1",
        "redirects": "1",
    }
    data = get_json_with_retry(session, api, params=params, timeout=60, max_retries=10)
    if not data:
        return None

    html = data.get("parse", {}).get("text", {}).get("*")
    return html or None

def load_fandom_docs(source_cfg: dict, rate_limit_s: float = 1.0, max_pages: int | None = None):
    api = source_cfg["api"]
    ns = int(source_cfg.get("namespace", 0))
    session = requests.Session()

    count = 0
    for title in list_allpages(api, namespace=ns):
        html = fetch_page_html(session, api, title)
        if html:
            text = fandom_html_to_text(html)
            url = f"{api}?title={quote(title)}"
            yield url, title, text
        else:
            log.warning("Skipping page (fetch failed) title=%s", title)

        count += 1
        if max_pages is not None and count >= max_pages:
            break
        time.sleep(rate_limit_s)