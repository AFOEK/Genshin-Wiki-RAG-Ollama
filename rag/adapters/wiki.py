import requests
import time
import random
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import quote
from requests.exceptions import RequestException, Timeout, ConnectionError

log = logging.getLogger(__name__)
_RETRY_STATUSES = {429, 500, 502, 503, 504}

def sleep_backoff(attemp: int, base: float = 1.0, cap: float = 60.0) -> None:
    delay = min(cap, base * (2 ** attemp))
    delay *= (0.75 + random.random() * 0.6)
    time.sleep(delay)

def iter_recently_changed_titles(api: str, session: requests.Session, *, start_iso: str, namespace: int = 0, limit: int = 200):
    cont = None
    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "recentchanges",
            "rcnamespace": str(namespace),
            "rclimit": str(limit),
            "rcprop": "title|timestamp",
            "rcstart": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rcend": start_iso,
            "rcdir": "older",
        }
        if cont:
            params.update(cont)

        data = get_json_with_retry(session, api, params=params, timeout=60, max_retries=10)
        if not data:
            break

        rows = data.get("query", {}).get("recentchanges", [])
        for r in rows:
            title = r.get("title")
            ts = r.get("timestamp")
            if title and ts:
                yield title, ts

        cont = data.get("continue")
        if not cont:
            break

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
                log.warning("[WIKI] HTTP %s for %s params=%s", r.status_code, url, params)
                return None

            return r.json()

        except (Timeout, ConnectionError) as e:
            log.warning("[WIKI] Network error (%s) url=%s attempt=%d/%d", type(e).__name__, url, attempt+1, max_retries)
            sleep_backoff(attempt)
            continue
        except RequestException as e:
            log.warning("[WIKI] RequestException url=%s attempt=%d/%d err=%s", url, attempt+1, max_retries, e)
            sleep_backoff(attempt)
            continue
        except ValueError as e:
            log.warning("[WIKI] Bad JSON url=%s err=%s", url, e)
            return None

    log.error("[WIKI] Giving up after %d retries url=%s", max_retries, url)
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
            log.warning("[WIKI] allpages failed; sleeping and retrying")
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

    state_path = Path(source_cfg.get("state_file", "data/fandom_last_run.txt"))
    state_path.parent.mkdir(parents=True, exist_ok=True)

    last_run = None
    if state_path.exists():
        last_run = state_path.read_text(encoding="utf-8").strip() or None

    incremental = bool(last_run)

    if incremental:
        changes = list(iter_recently_changed_titles(api, session, start_iso=last_run, namespace=ns))
        log.info("[WIKI] incremental crawl since %s changed_titles=%d", last_run, len(changes))
    else:
        changes = [(title, None) for title in list_allpages(api, namespace=ns)]
        log.info("[WIKI] full crawl (no state_file)")

    count = 0
    failed = 0
    partial = False
    oldest_success_ts = None

    for title, change_ts in changes:
        html = fetch_page_html(session, api, title)
        if html:
            text = fandom_html_to_text(html) or ""
            url = f"{api}?title={quote(title)}"
            yield url, title, text, None, None

            if incremental and change_ts:
                if oldest_success_ts is None or change_ts < oldest_success_ts:
                    oldest_success_ts = change_ts
        else:
            failed += 1
            log.warning("[WIKI] Skipping page (fetch failed) title=%s", title)

        count += 1
        if max_pages is not None and count >= max_pages:
            partial = True
            log.warning("[WIKI] Stopped early due to max_pages=%d", max_pages)
            break
        time.sleep(rate_limit_s)

    if incremental:
        if not partial and failed == 0:
            new_state = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            state_path.write_text(new_state, encoding="utf-8")
            log.info("[WIKI] incremental state advanced to %s", new_state)
        else:
            log.warning(
                "[WIKI] incremental state NOT advanced (partial=%s failed=%d count=%d)",
                partial, failed, count
            )
    else:
        new_state = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        state_path.write_text(new_state, encoding="utf-8")
        log.info("[WIKI] full crawl state set to %s", new_state)
    
    log.info("[WIKI] done incremental=%s processed=%d failed=%d partial=%s", incremental, count, failed, partial)