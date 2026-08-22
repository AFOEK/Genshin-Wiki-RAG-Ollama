import requests
import time
import random
import logging
import threading

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import RequestException, Timeout, ConnectionError

from utils.crawl import SharedRateLimiter, limited_get

log = logging.getLogger(__name__)
_RETRY_STATUSES = {429, 500, 502, 503, 504}

def sleep_backoff(attempt: int, base: float = 1.0, cap: float = 60.0) -> None:
    delay = min(cap, base * (2 ** attempt))
    delay *= (0.75 + random.random() * 0.6)
    time.sleep(delay)

def iter_recently_changed_titles(api: str, session: requests.Session, *, start_iso: str, namespace: int = 0, limit: int = 200, request_semaphore=None):
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

        data = get_json_with_retry(session, api, params=params, timeout=60, max_retries=10, request_semaphore=None)
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

def get_json_with_retry(session: requests.Session, url: str, *, params: dict[str, Any], timeout: float = 60.0, max_retries: int = 10, rate_limiter: SharedRateLimiter | None = None, request_semaphore=None) -> Optional[dict[str, Any]]:
    for attempt in range(max_retries):
        try:
            r=limited_get(session, url, params=params, timeout=timeout, semaphore=request_semaphore, rate_limiter=rate_limiter)
            if r.status_code in _RETRY_STATUSES:
                ra=r.headers.get("Retry-After")
                if ra and ra.isdigit():
                    time.sleep(min(int(ra),120))
                else:
                    sleep_backoff(attempt)
                continue
            if r.status_code>=400:
                log.warning("[WIKI] HTTP %s for %s params=%s", r.status_code, url, params)
                return None
            return r.json()
        except (Timeout,ConnectionError) as e:
            log.warning("[WIKI] Network error (%s) url=%s attempt=%d/%d", type(e).__name__, url, attempt+1, max_retries)
            sleep_backoff(attempt)
        except RequestException as e:
            log.warning("[WIKI] RequestException url=%s attempt=%d/%d err=%s", url, attempt+1, max_retries, e)
            sleep_backoff(attempt)
        except ValueError as e:
            log.warning("[WIKI] Bad JSON url=%s err=%s", url, e)
            return None
    log.error("[WIKI] Giving up after %d retries url=%s", max_retries, url)
    return None


def list_allpages(api: str, limit: int = 100, namespace: int = 0, request_semaphore = None):
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

        data = get_json_with_retry(session, api, params=params, timeout=60, max_retries=10, request_semaphore=request_semaphore)
        if not data:
            log.warning("[WIKI] allpages failed; sleeping and retrying !")
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

def fetch_page_html(session: requests.Session, api: str, title: str, *, rate_limiter: SharedRateLimiter | None=None, request_semaphore=None) -> str | None:
    params={
        "action":"parse",
        "format":"json",
        "page":title,
        "prop":"text",
        "disabletoc":"1",
        "disablelimitreport":"1",
        "redirects":"1",
    }
    data = get_json_with_retry(session, api, params=params, timeout=60, max_retries=10, rate_limiter=rate_limiter, request_semaphore=request_semaphore)
    if not data:
        return None
    html = data.get("parse", {}).get("text", {}).get("*")
    return html or None

def load_fandom_docs(source_cfg: dict, rate_limit_s: float=1.0, max_pages: int | None=None, workers: int=4, request_semaphore=None):
    api=source_cfg["api"]
    ns=int(source_cfg.get("namespace",0))
    discovery_session=requests.Session()
    state_path=Path(source_cfg.get("state_file", "data/fandom_last_run.txt"))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    crawl_started_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    last_run=state_path.read_text(encoding="utf-8").strip() or None if state_path.exists() else None
    incremental=bool(last_run)

    if incremental:
        changes=list(iter_recently_changed_titles(api, discovery_session, start_iso=last_run, namespace=ns, request_semaphore=request_semaphore))
        log.info("[WIKI] incremental crawl since %s changed_titles=%d", last_run, len(changes))
    else:
        changes=[(title,None) for title in list_allpages(api, namespace=ns, request_semaphore=request_semaphore)]
        log.info("[WIKI] full crawl (no state_file) titles=%d", len(changes))

    partial=max_pages is not None and len(changes)>max_pages
    targets=changes[:max_pages] if max_pages is not None else changes
    workers=max(1,int(workers))
    rate_limiter=SharedRateLimiter(rate_limit_s)
    thread_local=threading.local()
    failed=0

    def get_session() -> requests.Session:
        session=getattr(thread_local, "session", None)
        if session is None:
            session=requests.Session()
            thread_local.session=session
        return session

    def fetch_one(title: str, change_ts: str | None):
        html=fetch_page_html(get_session(), api, title, rate_limiter=rate_limiter, request_semaphore=request_semaphore)
        if not html:
            return None
        text=fandom_html_to_text(html) or ""
        url=f"{api}?title={quote(title)}"
        return url,title,text,None,None

    log.info("[WIKI] fetch workers=%d rate_limit_s=%.2f pages=%d", workers, rate_limit_s, len(targets))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fandom") as pool:
        futures={pool.submit(fetch_one,title,change_ts):title for title, change_ts in targets}
        for future in as_completed(futures):
            title=futures[future]
            try:
                item=future.result()
            except Exception:
                failed+=1
                log.exception("[WIKI] worker failed title=%s", title)
                continue
            if item is None:
                failed+=1
                log.warning("[WIKI] Skipping page (fetch failed) title=%s", title)
                continue
            yield item

    count=len(targets)
    if not partial and failed==0:
        state_path.write_text(crawl_started_at, encoding="utf-8")
        log.info("[WIKI] crawl state advanced to %s", crawl_started_at)
    else:
        log.warning("[WIKI] state NOT advanced partial=%s failed=%d count=%d", partial, failed,count)
    log.info("[WIKI] done incremental=%s processed=%d failed=%d partial=%s workers=%d", incremental, count, failed, partial, workers)