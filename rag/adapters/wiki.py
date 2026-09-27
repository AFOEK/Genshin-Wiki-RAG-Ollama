import requests
import time
import random
import logging
import threading
import json
import subprocess

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import quote, urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import RequestException, Timeout, ConnectionError

from utils.crawl import SharedRateLimiter, limited_get

log = logging.getLogger(__name__)
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_FANDOM_USER_AGENT = "GenshinWikiRAG/1.0 (academic research)"
_CURL_FALLBACK_LOCK = threading.Lock()

def make_fandom_session() -> requests.Session:
    session=requests.Session()
    session.headers.update({
        "User-Agent":_FANDOM_USER_AGENT,
        "Accept":"application/json",
        "Accept-Language":"en-US,en;q=0.9",
    })
    return session

def sleep_backoff(attempt: int, base: float = 1.0, cap: float = 60.0) -> None:
    delay = min(cap, base * (2 ** attempt))
    delay *= (0.75 + random.random() * 0.6)
    time.sleep(delay)

def iter_recently_changed_titles(api: str, session: requests.Session, *, start_iso: str, namespace: int=0, limit: int=200, rate_limiter: SharedRateLimiter | None=None, request_semaphore=None):
    cont=None
    rcstart=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    while True:
        params={
            "action":"query",
            "format":"json",
            "list":"recentchanges",
            "rcnamespace":str(namespace),
            "rclimit":str(limit),
            "rcprop":"title|timestamp",
            "rcstart":rcstart,
            "rcend":start_iso,
            "rcdir":"older",
        }
        if cont:
            params.update(cont)

        data=get_json_with_retry(session, api, params=params, timeout=60, max_retries=3, rate_limiter=rate_limiter, request_semaphore=request_semaphore)
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

def get_json_with_retry(session: requests.Session, url: str, *, params: dict[str,Any], timeout: float=60.0, max_retries: int=3, rate_limiter: SharedRateLimiter | None=None, request_semaphore=None) -> Optional[dict[str,Any]]:
    last_reason="unknown"
    for attempt in range(max_retries):
        try:
            r=limited_get(session, url, params=params, timeout=timeout, semaphore=request_semaphore, rate_limiter=rate_limiter,)
            if r.status_code<400:
                try:
                    data=r.json()
                except ValueError as exc:
                    last_reason=f"bad JSON: {exc}"
                    log.warning("[WIKI] Bad JSON url=%s attempt=%d/%d", url,attempt+1,max_retries,)
                else:
                    if isinstance(data,dict):
                        return data

                    last_reason="non-object JSON"
                    log.warning("[WIKI] Non-object JSON url=%s attempt=%d/%d", url,attempt+1,max_retries,)
            elif r.status_code==403:
                last_reason="HTTP 403"
                log.warning("[WIKI] HTTP 403 url=%s attempt=%d/%d", url,attempt+1,max_retries,)
            elif r.status_code in _RETRY_STATUSES:
                last_reason=f"HTTP {r.status_code}"
                log.warning("[WIKI] HTTP %d url=%s attempt=%d/%d", r.status_code,url,attempt+1,max_retries,)
                retry_after=r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(int(retry_after),120))
                    continue
            else:
                log.warning("[WIKI] HTTP %d url=%s params=%s", r.status_code,url,params,)
                return None

        except (Timeout,ConnectionError) as exc:
            last_reason=f"{type(exc).__name__}: {exc}"

            log.warning("[WIKI] Network error %s url=%s attempt=%d/%d", type(exc).__name__,url,attempt+1,max_retries,)

        except RequestException as exc:
            last_reason=f"RequestException: {exc}"
            log.warning("[WIKI] RequestException url=%s attempt=%d/%d err=%s", url,attempt+1,max_retries,exc,)

        if attempt+1<max_retries:
            sleep_backoff(attempt)

    log.warning("[WIKI] requests failed %d times url=%s reason=%s; falling back to curl", max_retries,url,last_reason,)
    data=curl_get_json(url, params, timeout=timeout, request_semaphore=request_semaphore,)
    if data is not None:
        log.info("[WIKI] curl fallback succeeded url=%s",url)
        return data

    log.error("[WIKI] requests + curl fallback failed url=%s", url,)
    return None

def curl_get_json(url: str, params: dict[str,Any], *, timeout: float=60.0, request_semaphore=None) -> Optional[dict[str,Any]]:
    full_url=f"{url}?{urlencode(params,doseq=True)}"
    marker="__HTTP_STATUS__:"

    cmd=[
        "curl",
        "-sS",
        "--compressed",
        "--location",
        "--connect-timeout","15",
        "--max-time",str(timeout),
        "-A",_FANDOM_USER_AGENT,
        "-H","Accept: application/json",
        "-w",f"\n{marker}%{{http_code}}",
        full_url,
    ]

    if request_semaphore is not None:
        request_semaphore.acquire()

    try:
        with _CURL_FALLBACK_LOCK:
            if request_semaphore is not None:
                request_semaphore.acquire()
            try:
                result=subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+10,)
            finally:
                if request_semaphore is not None:
                    request_semaphore.release()
    finally:
        if request_semaphore is not None:
            request_semaphore.release()

    if result.returncode!=0:
        log.warning("[WIKI] curl fallback failed exit=%d stderr=%r", result.returncode, result.stderr[:500],)
        return None

    split_marker=f"\n{marker}"
    pos=result.stdout.rfind(split_marker)

    if pos<0:
        log.warning("[WIKI] curl fallback missing HTTP status")
        return None

    body=result.stdout[:pos]

    try:
        status=int(result.stdout[pos+len(split_marker):].strip())
    except ValueError:
        log.warning("[WIKI] curl fallback invalid HTTP status")
        return None

    if status>=400:
        log.warning("[WIKI] curl fallback HTTP %d url=%s body=%r", status,url,body[:500],)
        return None

    try:
        data=json.loads(body)
    except json.JSONDecodeError as exc:
        log.warning("[WIKI] curl fallback bad JSON url=%s err=%s", url,exc,)
        return None

    return data if isinstance(data, dict) else None

def list_allpages(api: str, limit: int = 100, namespace: int = 0, request_semaphore = None, rate_limiter: SharedRateLimiter | None = None, max_outer_failures: int = 3):
    session=make_fandom_session()
    cont = None
    failures = 0

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

        data=get_json_with_retry(session, api, params=params, timeout=60, max_retries=3, rate_limiter=rate_limiter, request_semaphore=request_semaphore)
        if not data:
            failures += 1
            if failures >= max_outer_failures:
                raise RuntimeError(f"[WIKI] Allpages discovery failed {failures} consecutive times")

            delay = min(15 * failures, 60)
            log.warning("[WIKI] allpages failed outer=%d/%d; sleeping %ds and retrying !", failures, max_outer_failures, delay)
            time.sleep(delay)
            continue

        failures = 0
        
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

    data=get_json_with_retry(session, api, params=params, timeout=60, max_retries=3, rate_limiter=rate_limiter, request_semaphore=request_semaphore,)
    if not data:
        return None

    return data.get("parse",{}).get("text",{}).get("*") or None

def load_fandom_docs(source_cfg: dict, rate_limit_s: float=1.0, max_pages: int | None=None, workers: int=4, request_semaphore=None):
    api=source_cfg["api"]
    ns=int(source_cfg.get("namespace",0))
    discovery_session=make_fandom_session()
    state_path=Path(source_cfg.get("state_file", "data/fandom_last_run.txt"))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    crawl_started_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    last_run=state_path.read_text(encoding="utf-8").strip() or None if state_path.exists() else None
    incremental=bool(last_run)
    workers=max(1,int(workers))
    rate_limiter=SharedRateLimiter(rate_limit_s)

    if incremental:
        raw_changes=iter_recently_changed_titles(api, discovery_session, start_iso=last_run, namespace=ns, rate_limiter=rate_limiter, request_semaphore=request_semaphore,)
        changed_by_title={}
        for title,ts in raw_changes:
            changed_by_title.setdefault(title,ts)

        changes=list(changed_by_title.items())
        log.info("[WIKI] incremental crawl since %s changed_titles=%d", last_run, len(changes),)
    else:
        changes=[(title,None) for title in list_allpages(api, namespace=ns, request_semaphore=request_semaphore, rate_limiter=rate_limiter)]
        log.info("[WIKI] full crawl (no state_file) titles=%d", len(changes))

    partial=max_pages is not None and len(changes)>max_pages
    targets=changes[:max_pages] if max_pages is not None else changes
    thread_local=threading.local()
    failed=0

    def get_session() -> requests.Session:
        session=getattr(thread_local, "session", None)
        if session is None:
            session=make_fandom_session()
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