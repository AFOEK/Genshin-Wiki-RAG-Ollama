import requests
import time
import random
import logging
import threading
import pycurl
import io

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import RequestException, Timeout, ConnectionError
from requests.structures import CaseInsensitiveDict

from utils.crawl import SharedRateLimiter, limited_get

log = logging.getLogger(__name__)
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_FANDOM_USER_AGENT = "GenshinWikiRAG/1.0 (academic research)"
_TRANSPORT_LOCK = threading.Lock()
_CURL_LOCAL = threading.local()
_REQUEST_FAILURES = 0
_USE_PYCURL = False

def curl_mode_enabled() -> bool:
    with _TRANSPORT_LOCK:
        return _USE_PYCURL

def record_requests_failure(reason: str) -> None:
    global _REQUEST_FAILURES, _USE_PYCURL

    with _TRANSPORT_LOCK:
        if _USE_PYCURL:
            return

        _REQUEST_FAILURES += 1
        log.warning("[WIKI] Global requests failures=%d/3 reason=%s", _REQUEST_FAILURES, reason)
        if _REQUEST_FAILURES >= 3:
            _USE_PYCURL = True
            log.warning("[WIKI] Switching globally to PycURL")


class PycurlSession:
    def __init__(self):
        self.curl = pycurl.Curl()

    def get(self, url: str, params=None, timeout=60, **kwargs):
        full_url = requests.Request("GET", url, params=params).prepare().url
        body = io.BytesIO()
        headers = CaseInsensitiveDict()
        def read_header(chunk: bytes) -> int:
            line = chunk.decode("iso-8859-1").strip()
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip()] = value.strip()
            return len(chunk)

        c = self.curl
        c.reset()
        c.setopt(pycurl.URL, full_url)
        c.setopt(pycurl.FOLLOWLOCATION, 1)
        c.setopt(pycurl.MAXREDIRS, 5)
        c.setopt(pycurl.CONNECTTIMEOUT, 15)
        c.setopt(pycurl.TIMEOUT_MS, int(timeout * 1000))
        c.setopt(pycurl.USERAGENT, _FANDOM_USER_AGENT)
        c.setopt(pycurl.HTTPHEADER, ["Accept: application/json"])
        c.setopt(pycurl.ACCEPT_ENCODING, "")
        c.setopt(pycurl.WRITEDATA, body)
        c.setopt(pycurl.HEADERFUNCTION, read_header)
        c.perform()

        response = requests.Response()
        response.status_code = c.getinfo(pycurl.RESPONSE_CODE)
        response.url = c.getinfo(pycurl.EFFECTIVE_URL)
        response.headers = headers
        response._content = body.getvalue()
        response.encoding = "utf-8"
        return response

def get_pycurl_session() -> PycurlSession:
    session = getattr(_CURL_LOCAL, "session", None)
    if session is None:
        session = PycurlSession()
        _CURL_LOCAL.session = session
    return session

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

        data = get_json_with_retry(session, api, params=params, timeout=60, max_retries=3, rate_limiter=rate_limiter, request_semaphore=request_semaphore)
        if not data:
            raise RuntimeError("[WIKI] Recentchanges discovery failed; state not advanced")

        rows = data.get("query", {}).get("recentchanges")
        if not isinstance(rows, list):
            raise RuntimeError(f"[WIKI] Invalid recentchanges response: {data.get('error', data)}")

        rows = data.get("query", {}).get("recentchanges", [])
        for r in rows:
            title = r.get("title")
            ts = r.get("timestamp")
            if title and ts:
                yield title, ts

        cont = data.get("continue")
        if not cont:
            break

def get_json_with_retry(session: requests.Session, url: str, *, params: dict[str, Any], timeout: float = 60.0, max_retries: int = 3, rate_limiter: SharedRateLimiter | None = None, request_semaphore=None) -> Optional[dict[str, Any]]:

    def fetch_curl() -> Optional[dict[str, Any]]:
        for attempt in range(3):
            try:
                r = limited_get(
                    get_pycurl_session(),
                    url,
                    params=params,
                    timeout=timeout,
                    semaphore=request_semaphore,
                    rate_limiter=rate_limiter,
                )

                if r.status_code in _RETRY_STATUSES:
                    log.warning("[WIKI] PycURL HTTP %d attempt=%d/3", r.status_code, attempt+1)
                    if attempt < 2:
                        ra = r.headers.get("Retry-After")
                        if ra and ra.isdigit():
                            time.sleep(min(int(ra), 120))
                        else:
                            sleep_backoff(attempt)
                    continue

                if r.status_code >= 400:
                    log.warning("[WIKI] PycURL HTTP %d url=%s", r.status_code, url)
                    return None

                data = r.json()
                return data if isinstance(data, dict) else None

            except (pycurl.error, ValueError) as exc:
                log.warning("[WIKI] PycURL error attempt=%d/3 err=%s", attempt+1, exc)
                if attempt < 2:
                    sleep_backoff(attempt)

        return None

    for attempt in range(max_retries):
        if curl_mode_enabled():
            return fetch_curl()

        try:
            r = limited_get(session, url, params=params, timeout=timeout, semaphore=request_semaphore, rate_limiter=rate_limiter,)
            if r.status_code == 403:
                record_requests_failure("HTTP 403")
            elif r.status_code in _RETRY_STATUSES:
                log.warning("[WIKI] HTTP %d attempt=%d/%d url=%s", r.status_code, attempt + 1, max_retries, url,)
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(int(retry_after), 120))
                    continue
            elif r.status_code >= 400:
                log.warning("[WIKI] HTTP %d url=%s", r.status_code, url)
                return None
            else:
                try:
                    data = r.json()
                except ValueError:
                    record_requests_failure("Invalid JSON")
                else:
                    if isinstance(data, dict):
                        return data
                    record_requests_failure("Non-object JSON")

        except RequestException as exc:
            record_requests_failure(type(exc).__name__)
            log.warning("[WIKI] requests error url=%s err=%s", url, exc)

        if curl_mode_enabled():
            return fetch_curl()

        if attempt + 1 < max_retries:
            sleep_backoff(attempt)

    return None

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
        
        pages = data.get("query", {}).get("allpages")
        if not isinstance(pages, list):
            raise RuntimeError(f"[WIKI] Invalid allpages response: {data.get('error', data)}")
        
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