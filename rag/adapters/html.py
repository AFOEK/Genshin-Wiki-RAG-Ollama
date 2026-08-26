import time
import requests
import logging
import random
import re
import threading

from urllib.parse import urlsplit, urlunsplit, parse_qs
from collections import deque
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from requests.exceptions import RequestException, Timeout, ConnectionError
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from utils.crawl import SharedRateLimiter, limited_get
from utils.browser_fallback import BrowserFallback

log = logging.getLogger(__name__)

_RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}

SKIP_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp",
            ".svg", ".pdf", ".zip", ".mp4", ".mp3",
            ".ico", ".css", ".js", ".woff", ".woff2",
            ".avi", ".mkv", ".webm")

_GAME8_BLOCK_MARKERS = ("access denied", "temporarily blocked", "too many requests", "verify you are human",
    "checking your browser",  "just a moment",  "cf-chl-",  "enable javascript and cookies")
GAME8_NOISE_SELECTORS=(
    ".p-membershipModal__backdrop",
    ".p-membershipModal__container",
    ".p-membershipModal__scrollContainer",
    ".p-membershipModal__content",
    ".c-commentItem__container--padding-sp",
    ".c-commentItem__header",
    ".c-commentItem__body",
    "a[href*='/comments']",
    "[data-track-mario-keyword*='comment']",
    "#comments",
    ".comments",
    ".comment-list",
    ".comment-thread",
    ".reply",
    ".replies",
    ".discussion",
    ".message-board",
    "img[src^='data:image']",
    "dialog",
    "aside",
)
GAME8_DISCOVERY_PATHS={
    "/games/Genshin-Impact",
    "/games/Genshin-Impact/archives",
}

class CloudflareChallengeError(RuntimeError):
    pass

def is_cloudflare_challenge(response) -> bool:
    cf_mitigated=(response.headers.get("cf-mitigated") or "").lower()

    if cf_mitigated=="challenge":
        return True

    content_type=(response.headers.get("content-type") or "").lower()
    server=(response.headers.get("server") or "").lower()

    if "text/html" not in content_type:
        return False

    body=(response.text or "")[:20000].lower()
    return ("cloudflare" in server and "just a moment" in body and "challenges.cloudflare.com" in body)

def is_game8_discovery_url(url:str) -> bool:
    path = urlparse(url).path.rstrip("/")
    return path in GAME8_DISCOVERY_PATHS

def is_game8_article(url:str) -> bool:
    path = urlparse(url).path.rstrip("/")
    return bool(re.fullmatch(r"/games/Genshin-Impact/archives/\d+", path))

def is_allowed_game8_url(url:str)->bool:
    return is_game8_discovery_url(url) or is_game8_article(url)

def allow_lang(url: str, allowed_lang: str = "EN") -> bool:
    qs = parse_qs(urlsplit(url).query)
    langs = qs.get("lang")
    if not langs:
        return True
    return langs[0].upper() == allowed_lang.upper()

def drop_game8_noise(root) -> None:
    for selector in GAME8_NOISE_SELECTORS:
        for node in list(root.select(selector)):
            node.decompose()

    membership_markers = ("what can you do as a free member", "create your free account today", "article watchlist", "game bookmarks", "cross-device sync", "premium articles", "site interface", "game tools")

    for node in list(root.find_all(["section", "aside", "dialog"])):
        node_text = node.get_text(" ", strip=True).casefold()
        if len(node_text) > 12_000:
            continue

        hits = sum(marker in node_text for marker in membership_markers)

        if hits >= 2:
            node.decompose()

def find_game8_article_root(soup:BeautifulSoup):
    canonical=soup.select_one(".archive-style-wrapper")
    if canonical is not None:
        text=canonical.get_text(" ",strip=True)
        if len(text) >= 800:
            return canonical

    selectors=(
        ".p-archiveContent__main",
        ".p-archiveContent__container",
        ".p-archiveBody__main",
    )
    reject_markers = ("membership", "modal", "login", "signup", "register", "premium", "comment")
    candidates=[]

    for selector in selectors:
        for candidate in soup.select(selector):
            classes=" ".join(candidate.get("class", [])).casefold()
            node_id=str(candidate.get("id") or "").casefold()
            identity=f"{classes} {node_id}"
            if any(marker in identity for marker in reject_markers):
                continue

            text=candidate.get_text(" ",strip=True)
            if len(text)<800:
                continue

            headings=candidate.find_all(["h1","h2","h3"])
            score=len(text)+(len(headings)*400)
            if candidate.find("h1"):
                score+=1200
            candidates.append((score,candidate))

    if not candidates:
        return None

    candidates.sort(key=lambda item:item[0],reverse=True)
    return candidates[0][1]

def is_low_value_game8_text(text: str) -> bool:
    normalized = " ".join((text or "").split())
    lowered = normalized.lower()
    if len(normalized) < 800:
        return True

    bad_phrases = (
        "what can you do as a free member",
        "create your free account today",
        "article watchlist",
        "game bookmarks",
        "cross-device sync",
        "comment rating",
        "premium articles",
        "site interface",
        "game tools",
        "interactive map pins",
        "build planner",
        "stat calculator",
        "diagnostic tool",
        "weapon/armor wishlist",
    )

    bad_hits = sum(phrase in lowered for phrase in bad_phrases)

    if bad_hits >= 3:
        return True

    if "what can you do as a free member" in lowered:
        return True

    if "create your free account today" in lowered and "article watchlist" in lowered:
        return True

    return False

def drop_honey_comment(soup: BeautifulSoup) -> None:
    selectors = [
        "#comment_page",
        "#comment_page_nav1",
        ".comments",
        ".commentlist",
        ".comment-respond",
        "li.comment",
        "ol.commentlist",
        "a[href*='#comment-']",
        "a[href*='replytocom=']",
    ]
    for sel in selectors:
        for node in soup.select(sel):
            node.decompose()

def normalize_url(u: str) -> str:
    parts = urlsplit(u)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

def sleep_backoff(attempt: int, base: float = 1.0, cap: float = 60.0) -> None:
    delay = min(cap, base * (2 ** attempt))
    delay *= (0.7 + random.random() * 0.6)
    time.sleep(delay)

def normalized_host(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host

def same_site(url: str, base_url: str) -> bool:
    return (normalized_host(url) == normalized_host(base_url))

def is_game8_url(url: str) -> bool:
    return (normalized_host(url) == "game8.co")

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

def html_to_text(html: str, url: str | None = None) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    host = (urlparse(url).netloc.lower() if url else "")

    if "game8.co" in host:
        main = find_game8_article_root(soup)

        if main is None:
            log.warning("[GAME8] article root not found url=%s", url)
            return ""
        before_noise = main.get_text(" ", strip=True)
        log.info("[GAME8] article root before cleanup url=%s chars=%d tag=%s class=%s", url, len(before_noise), main.name, main.get("class"))
        drop_game8_noise(main)
        main_preview = main.get_text(" ", strip=True)
        log.info("[GAME8] article root after cleanup url=%s chars=%d", url, len(main_preview))
        if is_low_value_game8_text(main_preview):
            log.warning("[GAME8] rejected low-value article root url=%s chars=%d", url, len(main_preview))
            return ""

    else:
        for tag in soup(["header", "footer", "nav", "aside"]):
            tag.decompose()

        if "honeyhunterworld.com" in host:
            drop_honey_comment(soup)

        main = (
            soup.select_one("main")
            or soup.select_one("article")
            or soup.select_one("#content")
            or soup.select_one(".mw-parser-output")
            or soup.body
            or soup
        )

    try:
        text = md(str(main))

    except RecursionError:
        log.warning("[HTML] markdownify recursion error; using text fallback")
        text = soup_text_fallback(main)

    except Exception as exc:
        log.warning("[HTML] markdownify failed type=%s; using text fallback", type(exc).__name__)
        text = soup_text_fallback(main)

    text = text.strip()

    if "game8.co" in host:
        if is_low_value_game8_text(text):
            log.warning("[GAME8] rejected extraction url=%s chars=%d", url, len(text))
            return ""

        log.info("[GAME8] extraction accepted url=%s chars=%d", url, len(text))

    return text

def game8_response_problem(response:requests.Response,html:str) -> str | None:
    if len(response.content)<1500:
        return f"undersized response body: {len(response.content)} bytes"
    lowered=html[:100_000].casefold()
    marker=next((marker for marker in _GAME8_BLOCK_MARKERS if marker in lowered),None)
    if marker:
        return f"blocking or challenge page: {marker!r}"
    return None

def crawl_site(base_url: str, seeds: list[str], deny_url, allow_url=None, rate_limit_s: float=1.0, max_pages: int | None=2000, allowed_langs: str="EN",workers: int=5, request_semaphore=None, browser_fallback: BrowserFallback | None=None):
    frontier=deque(normalize_url(url) for url in seeds)
    known:set[str]=set(frontier)
    seen:set[str]=set()
    workers=max(1, int(workers))
    game8_source=is_game8_url(base_url)
    request_delay_s=max(rate_limit_s, 1.5) if game8_source else max(0.0, rate_limit_s)
    max_retries=4 if game8_source else 10
    rate_limiter=SharedRateLimiter(request_delay_s)
    thread_local=threading.local()
    browser_mode=threading.Event()
    browser_switch_lock=threading.Lock()
    submitted=0

    def get_session() -> requests.Session:
        session=getattr(thread_local,"session",None)
        if session is None:
            session=requests.Session()
            session.headers.update({
                "User-Agent":"Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36",
                "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language":"en-US,en;q=0.9",
                "Cache-Control":"no-cache",
                "Pragma":"no-cache",
            })
            thread_local.session=session
        return session

    def fetch_one(requested_url: str):
        session=get_session()
        if browser_mode.is_set():
            if browser_fallback is None:
                return None, "browser mode enabled without BrowserFallback"

            try:
                response=browser_fallback.get_response(requested_url)
                return response,None
            except Exception as exc:
                return None, f"BrowserFallback: {type(exc).__name__}: {exc}"
            
        for attempt in range(max_retries+1):
            try:
                if game8_source:
                    time.sleep(random.uniform(0.05,0.25))
                response=limited_get(session,requested_url,timeout=(15,60),allow_redirects=True,semaphore=request_semaphore,rate_limiter=rate_limiter)
            except (Timeout, ConnectionError, RequestException) as exc:
                if attempt >= max_retries:
                    return None, f"{type(exc).__name__}: {exc}"
                log.warning("[HTML] Retrying url=%s attempt=%d/%d reason=%s",requested_url,attempt+1,max_retries,exc)
                sleep_backoff(attempt, base=2.0, cap=90.0)
                continue

            if is_cloudflare_challenge(response):
                log.warning("[HTML] Cloudflare challenge status=%d url=%s cf_ray=%s", response.status_code, requested_url, response.headers.get("cf-ray"))

                if browser_fallback is None:
                    return None, (f"Cloudflare challenge HTTP {response.status_code}; browser fallback unavailable")

                with browser_switch_lock:
                    if not browser_mode.is_set():
                        cleared=browser_fallback.wait_for_manual_clearance(requested_url)
                        if not cleared:
                            return None,"Cloudflare challenge was not cleared"

                        browser_mode.set()
                        log.warning("[HTML] source switched to browser mode base=%s", base_url,)

                try:
                    response=browser_fallback.get_response(requested_url)
                    return response,None
                except Exception as exc:
                    return None, (f"BrowserFallback: {type(exc).__name__}: {exc}")

            if game8_source and response.status_code==403:
                return response,"HTTP 403 blocked"

            if response.status_code in _RETRY_STATUSES:
                if attempt>=max_retries:
                    return None, (f"HTTP {response.status_code} after {max_retries} retries")

                retry_after=response.headers.get("Retry-After")
                log.warning("[HTML] Retrying url=%s attempt=%d/%d reason=HTTP %d", requested_url, attempt+1, max_retries, response.status_code)

                if retry_after and retry_after.isdigit():
                    time.sleep(min(int(retry_after),120))
                else:
                    sleep_backoff(attempt,base=2.0,cap=90.0)

                continue

            if game8_source and response.status_code<400:
                problem=game8_response_problem(response, response.text)
                if problem:
                    if attempt>=max_retries:
                        return None, (f"{problem} after {max_retries} retries")

                    log.warning("[HTML] Retrying url=%s attempt=%d/%d reason=%s", requested_url, attempt+1, max_retries, problem)
                    sleep_backoff(attempt,base=2.0,cap=90.0)
                    continue

            return response,None

        return None,"retry loop exhausted"

    def enqueue(link: str) -> bool:
        link=normalize_url(link)
        if link in known or link in seen:
            return False
        if not same_site(link,base_url):
            return False
        if deny_url and deny_url.search(link):
            return False
        if allow_url and not allow_url.search(link):
            if not (game8_source and is_game8_discovery_url(link)):
                return False
        if not allow_lang(link,allowed_langs):
            return False
        if urlsplit(link).path.lower().endswith(SKIP_EXT):
            return False
        known.add(link)
        frontier.append(link)
        return True

    def enqueue_game8_articles(html: str,page_url: str) -> int:
        added=0
        for link in extract_links(html,page_url):
            link=normalize_url(link)
            if is_game8_article(link) and enqueue(link):
                added+=1
        return added

    def submit_available(pool,pending: dict) -> None:
        nonlocal submitted
        while frontier and len(pending)<workers and (max_pages is None or submitted<max_pages):
            requested_url=normalize_url(frontier.popleft())
            if requested_url in seen:
                continue
            if not same_site(requested_url,base_url):
                seen.add(requested_url)
                continue
            if deny_url and deny_url.search(requested_url):
                log.warning("[HTML] Skip denied URL: %s",requested_url)
                seen.add(requested_url)
                continue
            if allow_url and not allow_url.search(requested_url):
                if not (game8_source and is_game8_discovery_url(requested_url)):
                    log.info("[HTML] Skip URL outside allow list: %s",requested_url)
                    seen.add(requested_url)
                    continue
            if not allow_lang(requested_url,allowed_langs) or urlsplit(requested_url).path.lower().endswith(SKIP_EXT):
                seen.add(requested_url)
                continue
            future=pool.submit(fetch_one,requested_url)
            pending[future]=requested_url
            submitted+=1

    log.info("[HTML] crawl start base=%s workers=%d rate_limit_s=%.2f max_pages=%s",base_url,workers,request_delay_s,max_pages)

    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix="crawl") as pool:
        pending={}
        submit_available(pool,pending)

        while pending:
            done, _=wait(tuple(pending), return_when=FIRST_COMPLETED)

            for future in done:
                requested_url=pending.pop(future)
                try:
                    response, fetch_error=future.result()
                except Exception as exc:
                    response, fetch_error=None,f"{type(exc).__name__}: {exc}"

                if response is None:
                    log.warning("[HTML] Giving up url=%s reason=%s", requested_url, fetch_error)
                    seen.add(requested_url)
                    continue

                final_url=normalize_url(response.url)

                if game8_source and response.status_code==403:
                    log.warning("[GAME8] HTTP 403 blocked url=%s", requested_url)
                    seen.add(requested_url)
                    continue

                if response.status_code>=400:
                    log.warning("[HTML] HTTP %d requested=%s final=%s", response.status_code, requested_url,final_url)
                    seen.add(requested_url)
                    continue

                if not same_site(final_url,base_url):
                    log.warning("[HTML] Redirect outside approved site requested=%s final=%s", requested_url, final_url)
                    seen.add(requested_url)
                    continue

                if allow_url and not allow_url.search(final_url):
                    if not (game8_source and is_game8_discovery_url(final_url)):
                        log.warning("[HTML] Redirect outside allow list requested=%s final=%s", requested_url, final_url)
                        seen.add(requested_url)
                        continue

                content_type=(response.headers.get("Content-Type") or "").lower()
                if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                    log.info("[HTML] Skip non-HTML content-type=%s url=%s", content_type, final_url)
                    seen.add(requested_url)
                    continue

                html=response.text

                if game8_source and is_game8_discovery_url(final_url):
                    added=enqueue_game8_articles(html, final_url)
                    seen.add(requested_url)
                    seen.add(final_url)
                    known.add(final_url)
                    log.info("[GAME8] discovery processed url=%s article_links_added=%d", final_url, added)
                    continue

                if game8_source and not is_game8_article(final_url):
                    log.info("[GAME8] skipping non-article url=%s", final_url)
                    seen.add(requested_url)
                    seen.add(final_url)
                    known.add(final_url)
                    continue

                text=html_to_text(html,final_url)
                if not text.strip():
                    log.warning("[HTML] Skipping empty extraction url=%s", final_url)
                    seen.add(requested_url)
                    seen.add(final_url)
                    known.add(final_url)
                    continue

                title=final_url
                try:
                    soup=BeautifulSoup(html,"lxml")
                    if soup.title and soup.title.string:
                        title=soup.title.string.strip()
                except Exception:
                    log.exception("[HTML] Failed to extract title url=%s", final_url)

                seen.add(requested_url)
                seen.add(final_url)
                known.add(final_url)

                if game8_source:
                    added=enqueue_game8_articles(html,final_url)
                    log.debug("[GAME8] article=%s discovered_articles=%d",final_url,added)
                else:
                    for link in extract_links(html,final_url):
                        enqueue(link)

                last_modified=response.headers.get("Last-Modified")
                etag=response.headers.get("ETag")
                log.info("[HTML] accepted requested=%s final=%s chars=%d",requested_url,final_url,len(text))
                yield final_url,title,text,last_modified,etag

            submit_available(pool,pending)

    log.info("[HTML] crawl done base=%s submitted=%d seen=%d remaining=%d workers=%d",base_url,submitted,len(seen),len(frontier),workers)