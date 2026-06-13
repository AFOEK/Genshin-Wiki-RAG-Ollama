import time
import requests
import logging
import random
from urllib.parse import urlsplit, urlunsplit, parse_qs
from collections import deque
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from requests.exceptions import RequestException, Timeout, ConnectionError

log = logging.getLogger(__name__)

_RETRY_STATUSES = {429, 500, 502, 503, 504}

GAME8_NOISE_SELECTORS = (
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
)

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

    membership_heading = ("what can you do as a free member")

    for heading in list(root.find_all(["h1", "h2", "h3", "h4"])):
        heading_text = heading.get_text(" ", strip=True).lower()

        if membership_heading not in heading_text:
            continue

        container = None
        for parent in heading.parents:
            if parent is root:
                break

            if parent.name not in {"dialog", "section", "aside", "div"}:
                continue

            parent_text = parent.get_text(
                " ",
                strip=True,
            ).lower()

            if len(parent_text) > 12_000:
                break

            membership_hits = sum(
                phrase in parent_text
                for phrase in (
                    "create your free account today",
                    "article watchlist",
                    "game bookmarks",
                    "cross-device sync",
                    "comment rating",
                    "premium articles",
                    "continue as a guest",
                )
            )

            if membership_hits >= 3:
                container = parent
                break

        if container is not None:
            container.decompose()
        else:
            heading.decompose()

def find_game8_article_root(soup: BeautifulSoup):
    selectors = (
        "main article",
        "article",
        "[role='main']",
        "main",
        "#content",
    )

    for selector in selectors:
        candidates = soup.select(selector)
        for candidate in candidates:
            text = candidate.get_text(
                " ",
                strip=True,
            )

            headings = candidate.find_all(
                ["h1", "h2", "h3"]
            )

            if (
                len(text) >= 1_000
                and candidate.find("h1") is not None
                and len(headings) >= 3
            ):
                return candidate

    for heading in soup.find_all("h1"):
        heading_text = heading.get_text(
            " ",
            strip=True,
        )

        if not heading_text:
            continue

        for parent in heading.parents:
            if parent.name not in {"article", "main", "section", "div"}:
                continue
            text = parent.get_text(" ", strip=True)

            subheadings = parent.find_all(["h2", "h3"])

            if (
                len(text) >= 1_000
                and len(subheadings) >= 2
            ):
                return parent

    return None

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
    )

    bad_hits = sum(phrase in lowered for phrase in bad_phrases)

    article_signals = (
        "last updated on",
        "list of contents",
        "genshin impact",
    )

    has_article_signal = any(signal in lowered for signal in article_signals)

    if (bad_hits >= 3 and len(normalized) < 6_000 and not has_article_signal):
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

def html_to_text(html: str, url: str | None = None) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    host = (
        urlparse(url).netloc.lower()
        if url
        else ""
    )

    if "game8.co" in host:
        main = find_game8_article_root(soup)

        if main is None:
            log.warning(
                "[GAME8] article root not found url=%s",
                url,
            )
            return ""

        drop_game8_noise(main)

    else:
        for tag in soup(
            ["header", "footer", "nav", "aside"]):
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
        log.warning(
            "[HTML] markdownify recursion error; "
            "using text fallback"
        )
        text = soup_text_fallback(main)

    except Exception as exc:
        log.warning(
            "[HTML] markdownify failed type=%s; "
            "using text fallback",
            type(exc).__name__,
        )
        text = soup_text_fallback(main)

    text = text.strip()

    if "game8.co" in host:
        if is_low_value_game8_text(text):
            log.warning("[GAME8] rejected extraction url=%s chars=%d", url, len(text))
            return ""

        log.info("[GAME8] extraction accepted url=%s chars=%d", url, len(text))

    return text

def crawl_site(base_url: str, seeds: list[str], deny_url, allow_url = None, rate_limit_s: float = 1.0, max_pages: int | None = 2000, allowed_langs: str = "EN"):
    q = deque(seeds)
    seen: set[str] = set()
    retries: dict[str, int] = {}
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "(X11; Linux aarch64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/136.0 Safari/537.36 "
                "GenshinRAG/1.0"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,*/*;q=0.8"
            ),
            "Accept-Language": (
                "en-US,en;q=0.9"
            ),
            "Cache-Control": "no-cache",
        }
    )

    while q and (max_pages is None or len(seen) < max_pages):
        url = q.popleft()
        if url in seen:
            continue

        if not same_site(url, base_url):
            continue
        if deny_url and deny_url.search(url):
            log.warning("[WARN] SKIP url:%s, illegal content detected", url)
            seen.add(url)
            continue
        if allow_url and not allow_url.search(url):
            log.info("[WARN] SKIP not in allow list: %s", url)
            seen.add(url)
            continue
        if not allow_lang(url, allowed_langs):
            seen.add(url)
            continue

        SKIP_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp",
                    ".svg", ".pdf", ".zip", ".mp4", ".mp3",
                    ".ico", ".css", ".js", ".woff", ".woff2",
                    ".avi", ".mkv", ".webm")
        
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
            )
            last_modified = r.headers.get("Last-Modified")
            etag = r.headers.get("ETag")

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
            if not allow_lang(link, allowed_langs):
                continue
            if (link not in seen and same_site(link, base_url) and not (deny_url and deny_url.search(link)) and not (allow_url and not allow_url.search(link))):
                q.append(link)

        text = html_to_text(html, url)
        if not text or not text.strip():
            log.warning("[CRAWL] Skipping empty extraction url=%s", url)
            time.sleep(rate_limit_s)
            continue

        title = url
        try:
            soup = BeautifulSoup(html, "lxml")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
        except Exception:
            log.exception("Failed to extract title url=%s", url)

        yield url, title, text, last_modified, etag
        time.sleep(rate_limit_s)