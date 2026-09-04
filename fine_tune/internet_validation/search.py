from __future__ import annotations

import logging
import time
import requests
import threading

from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from bs4 import BeautifulSoup

from policy_loader import SourcePolicy
from url_validation import is_allowed_url

log = logging.getLogger(__name__)

_search_lock = threading.Lock()
_last_search = 0.0

class SearchUnavailableError(RuntimeError):
    pass

def extract_page_text(html: str, *, max_chars: int) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript", "svg", "nav", "footer", "form"]):
        element.decompose()

    text = " ".join(soup.get_text(separator=" ", strip=True).split())
    return text[:max_chars]

def fetch_fandom_api(*, session: requests.Session, url: str, timeout_s: float, max_chars: int) -> tuple[str, str]:
    parsed = urlparse(url)
    if not parsed.path.startswith("/wiki/"):
        raise ValueError(f"Unsupported Fandom URL: {url}")

    page_title = unquote(parsed.path.removeprefix("/wiki/"))
    api_url = f"{parsed.scheme}://{parsed.netloc}/api.php"
    response = session.get(
        api_url,
        params={
            "action": "parse",
            "page": page_title,
            "prop": "text",
            "format": "json",
            "formatversion": 2,
        },
        timeout=timeout_s,
        headers={"User-Agent": "GenshinDatasetValidator/1.0"},
    )
    response.raise_for_status()
    payload = response.json()
    html = str((payload.get("parse", {}) or {}).get("text", ""))

    if not html:
        raise RuntimeError(f"Fandom API returned no page text for {page_title!r}")

    return url, extract_page_text(html, max_chars=max_chars)

def fetch_result_page(*, session: requests.Session, url: str, policy: SourcePolicy, timeout_s: float, max_chars: int) -> tuple[str, str]:
    response = session.get(url, timeout=timeout_s, allow_redirects=True, headers={"User-Agent": "GenshinDatasetValidator/1.0"})
    if response.status_code == 403:
        parsed = urlparse(url)
        if (parsed.hostname or "").lower().endswith("fandom.com"):
            return fetch_fandom_api(session=session, url=url, timeout_s=timeout_s, max_chars=max_chars)

    response.raise_for_status()
    final_url = response.url
    if not is_allowed_url(final_url, policy):
        raise ValueError(f"Result redirected outside approved source: {final_url}")

    content_type = response.headers.get("content-type", "").lower()
    text = (extract_page_text(response.text, max_chars=max_chars) if "text/html" in content_type else response.text[:max_chars])
    return final_url, text

def wait_for_search_slot(min_interval_s: float) -> None:
    global _last_search
    with _search_lock:
        now = time.monotonic()
        delay = min_interval_s - (now - _last_search)

        if delay > 0:
            time.sleep(delay)

        _last_search = time.monotonic()

def match_policy(url: str, policies: list[SourcePolicy]) -> SourcePolicy | None:
    for policy in policies:
        if is_allowed_url(url, policy):
            return policy
    return None

def search_one_source(*, question: str, policies: list[SourcePolicy], searxng_url: str, results_per_source: int, fetched_pages_per_source: int, search_timeout_s: float, fetch_timeout_s: float, max_chars_per_page: int, search_interval_s: float) -> list[dict[str, Any]]:
    session = requests.Session()
    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    rejected_urls = 0

    query = f"Genshin Impact {question}"
    wait_for_search_slot(search_interval_s)
    response = session.get(
        f"{searxng_url.rstrip('/')}/search",
        params={
            "q": query,
            "format": "json",
            "categories": "general",
            "language": "en",
            "safesearch": 1,
        },
        timeout=search_timeout_s,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "json" not in content_type:
        raise RuntimeError(f"[SearXNG] did not return JSON: status={response.status_code}, content_type={content_type!r}, body={response.text[:300]!r}")
    payload = response.json()
    raw_results = payload.get("results", [])
    unresponsive = payload.get("unresponsive_engines", [])
    log.info("[SEARCH_DEBUG] raw=%d unresponsive=%s", len(raw_results), unresponsive)
    if not raw_results and unresponsive:
        raise SearchUnavailableError(f"SearXNG returned no results and search engines are unavailable: {unresponsive}")
    
    for search_rank, result in enumerate(raw_results, start=1):
        url = str(result.get("url", "")).strip()

        if not url or url in seen_urls:
            continue

        matched_policy = match_policy(url, policies)

        if matched_policy is None:
            rejected_urls += 1
            continue

        seen_urls.add(url)

        candidates.append({
            "source": matched_policy.name,
            "tier": matched_policy.tier,
            "source_weight": matched_policy.weight,
            "title": str(result.get("title", "")),
            "url": url,
            "search_rank": search_rank,
            "snippet": str(result.get("content", "")),
            "_policy": matched_policy,
        })

        if len(candidates) >= results_per_source:
            break

    log.info("[SEARCH_DEBUG] candidates=%d rejected=%d", len(candidates), rejected_urls)
    evidence: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates[:fetched_pages_per_source]):
        candidate_policy = candidate["_policy"]
        if index:
            time.sleep(candidate_policy.rate_limit_s)

        try:
            final_url, page_text = fetch_result_page(
                session=session,
                url=candidate["url"],
                policy=candidate_policy,
                timeout_s=fetch_timeout_s,
                max_chars=max_chars_per_page,
            )
            candidate["url"] = final_url
            candidate["text"] = page_text
            candidate["fetch_ok"] = True
        except Exception as exc:
            candidate["text"] = candidate["snippet"]
            candidate["fetch_ok"] = False
            candidate["fetch_error"] = f"{type(exc).__name__}: {exc}"

        candidate["evidence_id"] = f"{candidate_policy.name}:{index + 1}"
        candidate.pop("_policy", None)
        evidence.append(candidate)

    return evidence

def deduplicate_and_trim_evidence(evidence: list[dict], *, max_total_chars: int) -> list[dict]:
    selected: list[dict] = []
    seen_urls: set[str] = set()
    total_chars = 0

    for row in evidence:
        url = row["url"]

        if url in seen_urls:
            continue

        text = str(row.get("text", "")).strip()

        if not text:
            continue

        remaining = (max_total_chars - total_chars)

        if remaining <= 0:
            break

        row = dict(row)
        row["text"] = text[:remaining]

        selected.append(row)
        seen_urls.add(url)

        total_chars += len(row["text"])

    return selected

def collect_parallel_evidence(*, executor: ThreadPoolExecutor, question: str, policies: list[SourcePolicy], validation_cfg: dict) -> list[dict]:
    rows = search_one_source(
        question=question,
        policies=policies,
        searxng_url=str(validation_cfg["searxng_url"]),
        results_per_source=int(validation_cfg.get("results_per_source", 3)),
        fetched_pages_per_source=int(validation_cfg.get("fetched_pages_per_source", 2)),
        search_timeout_s=float(validation_cfg.get("searxng_timeout_s", 30)),
        fetch_timeout_s=float(validation_cfg.get("fetch_timeout_s", 25)),
        max_chars_per_page=int(validation_cfg.get("max_chars_per_page", 3000)),
        search_interval_s=float(validation_cfg.get("search_interval_s", 3.0)),
    )

    rows.sort(key=lambda row: (int(row.get("search_rank", 999)), 0 if row.get("tier") == "primary" else 1, -float(row.get("source_weight", 0.0)), str(row.get("source", "")),))
    log.info("[INTERNET_VALIDATION] finished evidence=%d", len(rows))
    return deduplicate_and_trim_evidence(rows, max_total_chars=int(validation_cfg.get("max_total_evidence_chars", 24000)),)