from __future__ import annotations

import logging
import time
import requests

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any
from bs4 import BeautifulSoup

from policy_loader import SourcePolicy
from url_validation import is_allowed_url

log = logging.getLogger(__name__)

def extract_page_text(html: str, *, max_chars: int) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for element in soup(["script", "style", "noscript", "svg", "nav", "footer", "form"]):
        element.decompose()

    text = " ".join(soup.get_text(separator=" ", strip=True).split())
    return text[:max_chars]

def fetch_result_page(*, session: requests.Session, url: str, policy: SourcePolicy, timeout_s: float, max_chars: int) -> tuple[str, str]:
    response = session.get(url, timeout=timeout_s, allow_redirects=True, headers={"User-Agent": "GenshinDatasetValidator/1.0"})
    response.raise_for_status()

    final_url = response.url
    if not is_allowed_url(final_url, policy):
        raise ValueError(f"Result redirected outside approved source: {final_url}")

    content_type = response.headers.get("content-type", "").lower()
    text = (extract_page_text(response.text, max_chars=max_chars) if "text/html" in content_type else response.text[:max_chars])

    return final_url, text

def search_one_source(*, question: str, policy: SourcePolicy, searxng_url: str, results_per_source: int, fetched_pages_per_source: int, search_timeout_s: float, fetch_timeout_s: float, max_chars_per_page: int) -> list[dict[str, Any]]:
    session = requests.Session()
    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for domain in policy.search_domains:
        query = f"site:{domain} Genshin Impact {question}"
        response = session.get(
            f"{searxng_url.rstrip('/')}/search",
            params={
                "q": query,
                "format": "json",
                "language": "en",
                "safesearch": 1,
            },
            timeout=search_timeout_s,
        )
        response.raise_for_status()
        payload = response.json()

        for result in payload.get("results", []):
            url = str(result.get("url", "")).strip()

            if not url or url in seen_urls or not is_allowed_url(url, policy):
                continue

            seen_urls.add(url)
            candidates.append(
                {
                    "source": policy.name,
                    "tier": policy.tier,
                    "source_weight": policy.weight,
                    "title": str(result.get("title", "")),
                    "url": url,
                    "snippet": str(result.get("content", "")),
                }
            )

            if len(candidates) >= results_per_source:
                break

        if len(candidates) >= results_per_source:
            break

    evidence: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates[:fetched_pages_per_source]):
        if index:
            time.sleep(policy.rate_limit_s)

        try:
            final_url, page_text = fetch_result_page(session=session, url=candidate["url"], policy=policy, timeout_s=fetch_timeout_s, max_chars=max_chars_per_page,)
            candidate["url"] = final_url
            candidate["text"] = page_text
            candidate["fetch_ok"] = True
        except Exception as exc:
            candidate["text"] = candidate["snippet"]
            candidate["fetch_ok"] = False
            candidate["fetch_error"] = f"{type(exc).__name__}: {exc}"

        candidate["evidence_id"] = f"{policy.name}:{index + 1}"
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
    futures = {
        executor.submit(
            search_one_source,
            question=question,
            policy=policy,
            searxng_url=str(validation_cfg["searxng_url"]),
            results_per_source=int(validation_cfg.get("results_per_source", 3)),
            fetched_pages_per_source=int(validation_cfg.get("fetched_pages_per_source", 2)),
            search_timeout_s=float(validation_cfg.get("searxng_timeout_s", 30)),
            fetch_timeout_s=float(validation_cfg.get("fetch_timeout_s", 25)),
            max_chars_per_page=int(validation_cfg.get("max_chars_per_page", 3000)),
        ): policy for policy in policies}

    collected: list[dict] = []

    for future in as_completed(futures):
        policy = futures[future]
        try:
            rows = future.result()
        except Exception as exc:
            log.warning("[INTERNET_VALIDATION] source=%s failed: %s", policy.name, exc)
            continue

        collected.extend(rows)
        log.info("[INTERNET_VALIDATION] source=%s finished evidence=%d", policy.name, len(rows),)

    return deduplicate_and_trim_evidence(collected, max_total_chars=int(validation_cfg.get("max_total_evidence_chars", 24000)))