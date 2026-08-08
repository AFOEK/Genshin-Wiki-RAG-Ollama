from __future__ import annotations

import json
import logging
import re

from typing import Any

from .generators import generate
from .utils import as_bool

log = logging.getLogger(__name__)

def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", str(query)).strip()

def dedupe_queries(queries: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()

    for query in queries:
        query = normalize_query(query)
        if not query:
            continue

        key = query.casefold()

        if key in seen:
            continue

        seen.add(key)
        output.append(query)

    return output

def build_retrieval_queries(cfg: dict[str, Any], question: str, *, max_expansions: int | None = None, model: str | None = None) -> list[str]:
    expansion_cfg = cfg.get("query_expansion", {}) or {}
    original = normalize_query(question)
    expansions = expand_query(cfg, original, max_expansions=max_expansions, model=model)
    include_original = as_bool(expansion_cfg.get("include_original", True))
    if include_original:
        return dedupe_queries([original, *expansions])

    return dedupe_queries(expansions)

def parse_expansion_json(raw: str) -> dict[str, Any]:
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE,)
    text = re.sub(r"\s*```$", "", text,).strip()
    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end < start:
        raise json.JSONDecodeError("No JSON object found", text, 0,)

    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Query expansion must return a JSON object")

    return value

def expand_query(cfg: dict[str, Any], question: str, *, max_expansions: int | None = None, model: str | None = None) -> list[str]:
    expansion_cfg = cfg.get("query_expansion", {}) or {}

    question = normalize_query(question)
    max_query_chars = max(1, int(expansion_cfg.get("max_query_chars", 300)))
    question = question[:max_query_chars].strip()

    if not question:
        return []

    if max_expansions is None:
        max_expansions = int(expansion_cfg.get("max_expansions", 2))

    max_expansions = max(0, min(5, int(max_expansions)))
    if max_expansions == 0:
        return []

    if model is None:
        configured_model = str(expansion_cfg.get("model", "")).strip()
        model = configured_model or None

    prompt = f"""
Generate alternative search queries for a Genshin Impact retrieval system.

Original query:
{question}

Generate at most {max_expansions} alternative queries.

Rules:
- Preserve the exact user intent.
- Preserve important entity names.
- Do not answer the question.
- Do not add facts not present in the original query.
- Do not turn one question into unrelated questions.
- Prefer terminology likely to appear in Genshin guides, wiki pages, database pages, builds, quests, or game data.
- Keep each query concise.
- Do not include the original query.
- Return JSON only.

Format:
{{
  "queries": [
    "query 1",
    "query 2"
  ]
}}
""".strip()

    raw = generate(
        cfg,
        prompt,
        model_override=model,
        think_override=False,
        options_override={
            "temperature": float(expansion_cfg.get("temperature", 0.2)),
            "top_p": float(expansion_cfg.get("top_p", 0.9)),
            "top_k": int(expansion_cfg.get("top_k", 30)),
            "num_predict": int(expansion_cfg.get("num_predict", 256)),
        },
    )

    try:
        result = parse_expansion_json(str(raw))
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("[QUERY_EXPANSION] invalid JSON question=%r error=%s raw=%r", question, exc, str(raw)[:500])
        return []

    queries = result.get("queries", [])
    if not isinstance(queries, list):
        return []

    queries = dedupe_queries([str(query) for query in queries])
    original_key = question.casefold()
    queries = [query for query in queries if query.casefold() != original_key]

    return queries[:max_expansions]