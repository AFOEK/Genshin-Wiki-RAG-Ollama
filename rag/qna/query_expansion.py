from __future__ import annotations

import json
import logging
import re

from typing import Any

from qna.generators import generate

log = logging.getLogger(__name__)

QUERY_EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "maxItems": 5,
        },
    },
    "required": [
        "queries",
    ],
    "additionalProperties": False,
}

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

def build_retrieval_queries(cfg: dict[str, Any], question: str, *, max_expansions: int = 3, model: str | None = None) -> list[str]:
    original = normalize_query(question)
    expansions = expand_query(cfg, original, max_expansions=max_expansions, model=model,)
    return dedupe_queries([original, *expansions,])

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

def expand_query(cfg: dict[str, Any], question: str, *, max_expansions: int = 3, model: str | None = None) -> list[str]:
    question = normalize_query(question)

    if not question:
        return []

    prompt = f"""
Generate alternative search queries for a Genshin Impact
retrieval system.

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
  "queries": ["query 1", "query 2"]
}}
""".strip()

    raw = generate(cfg, prompt, model_override=model, think_override=False,
        options_override={
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 30,
            "num_predict": 256,
        })

    try:
        result = json.loads(str(raw).strip())
    except json.JSONDecodeError:
        log.warning("[QUERY_EXPANSION] invalid JSON " "question=%r raw=%r", question, str(raw)[:500],)
        return []

    queries = result.get("queries", [],)
    if not isinstance(queries, list,):
        return []

    queries = dedupe_queries([str(query) for query in queries])
    original_key = (question.casefold())
    queries = [query for query in queries if query.casefold() != original_key]
    return queries[:max_expansions]