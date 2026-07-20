from __future__ import annotations

import json
import logging
import re

from .generators import generate, strip_thinking_blocks
from .utils import as_bool

log = logging.getLogger(__name__)

COMPLEX_PATTERN = re.compile(
    r"\b("
    r"compare|comparison|versus|vs\.?|difference|differences|"
    r"similarities|both|respectively|advantages|disadvantages|"
    r"pros and cons|relationship|as well as"
    r")\b", re.IGNORECASE
)

CHANNELS = ("faiss", "bm25", "splade",  "hyde", "turbovec")

def should_decompose(question: str, *, mode: str, min_words: int) -> bool:
    mode = mode.strip().lower()

    if mode in {"off", "disabled", "never"}:
        return False
    
    if mode == "always":
        return True
    
    words = re.findall(r"\b\w+\b", question)
    if len(words) < min_words:
        return False
    
    if question.count("?") > 1:
        return True
    
    if COMPLEX_PATTERN.search(question):
        return True
    
    if re.search(r"\b(and|while|whereas)\b", question, re.IGNORECASE):
        return True
    
    return False

def parse_query_array(raw: str, *, original_question: str, max_subqueries: int) -> list[str]:
    cleaned = strip_thinking_blocks(raw)
    start = cleaned.find("[")
    end = cleaned.rfind("]")

    if start < 0 or end < start:
        return []

    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, list):
        return []
    
    original_key = " ".join(original_question.casefold().split())
    queries: list[str] = []
    seen = {original_key}

    for value in payload:
        if not isinstance(value, str):
            continue

        query = re.sub(r"\s+", " ", value,).strip(" \t\r\n-")

        if len(query) < 5 or len(query) > 300:
            continue

        key = " ".join(query.casefold().split())

        if key in seen:
            continue

        seen.add(key)
        queries.append(query)

        if len(queries) >= max_subqueries:
            break

    return queries

def decompose_query(cfg: dict, question: str, *, backend: str | None = None) -> list[str]:
    decomp_cfg = (cfg.get("query_decomposition", {}) or {})

    if not as_bool(decomp_cfg.get("enabled"), False):
        return []

    mode = str(decomp_cfg.get("mode", "auto")).strip().lower()

    max_subqueries = max(2, int(decomp_cfg.get("max_subqueries", 3)))

    if not should_decompose(question, mode=mode, min_words=int(decomp_cfg.get("min_words", 8))):
        log.info("[DECOMP] skipped simple question=%r", question)
        return []

    prompt = f"""
You decompose complex Genshin Impact questions into independent retrieval queries.

Return only a valid JSON array of strings.

Rules:
- Return 2 to {max_subqueries} queries only when decomposition improves retrieval.
- Return [] when the original question is already one focused retrieval query.
- Each query must be understandable independently.
- Preserve exact character, weapon, artifact, enemy, location, version, constellation, refinement, and talent names.
- Replace pronouns such as "it", "they", or "them" with the relevant entity name.
- Do not answer the question.
- Do not invent facts.
- Do not include explanations or Markdown.

Original question:
{question}
""".strip()

    model_value = str(decomp_cfg.get("model", "")).strip()

    try:
        raw = generate(
            cfg,
            prompt,
            provider_override=backend,
            model_override=(model_value or None),
            timeout=int(decomp_cfg.get("timeout", 120)),
            think_override=decomp_cfg.get("think", False),
            options_override={
                "temperature": float(
                    decomp_cfg.get(
                        "temperature",
                        0.0,
                    )
                ),
                "top_p": float(
                    decomp_cfg.get(
                        "top_p",
                        0.9,
                    )
                ),
                "top_k": int(
                    decomp_cfg.get(
                        "top_k",
                        30,
                    )
                ),
                "min_p": float(
                    decomp_cfg.get(
                        "min_p",
                        0.05,
                    )
                ),
                "repeat_penalty": float(
                    decomp_cfg.get(
                        "repeat_penalty",
                        1.0,
                    )
                ),
                "num_predict": int(
                    decomp_cfg.get(
                        "num_predict",
                        256,
                    )
                ),
            })
    except Exception as exc:
        log.warning("[DECOMP] generation failed; using original query: %s: %s", type(exc).__name__, exc,)
        return []

    queries = parse_query_array(raw, original_question=question, max_subqueries=max_subqueries,)
    log.info("[DECOMP] original=%r subqueries=%s", question, queries,)
    return queries

def merge_decomposition_runs(runs: list[tuple[str, list[tuple[int, float]], dict[int, dict], float,]], *, rrf_k: int, rrf_scale: float, max_total_candidates: int) -> tuple[list[tuple[int, float]], dict[int, dict]]:
    merged: dict[int, dict] = {}

    for query_number, (query_text, results, signals, query_weight,) in enumerate(runs):
        for rank, (chunk_id, _) in enumerate(results, start=1,):
            chunk_id = int(chunk_id)

            destination = merged.setdefault(
                chunk_id,
                {
                    "rrf_score": 0.0,
                    "decomposition_hits": 0,
                    "decomposition_best_rank": None,
                    "decomposition_query_ids": [],
                },
            )

            for channel in CHANNELS:
                destination.setdefault(f"{channel}_score", 0.0)
                destination.setdefault(f"{channel}_rank", None)
                destination.setdefault(f"in_{channel}", False)

            destination["rrf_score"] += (rrf_scale * float(query_weight) / (rrf_k + rank))

            destination["decomposition_hits"] += 1
            destination["decomposition_query_ids"].append(query_number)
            best_rank = destination["decomposition_best_rank"]
            destination["decomposition_best_rank"] = (rank if best_rank is None else min(best_rank, rank))
            source = signals.get(chunk_id, {})

            for channel in CHANNELS:
                in_key = f"in_{channel}"
                rank_key = f"{channel}_rank"
                score_key = f"{channel}_score"
                destination[in_key] = (bool(destination[in_key]) or bool(source.get(in_key)))
                source_rank = source.get(rank_key)

                if source_rank is not None:
                    current_rank = destination[rank_key]
                    destination[rank_key] = (int(source_rank) if current_rank is None else min(int(current_rank), int(source_rank)))

                destination[score_key] = max(float(destination.get(score_key, 0.0,)), float(source.get(score_key, 0.0,) or 0.0))

    ranked = sorted(((chunk_id, signal["rrf_score"]) for chunk_id, signal in merged.items()), key=lambda item: item[1], reverse=True)

    if max_total_candidates > 0:
        ranked = ranked[:max_total_candidates]

    retained_ids = {chunk_id for chunk_id, _ in ranked}
    retained_signals = {chunk_id: merged[chunk_id] for chunk_id in retained_ids}
    return ranked, retained_signals