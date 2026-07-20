from __future__ import annotations

import json
import logging
import re

from .generators import generate, strip_thinking_blocks
from .utils import as_bool

log = logging.getLogger(__name__)

MULTI_HOP_PATTERN = re.compile(
    r"\b(whose|dropped by|obtained from|located in|unlocked by|"
    r"required by|needed for|used by|leads to|after completing|"
    r"and where|and who|and how|which .+ that)\b",
    re.IGNORECASE
)

CHANNELS = ("faiss", "bm25", "splade", "hyde", "turbovec")

def should_use_multi_hop(question: str, *, mode: str, min_words: int) -> bool:
    mode = mode.strip().lower()
    if mode in {"off", "disable", "never"}:
        return False
    if mode == "always":
        return True
    if len(re.findall(r"\b\w+\b", question)) < min_words:
        return False
    return bool(MULTI_HOP_PATTERN.search(question))

def parse_queries(raw: str, *, original_question: str, prior_queries: list[str], max_queries: int) -> list[str]:
    cleaned = strip_thinking_blocks(raw)
    start = cleaned.find("[")
    end = cleaned.find("]")

    if start < 0 and end < start:
        return []
    
    try:
        payload = json.load(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return []
    
    seen = {" ".join(value.casefold().split()) for value in [original_question, *prior_queries]}
    queries =[]

    for value in payload:
        if not isinstance(value, str):
            continue

        query = re.sub(r"\s+", " ", value).strip(" \t\r\n-")
        key = " ".join(query.casefold().split())
        if len(query) < 5 or len(query) > 300 or key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= max_queries:
            break

    return queries

def build_evidence(chunks: list[dict], *, max_chunks: int, chars_per_chunk: int) -> str:
    blocks = []
    for row in chunks[:max_chunks]:
        chunk_id = int(row["chunk_id"])
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()[:chars_per_chunk]
        if text:
            blocks.append(f"[Chunk {chunk_id}]\nTitle: {title}\n{text}")
    return "\n\n".join(blocks)

def generate_bridge_queries(cfg: dict, question: str, evidence_chunks: list[dict], *, prior_queries: list[str] | None = None, backend: str | None = None) -> list[str]:
    hop_cfg = cfg.get("multi_hop", {}) or {}
    if not as_bool(hop_cfg.get("enabled"), False):
        return []

    mode = str(hop_cfg.get("mode", "auto")).strip().lower()
    if not should_use_multi_hop(question, mode=mode, min_words=int(hop_cfg.get("min_words", 7))):
        log.info("[MULTIHOP] skipped question=%r", question)
        return []

    evidence = build_evidence(
        evidence_chunks,
        max_chunks=int(hop_cfg.get("evidence_k", 6)),
        chars_per_chunk=int(hop_cfg.get("evidence_chars_per_chunk", 900)),
    )
    if not evidence:
        return []

    prior_queries = list(prior_queries or [])
    max_queries = max(1, int(hop_cfg.get("max_bridge_queries", 2)))
    prompt = f"""
You generate bridge retrieval queries for multi-hop Genshin Impact question answering.

Return only a valid JSON array of strings.

Rules:
- Generate 1 to {max_queries} bridge queries only when another retrieval step is necessary.
- Return [] when the supplied evidence already directly answers the original question.
- Each bridge query must depend on a concrete entity or fact discovered in the evidence.
- Use exact character, item, weapon, artifact, enemy, boss, quest, talent, material, and location names.
- Replace pronouns with explicit entity names.
- Ask for one missing fact per query.
- Do not answer the original question.
- Do not invent entities.
- Do not repeat the original query or prior retrieval queries.
- Do not include Markdown or explanations.

Original question:
{question}

Prior retrieval queries:
{json.dumps(prior_queries, ensure_ascii=False)}

First-hop evidence:
{evidence}
""".strip()

    model_name = str(hop_cfg.get("model", "")).strip()

    try:
        raw = generate(
            cfg,
            prompt,
            provider_override=backend,
            model_override=model_name or None,
            timeout=int(hop_cfg.get("timeout", 120)),
            think_override=hop_cfg.get("think", False),
            options_override={
                "temperature": float(hop_cfg.get("temperature", 0.0)),
                "top_p": float(hop_cfg.get("top_p", 0.9)),
                "top_k": int(hop_cfg.get("top_k", 30)),
                "min_p": float(hop_cfg.get("min_p", 0.05)),
                "num_predict": int(hop_cfg.get("num_predict", 256)),
            },
        )
    except Exception as exc:
        log.warning("[MULTIHOP] bridge generation failed: %s: %s", type(exc).__name__, exc)
        return []

    queries = parse_queries(raw, original_question=question, prior_queries=prior_queries, max_queries=max_queries)
    log.info("[MULTIHOP] original=%r bridge_queries=%s", question, queries)
    return queries

def ensure_signal(signals: dict[int, dict], chunk_id: int) -> dict:
    signal = signals.setdefault(chunk_id, {"rrf_score": 0.0})
    for channel in CHANNELS:
        signal.setdefault(f"{channel}_score", 0.0)
        signal.setdefault(f"{channel}_rank", None)
        signal.setdefault(f"in_{channel}", False)
    signal.setdefault("multi_hop_score", 0.0)
    signal.setdefault("multi_hop_hits", 0)
    signal.setdefault("multi_hop_best_rank", None)
    signal.setdefault("multi_hop_query_ids", [])
    return signal


def merge_source_signal(destination: dict, source: dict) -> None:
    for key, value in source.items():
        destination.setdefault(key, value)

    for channel in CHANNELS:
        in_key = f"in_{channel}"
        rank_key = f"{channel}_rank"
        score_key = f"{channel}_score"

        destination[in_key] = bool(destination.get(in_key)) or bool(source.get(in_key))

        source_rank = source.get(rank_key)
        if source_rank is not None:
            current_rank = destination.get(rank_key)
            destination[rank_key] = int(source_rank) if current_rank is None else min(int(current_rank), int(source_rank))

        destination[score_key] = max(float(destination.get(score_key, 0.0) or 0.0), float(source.get(score_key, 0.0) or 0.0))


def merge_multi_hop_results(base_results: list[tuple[int, float]], base_signals: dict[int, dict], hop_runs: list[tuple[str, list[tuple[int, float]], dict[int, dict]]], *, rrf_k: int, rrf_scale: float, hop_weight: float, max_total_candidates: int) -> tuple[list[tuple[int, float]], dict[int, dict]]:
    merged: dict[int, dict] = {}

    for chunk_id, score in base_results:
        chunk_id = int(chunk_id)
        source = dict(base_signals.get(chunk_id, {}))
        destination = ensure_signal(merged, chunk_id)
        destination.update(source)
        destination["rrf_score"] = float(source.get("rrf_score", score))

    for query_id, (_, results, signals) in enumerate(hop_runs):
        for rank, (chunk_id, _) in enumerate(results, start=1):
            chunk_id = int(chunk_id)
            destination = ensure_signal(merged, chunk_id)
            source = signals.get(chunk_id, {})
            merge_source_signal(destination, source)

            bonus = rrf_scale * hop_weight / (rrf_k + rank)
            destination["rrf_score"] = float(destination.get("rrf_score", 0.0)) + bonus
            destination["multi_hop_score"] += bonus
            destination["multi_hop_hits"] += 1
            destination["multi_hop_query_ids"].append(query_id)

            best_rank = destination["multi_hop_best_rank"]
            destination["multi_hop_best_rank"] = rank if best_rank is None else min(best_rank, rank)

    ranked = sorted(((chunk_id, signal["rrf_score"]) for chunk_id, signal in merged.items()), key=lambda item: item[1], reverse=True)
    if max_total_candidates > 0:
        ranked = ranked[:max_total_candidates]

    retained = {chunk_id for chunk_id, _ in ranked}
    return ranked, {chunk_id: merged[chunk_id] for chunk_id in retained}