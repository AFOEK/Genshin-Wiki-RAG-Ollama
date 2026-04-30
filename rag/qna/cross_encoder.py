from __future__ import annotations

import logging
from functools import lru_cache

log = logging.getLogger(__name__)

@lru_cache(maxsize=2)
def get_cross_encoder(model_name:str):
    from sentence_transformers import CrossEncoder
    log.info("[CROSS_ENCODER] Loading cross-encoder model=%s", model_name)
    return CrossEncoder(model_name)

def cross_encoder_rerank(
        question: str,
        chunks: list[dict],
        *,
        model_name: str,
        top_n: int = 32,
        batch_size: int = 8,
        max_pair_text_chars: int = 1200
) -> list[dict]:
    if not chunks:
        return chunks
    
    candidates = chunks[:top_n]
    rest = chunks[top_n:]

    pairs = []
    for row in candidates:
        title = row.get("title") or ""
        source = row.get("source") or ""
        text = row.get("text") or ""

        pair_text = f"Title: {title}\nSource: {source}\nText:\n{text}"
        pair_text = pair_text[:max_pair_text_chars]
        pairs.append((question, pair_text))

    model = get_cross_encoder(model_name)
    scores = model.predict(pairs, batch_size=batch_size)

    scored = []
    for row, score in zip(candidates, scores):
        r = dict(row)
        r["cross_encoder_score"] = float(score)
        scored.append(r)

    scored.sort(key=lambda r: r["cross_encoder_score"], reverse=True)

    return scored + rest
