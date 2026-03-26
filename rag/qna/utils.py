from __future__ import annotations

import logging
import sqlite3
import re
from typing import Iterable

import numpy as np
import yaml

log = logging.getLogger(__name__)

def load_cfg(path: str = "rag/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_only_connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn

def normalize_vec_from_blob(blob: bytes, dims: int) -> np.ndarray:
    v = np.frombuffer(blob, dtype=np.float32)
    if v.size != dims:
        raise ValueError(f"vector size mismatch: expected {dims}, got {v.size}")
    v = v.astype(np.float32, copy=False)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v

def normalize_query_vec(blob: bytes, dims: int) -> np.ndarray:
    q = np.frombuffer(blob, dtype=np.float32)
    if q.size != dims:
        raise ValueError(f"query dim mismatch: expected {dims}, got {q.size}")
    q = q.astype(np.float32, copy=False)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return q.reshape(1, -1)

def is_broad_question(q: str) -> bool:
    ql = q.lower()
    broad_markers = [
        "all the lore",
        "from beginning until now",
        "full lore",
        "entire lore",
        "everything about",
        "complete history",
        "chronology",
        "timeline",
    ]
    return any(m in ql for m in broad_markers)

def chunk_batch(seq: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def tokenize(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_']+", s.lower()))

def rerank_chunks(question: str, chunks: list[dict], initial_scores: dict[int, float]) -> list[dict]:
    q_terms = tokenize(question)
    media_exts = [".jpeg", ".jpg", ".png", ".webp", ".mp4", ".svg", ".ico" , ".webm", ".mp3", ".gif", ".wav", ".ogg"]
    ranked = []

    for row in chunks:
        chunk_id = int(row["chunk_id"])
        base_score = float(initial_scores.get(chunk_id, 0.0))
        text = row.get("text") or ""
        title = row.get("title") or ""
        tier = row.get("tier") or ""
        weight = float(row.get("weight") or 1.0)
        penalty = 0.0

        text_terms = tokenize(row.get("text", ""))
        title_terms = tokenize(row.get("title", ""))

        text_overlap = len(q_terms & text_terms)
        title_overlap = len(q_terms & title_terms)

        lexical_bonus = (
            0.02 * text_overlap
            + 0.1 * title_overlap
        )

        weighted_based = base_score * weight

        tier_bonus = 0.05 if tier == "primary" else 0.02 if tier == "secondary" else 0.0

        media_counts = sum(text.count(ext) for ext in media_exts)
        if media_counts > 3:
            penalty += 0.2
        
        if "character card" in title.lower() or "genius invokation" in title.lower():
            penalty += 0.15

        url_chars = sum(1 for c in text if c in "[]%?=&/:.#_+")
        if len(text.strip()) > 50 and url_chars / len(text) > 0.3:
            penalty += 0.15

        if len(text.strip()) < 100:
            penalty += 0.1
        
        final_score = final_score = weighted_based + lexical_bonus + tier_bonus + initial_scores - penalty
        ranked.append((final_score, row))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in ranked]