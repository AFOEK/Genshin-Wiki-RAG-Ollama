from __future__ import annotations

import logging
import sqlite3
import re
from typing import Iterable

import numpy as np
import yaml

log = logging.getLogger(__name__)

INTENT_PROFILES = {
    "build": {
        "source_bonus": {"kqm_tcl": 0.15, "game8": 0.15, "genshin_gg":0.08},
        "source_penalty": {"kqm_news": 0.15, "genshin_wiki": 0.15},
        "title_penalize": ["storyline", "voice", "voice-over", "story quest", "archon quest",
                           "world quest", "character card", "genius invokation", "comments",
                           "lore", "dialogue", "comments", "normal attack", "utility passive", "constellation",
                           "skill", "burst", "storyline", "voice", "quest", "character card", "genius invokation",
                           "elemental skill", "elemental burst"],
        "title_boost": ["build", "guide", "weapon", "weapons", "artifact", "artifacts", "recommended", "signature weapon"],
        "title_boost_v": 0.20,
        "text_require_any": ["weapon", "polearm", "sword", "bow", "catalyst",
                             "claymore", "artifact", "set bonus", "recommended",
                             "signature weapon", "bis weapon", "4pc", "2pc", 
                             "main stat", "substat", "dmg bonus", "dmg", "attack", "crit rate",
                             "elemental mastery", "em", "cr", "crit dmg", "cr%", "def%", "hp%",
                             "atk%", "staff"],
        "text_require_penalty": 0.30,
        "source_priority": {
            "required": ["kqm_tcl", "genshin_gg", "game8"],
            "preferred": ["honey"],
            "excluded": ["kqm_news"]
        },
    },
    "lore": {
        "source_bonus": {"genshin_wiki": 0.12},
        "source_penalty": {"honey": 0.08, "kqm_tcl": 0.05, "kqm_news": 0.03, "game8": 0.05, "genshin_gg":0.1},
        "title_penalize": ["change history", "voice-overs", "character card",
                           "genius invokation", "normal attack", "constellation",
                           "ascension", "recommended", "signature weapon", "bis weapon",
                           "comments", "skill", "burst", "elemental skill",
                           "elemental burst", "utility passive", "charged attack", "plunging attack"],
        "title_boost": ["storyline", "lore", "story", "history", "quest",
                        "story quest", "world quest", "archon quest", "character story"],
        "title_boost_v": 0.15,
        "text_require_any": [],
        "text_require_penalty": 0.0,
        "source_priority":{
            "required": ["genshin_wiki"],
            "preferred": [],
            "excluded": ["honey", "game8", "kqm_tcl", "kqm_news", "genshin_gg"]
        },
    },
    "mechanic":{
        "source_bonus": {"genshin_wiki": 0.05, "kqm_tcl": 0.12, "game8": 0.07, "genshin_gg":0.06},
        "source_penalty": {"kqm_news": 0.05, "honey": 0.08},
        "title_penalize": ["storyline", "voice", "farewell", "wishes",
                           "character card", "story", "world quest", "archon quest", "character story", "comments"],
        "title_boost": ["talent", "constellation", "passive", "ability",
                        "normal attack", "skill", "burst", "elemental skill",
                        "elemental burst", "utility passive", "charged attack", "plunging attack"],
        "title_boost_v": 0.08,
        "text_require_any": [],
        "text_require_penalty": 0.0,
        "source_priority":{
            "required": ["game8", "kqm_tcl", "genshin_gg"],
            "preferred": ["genshin_wiki"],
            "excluded": ["kqm_news"]
        },
    },
    "location": {
        "source_bonus":   {"genshin_wiki": 0.20, "game8": 0.1},
        "source_penalty": {"kqm_tcl": 0.1, "kqm_news": 0.08},
        "title_penalize": [
            "change history", "voice-overs", "character card", "genius invokation",
            "normal attack", "constellation", "ascension", "comments",
            "character story", "storyline",
        ],
        "title_boost": [
            "region", "nation", "area", "domain", "dungeon", "cave",
            "ruins", "city", "village", "mountain", "island", "forest",
            "archon", "map", "exploration", "liyue", "mondstadt", "inazuma",
            "sumeru", "fontaine", "natlan", "snezhnaya", "khaenri'ah", "nod-krai",
            "isle", "enkanomiya", "sub-area", "sub-region", "the chasm", "chenyu vale", "dharma forest",
            "great red sand", "girdle of the sands", "sea of bygone eras sea of bygone eras", "ancient sacred mountain",
            "temple of space", "golden apple archipelago", "three realms gateway offering", "veluriyam mirage",
            "simulanka", "dragonspine", "windrest peak", "celestia", "abyss", "hyperborea", "sea of stars",
            "the sea of flowers at the end"
        ],
        "title_boost_v": 0.15,
        "text_require_any": [
            "located", "region", "nation", "area", "north", "south", "east", "west",
            "near", "found in", "place", "city", "village", "ruins", "domain",
            "map", "exploration", "coordinates", "waypoint", "liyue", "mondstadt", "inazuma",
            "sumeru", "fontaine", "natlan", "snezhnaya", "khaenri'ah", "nod-krai", "enkanomiya", "the chasm",
            "chenyu vale", "dharma forest", "great red sand", "girdle of the sands", 
            "sea of bygone eras sea of bygone eras", "ancient sacred mountain",
            "temple of space", "golden apple archipelago", "three realms gateway offering", "veluriyam mirage",
            "simulanka", "dragonspine", "windrest peak", "celestia", "abyss", "hyperborea", "sea of stars",
            "the sea of flowers at the end"
        ],
        "text_require_penalty": 0.20,
        "source_priority":{
            "required": ["genshin_wiki"],
            "preferred": ["game8"],
            "excluded": ["kqm_news", "kqm_tcl", "honey"]
        },
    },

    "biography": {
        "source_bonus":   {"genshin_wiki": 0.15},
        "source_penalty": {"kqm_tcl": 0.15, "honey": 0.05, "kqm_news": 0.05},
        "title_penalize": [
            "change history", "character card", "genius invokation",
            "normal attack", "constellation", "ascension", "comments",
            "voice-overs",
        ],
        "title_boost": [
            "storyline", "character story", "lore", "profile",
            "birthday", "background", "history",
        ],
        "title_boost_v": 0.20,
        "text_require_any": [
            "birthday", "born", "age", "from", "vision", "constellation",
            "affiliation", "family", "related to", "sibling", "parent",
            "friend", "title", "occupation", "voiced by",
        ],
        "text_require_penalty": 0.15,
        "source_priority":{
            "required": ["genshin_wiki"],
            "preferred": ["honey", "game8", "genshin_gg"],
            "excluded": ["kqm_tcl", "kqm_news"]
        },
    },
    "general": {
        "source_bonus":   {"genshin_wiki": 0.05, "game8":0.04},
        "source_penalty": {"kqm_news": 0.03},
        "title_penalize": ["character card", "genius invokation"],
        "title_boost":    [],
        "title_boost_v":  0.0,
        "text_require_any": [],
        "text_require_penalty": 0.0,
        "source_priority": {
            "required": [],
            "preferred": ["genshin_wiki", "game8"],
            "excluded": []
        },
    },
}

BM25_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "who", "what", "when", "where", "why", "how",
    "i", "you", "he", "she", "it", "they", "we",
    "of", "to", "in", "on", "for", "from", "with", "and", "or",
    "does", "do", "did", "can", "could", "would", "should",
    "come", "comes",
}

def rrf_fuse(*result_lists, k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}

    for results in result_lists:
        for rank, (chunk_id, _score) in enumerate(results, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused

def build_hybrid_signal(faiss_results: list[tuple[int, float]], bm25_results: list[tuple[int, float]], *, rrf_k: int = 60, rrf_scale: float = 10.0) -> dict[int, dict]:
    signals: dict[int, dict] = {}

    def ensure(cid: int) -> dict:
        if cid not in signals:
            signals[cid] = {
                "rrf_score": 0.0,
                "faiss_score": 0.0,
                "bm25_score": 0.0,
                "faiss_rank": None,
                "bm25_rank": None,
                "in_faiss": False,
                "in_bm25": False
            }
        return signals[cid]
    
    for rank, (cid, score) in enumerate(faiss_results, start=1):
        cid = int(cid)
        s = ensure(cid)
        s["faiss_score"] = float(score)
        s["faiss_rank"] = rank
        s["in_faiss"] = True
        s["rrf_score"] += 1.0 / (rrf_k + rank)

    for rank, (cid, score) in enumerate(bm25_results, start=1):
        cid = int(cid)
        s = ensure(cid)
        s["bm25_score"] = float(score)
        s["bm25_rank"] = rank
        s["in_bm25"] = True
        s["rrf_score"] += 1.0 / (rrf_k + rank)

    for s in signals.values():
        s["rrf_score"] *= rrf_scale

    return signals

def make_fts5_query(user_query: str) -> str:
    raw_tokens = re.findall(r"[A-Za-z0-9_']+", user_query.lower())

    tokens = []
    for t in raw_tokens:
        t = t.strip("'")
        if not t:
            continue
        if t in BM25_STOPWORDS:
            continue
        if len(t) < 2:
            continue
        t = t.replace('"', '""')
        tokens.append(t)

    if not tokens:
        return ""

    parts = []

    if 1 <= len(tokens) <= 4:
        phrase = " ".join(tokens)
        parts.append(f'"{phrase}"')

    parts.extend(f'"{t}"' for t in tokens)

    seen = set()
    uniq = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)

    return " OR ".join(uniq)

def filter_by_intent_source(conn: sqlite3.Connection, chunk_ids: list[int], intent: str, min_required: int=5, max_fallback: int=30) -> list[int]:
    if not chunk_ids:
        return []
    
    profile = INTENT_PROFILES.get(intent, INTENT_PROFILES["general"])
    priority = profile.get("source_priority", {})

    required = set(priority.get("required", []))
    preferred = set(priority.get("preferred", []))
    excluded = set(priority.get("excluded", []))

    if not required and not excluded:
        return chunk_ids
    
    placeholder = ",".join("?" for _ in chunk_ids)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT c.chunk_id, d.source
        FROM chunks c
        JOIN docs d ON d.doc_id = c.doc_id
        WHERE c.chunk_id IN ({placeholder})
    """, chunk_ids)
    source_map = {row["chunk_id"]: row["source"] for row in cur}

    required_ids = []
    preferred_ids = []
    neutral_ids = []

    for cid in chunk_ids:
        src = source_map.get(cid, "")
        if src in excluded:
            continue
        elif src in required:
            required_ids.append(cid)
        elif src in preferred:
            preferred_ids.append(cid)
        else:
            neutral_ids.append(cid)
    
    result = required_ids[:]

    if len(result) < min_required:
        result.extend(preferred_ids)

    if len(result) < min_required:
        needed = max_fallback - len(result)
        result.extend(neutral_ids[:needed])
    
    return result

def load_cfg(path: str = "rag/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def as_bool(x) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")

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
        "from beginning until",
        "from the beginning",
        "full lore",
        "entire lore",
        "everything about",
        "complete history",
        "complete lore",      
        "summarize",
        "summarized",         
        "overview of",  
        "chronology",
        "timeline",
        "tell me everything", 
    ]
    return any(m in ql for m in broad_markers)

def chunk_batch(seq: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def tokenize(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_']+", s.lower()))

def dedupe_chunks(chunks: list[dict], initial_scores: dict[int, float], *, max_per_doc: int = 1) -> list[dict]:
    ordered = sorted(chunks, key=lambda r: float(initial_scores.get(int(r["chunk_id"]), 0.0)), reverse=True)
    kept = []
    seen_counts: dict[object, int] = {}

    for row in ordered:
        key = row.get("doc_id")
        if key is None:
            key = (str(row.get("source") or ""), str(row.get("title") or "").strip().lower())
        
        n = seen_counts.get(key, 0)
        if n >= max_per_doc:
            continue

        seen_counts[key] = n+1
        kept.append(row)
    return kept

def detect_intent(question: str) -> str:
    q = question.lower()
    BUILD_MARKERS = [
        "weapon", "artifact", "build", "damage", "dps", "support",
        "team", "comp", "rotation", "stats", "crit", "er", "em",
        "signature", "best", "recommended", "bis", "bis weapon",
        "talent priority", "ascension", "investment", "crit%", "atk", "atk%",
        "def", "def%", "elemental mastery", "dmg", "energy recharge", "sub-dps",
        "normal attack", "attack", "defense", "level"
    ]
    LORE_MARKERS = [
        "lore", "story", "history", "background", "past",
        "origin", "archon", "god", "nation", "region",
        "what happened", "mythology", "legend", "tale",
    ]
    MECHANICS_MARKERS = [
        "skill", "burst", "talent", "constellation", "passive",
        "ability", "how does", "work", "effect", "infusion",
        "reaction", "element", "shield", "heal", "stack", "refinement", "talent",
        "elemental skill", "elemental burst", "stamina", "scaling", "utility passive"
    ]

    LOCATION_MARKERS = [
        "where is", "where are", "where can i find", "location of",
        "how to get to", "how do i get to", "map", "region", "nation",
        "area", "domain", "dungeon", "explore", "exploration",
        "directions", "waypoint", "teleport", "near", "coordinates",
        "liyue", "mondstadt", "inazuma", "sumeru", "fontaine", "natlan", "enkanomiya"
    ]

    BIOGRAPHY_MARKERS = [
        "birthday", "born", "age", "who is", "background of",
        "related to", "family", "sibling", "parent", "friend of",
        "relationship", "affiliated with", "affiliation", "vision holder",
        "from where", "hometown", "nationality", "occupation", "bio",
        "profile", "voiceline", "voiced by", "height", "who are", "comes from",
        "come from", "who was", "where is she from", "where is he from", "where are they from"
    ]

    if any(m in q for m in BIOGRAPHY_MARKERS):
        return "biography"
    if any(m in q for m in LOCATION_MARKERS):
        return "location"
    if any(m in q for m in MECHANICS_MARKERS):
        return "mechanic"
    if any(m in q for m in BUILD_MARKERS):
        return "build"
    if any(m in q for m in LORE_MARKERS):
        return "lore"

    return "general"

def is_recency_sensitive_question(question: str) -> bool:
    q = question.lower()

    RECENCY_MARKERS = [
        "recent", "latest", "current",
        "currently", "now", "new",
        "updated", "update", "patch",
        "version", "meta", "this patch",
        "this version", "new build", "current build",
        "latest build", "best build", "today",
    ]

    return any(m in q for m in RECENCY_MARKERS)

def get_kqm_news_fetch_version_baseline(conn: sqlite3.Connection, max_version_ord: int | None = None) -> tuple[str | None, int | None]:
    cur = conn.cursor()

    if max_version_ord is None:
        max_version_ord = 699

    cur.execute("""
        SELECT version_label, version_ord
        FROM docs
        WHERE source = 'kqm_news'
            AND version_ord IS NOT NULL
            AND version_ord <= ?
            AND COALESCE(status, 1) = 1
        ORDER BY version_ord DESC, fetched_at DESC
        LIMIT 1
    """, (max_version_ord,))

    row = cur.fetchone()
    if not row:
        return None, None

    return row["version_label"], int(row["version_ord"])

def extract_entity_terms(question: str) -> list[str]:
    q = question.strip()

    patterns = [
        r"\bwho\s+is\s+([A-Za-z][A-Za-z' -]{1,40})",
        r"\bwho\s+was\s+([A-Za-z][A-Za-z' -]{1,40})",
        r"\bwhat\s+is\s+([A-Za-z][A-Za-z' -]{1,40})\s+(?:recommended|best|signature|weapon|build|artifact)",
        r"\bwhat\s+are\s+([A-Za-z][A-Za-z' -]{1,40})\s+(?:recommended|best|signature|weapons|builds|artifacts)",
        r"\bbest\s+(?:weapon|artifact|build|team)\s+for\s+([A-Za-z][A-Za-z' -]{1,40})",
        r"\brecommended\s+(?:weapon|artifact|build|team)\s+for\s+([A-Za-z][A-Za-z' -]{1,40})",
        r"\b([A-Z][A-Za-z' -]{1,40})\s+(?:recommended|best|signature)\s+(?:weapon|artifact|build|team)",
    ]

    for pat in patterns:
        m = re.search(pat, q, re.I)
        if not m:
            continue

        ent = m.group(1)
        ent = re.split(r"\?|,|\.|\band\b|\bwhere\b|\bwhat\b", ent, flags=re.I)[0]
        ent = ent.strip().lower()
        ent = re.sub(r"'s\b", "", ent).strip()
        ent = re.sub(r"\b(recommended|best|signature|weapon|weapons|build|artifact|artifacts)$", "", ent).strip()

        if ent:
            return [ent] + ent.split()

    return []

def prefer_entity_seed_chunks(question: str, chunks: list[dict], min_keep: int = 3) -> list[dict]:
    terms = extract_entity_terms(question)
    if not terms:
        return chunks
    
    main_entity = terms[0]
    sub_terms = terms[1:]

    strong = []
    weak = []
    rest = []

    for row in chunks:
        title = (row.get("title") or "").lower()
        text = (row.get("text") or "").lower()
        hay = f"{title}\n{text[:2500]}"

        if main_entity in hay:
            strong.append(row)
        elif sub_terms and any(t in hay for t in sub_terms):
            weak.append(row)
        else:
            rest.append(row)

    if len(strong) >= min_keep:
        return strong + weak + rest

    if strong:
        return strong + weak + rest

    return chunks

def rerank_chunks(question: str, chunks: list[dict], retrieval_signals: dict[int, float | dict], current_version_ord: int | None = None) -> list[dict]:
    q_terms = tokenize(question)
    intent  = detect_intent(question)
    profile = INTENT_PROFILES[intent]
    media_exts = [".jpeg", ".jpg", ".png", ".webp", ".mp4", ".svg", ".ico" , ".webm", ".mp3", ".gif", ".wav", ".ogg", ".woff", ".woff2"]
    ranked = []
    priority = profile.get("source_priority", {})
    required = set(priority.get("required", []))
    excluded = set(priority.get("excluded", []))
    entity_terms = extract_entity_terms(question)

    for row in chunks:
        chunk_id   = int(row["chunk_id"])
        signal = retrieval_signals.get(chunk_id, 0.0)
        if isinstance(signal, dict):
            base_score = float(signal.get("rrf_score", 0.0))
            in_faiss = bool(signal.get("in_faiss", False))
            in_bm25 = bool(signal.get("in_bm25", False))
            faiss_rank = signal.get("faiss_rank")
            bm25_rank = signal.get("bm25_rank")
        else:
            base_score = float(signal)
            in_faiss = False
            in_bm25 = False
            faiss_rank = None
            bm25_rank = None

        text       = row.get("text") or ""
        title      = row.get("title") or ""
        title_l    = title.lower()
        tier       = row.get("tier") or ""
        weight     = float(row.get("weight") or 1.0)
        source     = row.get("source") or ""

        text_terms    = tokenize(text)
        title_terms   = tokenize(title)
        text_overlap  = len(q_terms & text_terms)
        title_overlap = len(q_terms & title_terms)

        weighted_base = base_score * weight

        lexical_bonus = (
            0.02 * text_overlap
          + 0.10 * title_overlap
        )

        tier_bonus = (
            0.05 if tier == "primary"
            else 0.02 if tier == "supplementary"
            else 0.0
        )

        intent_source_bonus = float(
            profile.get("source_bonus",   {}).get(source, 0.0)
          - profile.get("source_penalty", {}).get(source, 0.0)
        )

        title_boost_v = float(profile.get("title_boost_v", 0.0))
        intent_title_boost = (
            title_boost_v
            if any(m in title_l for m in profile.get("title_boost", []))
            else 0.0
        )

        retrieval_bonus = 0.0
        if in_faiss and in_bm25:
            retrieval_bonus += 0.08

        if intent in {"mechanic", "lore", "biography", "location", "general"}:
            if in_bm25:
                retrieval_bonus += 0.05
            if bm25_rank is not None and bm25_rank <= 5:
                retrieval_bonus += 0.07
            elif bm25_rank is not None and bm25_rank <= 15:
                retrieval_bonus += 0.04
        
        if in_faiss:
            retrieval_bonus += 0.03
        if faiss_rank is not None and faiss_rank <= 5:
            retrieval_bonus += 0.05
        elif faiss_rank is not None and faiss_rank <= 15:
            retrieval_bonus += 0.03
        
        penalty = 0.0
        media_count = sum(text.count(ext) for ext in media_exts)
        if media_count > 3:
            penalty += 0.25
        if "character card" in title_l or "genius invokation" in title_l:
            penalty += 0.15

        url_chars = sum(1 for c in text if c in "[]%?=&/:.#_+")
        if len(text.strip()) > 50 and url_chars / len(text) > 0.30:
            penalty += 0.15
        if len(text.strip()) < 100:
            penalty += 0.10
        if any(m in title_l for m in profile.get("title_penalize", [])):
            penalty += 0.15

        require_any = profile.get("text_require_any", [])
        if require_any:
            text_l = text.lower()
            if not any(r in text_l for r in require_any):
                penalty += float(profile.get("text_require_penalty", 0.0))

        source = row.get("source") or ""
        if source in excluded:
            penalty += 0.95
        # elif source not in required and source not in priority.get("preferred", []):
        #     penalty += 0.25

        recency_bonus = 0.0
        recency_penalty = 0.0

        recency_explicit = is_recency_sensitive_question(question)
        recency_active = recency_explicit or intent in {"build", "mechanic"}

        if recency_active and current_version_ord is not None:
            row_version_ord = row.get("version_ord")
            if row_version_ord is not None:
                try:
                    row_version_ord = int(row_version_ord)
                    distance = current_version_ord - row_version_ord

                    if distance <= 0:
                        recency_bonus += 0.15 if recency_explicit else 0.08

                    elif distance <= 1:
                        recency_bonus += 0.08 if recency_explicit else 0.04

                    elif distance <= 3:
                        recency_bonus += 0.03 if recency_explicit else 0.01
                    else:
                        recency_penalty += min(0.25, 0.04 * distance) if recency_explicit else min(0.12, 0.02 * distance)

                except Exception:
                    pass

            else:
                if recency_explicit:
                    recency_penalty += 0.04
        
        entity_bonus = 0.0

        if entity_terms:
            main_entity =  entity_terms[0]
            title_norm = re.sub(r"[^a-z0-9]+", " ", title_l).strip()

            if title_norm == main_entity:
                entity_bonus += 0.4
            elif title_norm.startswith(main_entity + " "):
                entity_bonus += 0.3
            elif main_entity in title_l:
                entity_bonus += 0.15

            if intent == "biography":
                bad_profile_titles = [
                    "avatar", "namecard", "fan art contest", "quest item",
                    "normal attack", "constellation", "utility passive",
                    "taking pictures", "change history"]
                if any(x in title_l for x in bad_profile_titles):
                    entity_bonus -= 0.20

        final_score = (
            weighted_base
            + lexical_bonus
            + entity_bonus
            + tier_bonus
            + intent_source_bonus
            + intent_title_boost
            + retrieval_bonus
            # + recency_bonus
            # - recency_penalty
            - penalty
        )

        ranked.append((final_score, row))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in ranked]