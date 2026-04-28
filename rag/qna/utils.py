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
        "source_penalty": {"kqm_news": 0.15},
        "title_penalize": ["storyline", "voice", "voice-over", "story quest", "archon quest",
                           "world quest", "character card", "genius invokation",
                           "lore", "dialogue", "comments"],
        "text_require_any": ["weapon", "polearm", "sword", "bow", "catalyst",
                             "claymore", "artifact", "set bonus", "recommended",
                             "signature weapon", "bis weapon", "4pc", "2pc", 
                             "main stat", "substat", "dmg bonus", "dmg", "attack", "crit rate",
                             "elemental mastery", "em", "cr", "crit dmg", "cr%", "def%", "hp%",
                             "atk%", "staff"],
        "text_require_penalty": 0.30,
        "source_priority":{
            "required": ["kqm_tcl", "genshin_gg", "honey", "genshin_wiki"],
            "preferred": ["game8"],
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
            "preferred": [],
            "excluded": ["kqm_tcl", "game8", "genshin_gg"]
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

def rrf_fuse(*result_lists, k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}

    for results in result_lists:
        for rank, (chunk_id, _score) in enumerate(results, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused

def filter_by_intent_source(conn: sqlite3.Connection, chunk_ids: list[int], intent: str, min_required: int=5, max_fallback: int=30) -> list[int]:
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

def dedup_chunks(chunks: list[dict], initial_scores: dict[int, float], *, max_per_doc: int = 1) -> list[dict]:
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
        "lore", "story", "history", "background", "who is", "who was",
        "past", "origin", "archon", "god", "nation", "region",
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
        "profile", "voiceline", "voiced by", "height",
]

    build_hits = sum(1 for m in BUILD_MARKERS if m in q)
    lore_hits = sum(1 for m in LORE_MARKERS if m in q)
    mechanics_hits = sum(1 for m in MECHANICS_MARKERS if m in q)
    location_hits = sum(1 for m in LOCATION_MARKERS if m in q)
    biography_hits = sum(1 for m in BIOGRAPHY_MARKERS if m in q)

    scores = {
        "build":     build_hits,
        "lore":      lore_hits,
        "mechanic":  mechanics_hits,
        "location":  location_hits,
        "biography": biography_hits,
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "general"
    return best

def rerank_chunks(question: str, chunks: list[dict], initial_scores: dict[int, float]) -> list[dict]:
    q_terms = tokenize(question)
    intent  = detect_intent(question)
    profile = INTENT_PROFILES[intent]
    media_exts = [".jpeg", ".jpg", ".png", ".webp", ".mp4", ".svg", ".ico" , ".webm", ".mp3", ".gif", ".wav", ".ogg", ".woff", ".woff2"]
    ranked = []
    priority = profile.get("source_priority", {})
    required = set(priority.get("required", []))
    excluded = set(priority.get("excluded", []))

    for row in chunks:
        chunk_id   = int(row["chunk_id"])
        base_score = float(initial_scores.get(chunk_id, 0.0))
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

        final_score = (
            weighted_base
            + lexical_bonus
            + tier_bonus
            + intent_source_bonus
            + intent_title_boost
            - penalty
        )

        ranked.append((final_score, row))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in ranked]