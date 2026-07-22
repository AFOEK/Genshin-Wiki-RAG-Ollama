from __future__ import annotations

import logging
import sqlite3
import re
import json
import hashlib
import yaml

import numpy as np

from typing import Iterable, Any
from pathlib import Path
from difflib import SequenceMatcher

from .types import RetrievalResult

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
        "bm25_weights": {
            "chunk_id": 0.0,
            "doc_id": 0.0,
            "source": 0.0,
            "title": 8.0,
            "text": 1.0,
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
        "bm25_weights": {
            "chunk_id": 0.0,
            "doc_id": 0.0,
            "source": 0.0,
            "title": 3.0,
            "text": 2.0,
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
        "bm25_weights": {
            "chunk_id": 0.0,
            "doc_id": 0.0,
            "source": 0.0,
            "title": 4.0,
            "text": 2.0,
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
            "the sea of flowers at the end", "frost moon"
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
        "bm25_weights": {
            "chunk_id": 0.0,
            "doc_id": 0.0,
            "source": 0.0,
            "title": 5.0,
            "text": 1.5,
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
        "bm25_weights": {
            "chunk_id": 0.0,
            "doc_id": 0.0,
            "source": 0.0,
            "title": 6.0,
            "text": 1.5,
        },
    },
        "lookup": {
        "source_bonus": {
            "genshin_wiki": 0.10,
            "honey": 0.05,
            "game8": 0.03,
            "genshin_gg": 0.03,
        },
        "source_penalty": {
            "kqm_news": 0.10,
        },
        "title_penalize": [
            "change history",
            "gallery",
        ],
        "title_boost": [],
        "title_boost_v": 0.0,
        "text_require_any": [],
        "text_require_penalty": 0.0,
        "source_priority": {
            "required": [],
            "preferred": [
                "genshin_wiki",
                "honey",
                "game8",
                "genshin_gg",
            ],
            "excluded": [
                "kqm_news",
            ],
        },
        "bm25_weights": {
            "chunk_id": 0.0,
            "doc_id": 0.0,
            "source": 0.0,
            "title": 8.0,
            "text": 1.0,
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
        "bm25_weights": {
            "chunk_id": 0.0,
            "doc_id": 0.0,
            "source": 0.0,
            "title": 4.0,
            "text": 1.0,
        },
    },
    "version": {
        "source_bonus": {
            "kqm_news": 0.25,
            "game8": 0.18,
            "genshin_wiki": 0.12,
            "genshin_gg": 0.05,
        },
        "source_penalty": {},
        "title_penalize": [
            "change history",
            "gallery",
            "comments",
        ],
        "title_boost": [
            "version",
            "livestream",
            "release date",
            "latest news",
            "game info",
            "update",
        ],
        "title_boost_v": 0.30,
        "text_require_any": [
            "version",
            "luna",
            "patch",
            "update",
        ],
        "text_require_penalty": 0.08,
        "source_priority": {
            "required": [],
            "preferred": [
                "kqm_news",
                "game8",
                "genshin_wiki",
            ],
            "excluded": [],
        },
        "bm25_weights": {
            "chunk_id": 0.0,
            "doc_id": 0.0,
            "source": 0.0,
            "title": 6.0,
            "text": 2.0,
        },
    },
}

BUILD_RECOMMENDATION_MARKERS = (
    "recommended",
    "recommend",
    "bis",
    "signature weapon",
    "talent priority",
    "talent order",
    "stat priority",
    "main stats",
    "substats",
    "team comp",
    "team composition",
    "rotation",
    "build for",
)

CURRENT_VERSION_PATTERN = re.compile(
    r"(?:"
    r"\b(?:latest|current(?:ly)?|newest|most\s+recent)\b"
    r".{0,60}"
    r"\b(?:version|patch|update)\b"
    r"|"
    r"\b(?:version|patch|update)\b"
    r".{0,60}"
    r"\b(?:latest|current(?:ly)?|newest|most\s+recent)\b"
    r"|"
    r"\b(?:what|which)\s+(?:game\s+)?version\s+(?:is|are)\b"
    r")",
    re.IGNORECASE,
)

LOOKUP_FACET_ALIASES = {
    "skin": {
        "skin",
        "skins",
        "outfit",
        "outfits",
        "costume",
        "costumes",
        "appearance",
        "attire",
    },
}

BUILD_SUBTYPE_PROFILES = {
    "weapon": {
        "query_markers": (
            "weapon",
            "weapons",
            "best weapon",
            "recommended weapon",
            "signature weapon",
            "bis weapon",
        ),
        "section_markers": (
            "best weapons",
            "recommended weapons",
            "recommended weapon",
            "weapon ranking",
            "signature weapon",
            "bis weapon",
        ),
        "fts_terms": (
            "best weapons",
            "recommended weapon",
            "weapon",
            "weapons",
        ),
        "bonus": 0.40,
        "missing_penalty": 0.35,
    },
    "artifact": {
        "query_markers": (
            "artifact",
            "artifacts",
            "artifact set",
            "best artifact",
            "recommended artifact",
            "main stat",
            "main stats",
            "substat",
            "substats",
            "sands",
            "goblet",
            "circlet",
        ),
        "section_markers": (
            "best artifacts",
            "recommended artifacts",
            "recommended artifact",
            "artifact sets",
            "best artifact set",
            "main stats",
            "artifact main stats",
        ),
        "fts_terms": (
            "best artifacts",
            "recommended artifact",
            "artifact",
            "artifacts",
            "main stats",
        ),
        "bonus": 0.40,
        "missing_penalty": 0.35,
    },

    "talent": {
        "query_markers": (
            "talent priority",
            "talent order",
            "which talent",
            "level first",
            "leveling priority",
            "crown first",
            "skill priority",
            "burst priority",
            "normal attack priority",
        ),
        "section_markers": (
            "talent priority",
            "talent order",
            "talent leveling priority",
            "talent leveling",
            "talent levels",
            "talents",
            "leveling priority",
            "skill priority",
            "burst priority",
        ),
        "fts_terms": (
            "talent priority",
            "talent order",
            "talents",
            "leveling priority",
        ),
        "bonus": 0.40,
        "missing_penalty": 0.35,
    },
    "team": {
        "query_markers": (
            "team",
            "teams",
            "team comp",
            "team composition",
            "best team",
            "recommended team",
            "party",
        ),
        "section_markers": (
            "best teams",
            "recommended teams",
            "team composition",
            "team comps",
            "best team",
        ),
        "fts_terms": (
            "best teams",
            "recommended team",
            "team composition",
            "teams",
        ),
        "bonus": 0.40,
        "missing_penalty": 0.35,
    },
    "stats": {
        "query_markers": (
            "stats",
            "stat priority",
            "main stat",
            "substat",
            "crit ratio",
            "energy recharge",
            "elemental mastery",
        ),
        "section_markers": (
            "stat priority",
            "recommended stats",
            "main stats",
            "substats",
            "crit ratio",
        ),
        "fts_terms": (
            "stat priority",
            "recommended stats",
            "main stats",
            "substats",
        ),
        "bonus": 0.35,
        "missing_penalty": 0.30,
    },
    "rotation": {
        "query_markers": (
            "rotation",
            "rotations",
            "combo",
            "skill order",
        ),
        "section_markers": (
            "rotation",
            "recommended rotation",
            "team rotation",
            "combo",
        ),
        "fts_terms": (
            "rotation",
            "recommended rotation",
            "team rotation",
        ),
        "bonus": 0.35,
        "missing_penalty": 0.30,
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

def is_current_version_question(question: str) -> bool:
    return bool(CURRENT_VERSION_PATTERN.search(str(question or "").strip()))

def detect_lookup_facets(question: str) -> set[str]:
    normalized = normalize_title_key(question)
    tokens = set(normalized.split())
    detected: set[str] = set()

    for facet, aliases in LOOKUP_FACET_ALIASES.items():
        if any(alias in tokens for alias in aliases):
            detected.add(facet)

    return detected


def extract_lookup_target(question: str,) -> tuple[str | None, set[str]]:
    q = str(question or "").strip().replace("’", "'")
    facets = detect_lookup_facets(q)

    if is_current_version_question(q):
        return None, facets

    # "What is Citlali's new skin?"
    # "What is Citlali new skin?"
    character_facet_patterns = (
        r"^\s*(?:what|which)\s+(?:is|are)\s+"
        r"(?:the\s+)?"
        r"(?P<entity>.+?)(?:'s)?\s+"
        r"(?:(?:new|latest|current|upcoming)\s+)?"
        r"(?:skin|outfit|costume|appearance|attire)"
        r"(?:\s+called)?\s*[?!.]*$",

        # "What is the new skin for Citlali?"
        r"^\s*(?:what|which)\s+(?:is|are)\s+"
        r"(?:the\s+)?"
        r"(?:(?:new|latest|current|upcoming)\s+)?"
        r"(?:skin|outfit|costume|appearance|attire)\s+"
        r"(?:for|of)\s+"
        r"(?P<entity>.+?)"
        r"(?:\s+called)?\s*[?!.]*$",
    )

    for pattern in character_facet_patterns:
        match = re.match(
            pattern,
            q,
            re.IGNORECASE,
        )

        if not match:
            continue

        entity = re.sub(
            r"\s+",
            " ",
            match.group("entity"),
        ).strip(" '")

        entity = re.sub(
            r"^the\s+",
            "",
            entity,
            flags=re.IGNORECASE,
        )

        return entity or None, facets

    generic_patterns = (
        r"^\s*what\s+(?:is|are)\s+(.+?)\s*[?!.]*$",
        r"^\s*where\s+(?:is|are)\s+(.+?)\s*[?!.]*$",
        r"^\s*who\s+(?:is|are|was|were)\s+(.+?)\s*[?!.]*$",
        r"^\s*what\s+(?:is|are|was|were)\s+(.+?)\s*[?!.]*$",
        r"^\s*where\s+(?:is|are|was|were)\s+(.+?)\s*[?!.]*$",
    )

    entity: str | None = None

    for pattern in generic_patterns:
        match = re.match(pattern, q, re.IGNORECASE)

        if match:
            entity = match.group(1)
            break

    if not entity:
        return None, facets

    entity = re.sub(r"\s+", " ", entity,).strip(" '")
    entity = re.sub( r"^the\s+", "", entity, flags=re.IGNORECASE)

    if "skin" in facets:
        entity = re.sub(r"\b(?:new|latest|current|upcoming)\b", " ", entity, flags=re.IGNORECASE)
        entity = re.sub(
            r"\b(?:skin|skins|outfit|outfits|costume|costumes|"
            r"appearance|attire)\b", " ", entity, flags=re.IGNORECASE)

        entity = re.sub(r"\s+", " ", entity,).strip(" '")

    return entity or None, facets

def extract_build_entity(question: str) -> str | None:
    q = re.sub(r"\s+", " ", str(question or "").strip())
    patterns = (
        r"^\s*what\s+(?:is|are)\s+"
        r"(?P<entity>.+?)"
        r"(?:'s)?\s+best\s+build"
        r"(?:s)?\s*[?!.]*$",

        r"^\s*(?:what|which)\s+"
        r"(?:is|are)\s+the\s+best\s+build"
        r"(?:s)?\s+(?:for|of)\s+"
        r"(?P<entity>.+?)\s*[?!.]*$",

        r"^\s*best\s+build"
        r"(?:s)?\s+(?:for|of)\s+"
        r"(?P<entity>.+?)\s*[?!.]*$",
    )

    for pattern in patterns:
        match = re.match(pattern, q, re.IGNORECASE,)

        if match:
            return re.sub(r"\s+", " ", match.group("entity")).strip(" '")

    return None

def normalize_model_name(x) -> str:
    if x is None:
        return ""

    if isinstance(x, (list, tuple, set)):
        x = next(iter(x), "")
    elif isinstance(x, dict):
        x = x.get("name") or x.get("model") or x.get("embedding_model") or ""

    s = str(x).strip().lower()
    s = s.replace("\\", "/")

    if "/" in s and not s.startswith(("http://", "https://")):
        parts = s.split("/")
        if len(parts) > 2 or s.startswith("/"):
            s = parts[-1]

    if s.endswith(":latest"):
        s = s[:-7]

    return s

def entity_title_similarity(entity: str, title: str) -> float:
    entity_tokens = normalize_title_key(entity).split()
    title_tokens = normalize_title_key(primary_page_title(title)).split()

    if not entity_tokens or not title_tokens:
        return 0.0

    window_size = len(entity_tokens)

    if len(title_tokens) < window_size:
        candidate_windows = [title_tokens]
    else:
        candidate_windows = [
            title_tokens[index:index + window_size]
            for index in range(len(title_tokens) - window_size + 1)]

    entity_key = " ".join(entity_tokens)
    best = 0.0

    for window in candidate_windows:
        window_key = " ".join(window)
        similarity = SequenceMatcher(None, entity_key, window_key,).ratio()
        best = max(best, similarity,)

    return best

def resolve_lookup_entity_from_chunks(entity: str, chunks: list[dict], *, minimum_similarity: float = 0.84) -> tuple[str, float]:
    entity = re.sub(r"\s+", " ", entity).strip()

    if not entity:
        return entity, 0.0

    entity_token_count = len(normalize_title_key(entity).split())

    best_entity = entity
    best_similarity = 0.0

    for row in chunks:
        title = primary_page_title(str(row.get("title") or ""))
        original_tokens = re.findall(r"[A-Za-z0-9'-]+", title,)
        normalized_tokens = [normalize_title_key(token) for token in original_tokens]
        normalized_tokens = [token for token in normalized_tokens if token]

        if not normalized_tokens:
            continue

        if len(normalized_tokens) < entity_token_count:
            windows = [(original_tokens, normalized_tokens,)]
        else:
            windows = []
            for index in range(len(normalized_tokens) - entity_token_count + 1):
                windows.append((original_tokens[index:index + entity_token_count], normalized_tokens[index:index + entity_token_count]))

        entity_key = normalize_title_key(entity)

        for original_window, normalized_window in windows:
            candidate_key = " ".join(normalized_window)
            similarity = SequenceMatcher(None, entity_key, candidate_key,).ratio()
            if similarity > best_similarity:
                best_similarity = similarity
                best_entity = " ".join(original_window)

    if best_similarity < minimum_similarity:
        return entity, best_similarity

    return best_entity, best_similarity

def expected_model_from_cfg(cfg: dict, backend: str | None = None, source: str = "runtime") -> str:
    source = str(source or "runtime").strip().lower()

    runtime = cfg.get("runtime", {}) or {}
    provider = (backend or runtime.get("embedding_provider", "ollama")).strip().lower()
    if provider == "llama.cpp":
        provider = "llamacpp"

    if source == "kaggle":
        return str(cfg.get("kaggle", {}).get("embedding_model", ""))

    if source == "ollama":
        return str(cfg.get("ollama", {}).get("embedding_model", ""))

    if source in ("llamacpp", "llama.cpp"):
        return str(cfg.get("llamacpp", {}).get("embedding_model", ""))

    if provider == "llamacpp":
        return str(cfg.get("llamacpp", {}).get("embedding_model", ""))

    return str(cfg.get("ollama", {}).get("embedding_model", ""))

def check_faiss_model_match(*, actual_model, expected_model, policy: str = "error") -> None:
    actual_n = normalize_model_name(actual_model)
    expected_n = normalize_model_name(expected_model)

    if not actual_n or not expected_n:
        log.warning(
            "[FAISS] cannot fully validate embedding model: actual=%r expected=%r",
            actual_model,
            expected_model,
        )
        return

    if actual_n == expected_n:
        return

    msg = (
        "[FAISS] embedding model mismatch: "
        f"meta.json={actual_model!r} config_expected={expected_model!r}. "
        "Rebuild/query with the same embedding model, or switch retrieval.faiss_expected_model."
    )

    policy = str(policy or "error").strip().lower()

    if policy == "error":
        raise RuntimeError(msg)

    if policy == "warn":
        log.warning(msg)
        return

    if policy == "ignore":
        log.warning("[FAISS] ignoring model mismatch by config: %s", msg)
        return

    raise RuntimeError(f"Unknown retrieval.faiss_model_mismatch policy: {policy}")

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

def build_hybrid_hyde_signal(faiss_results: list[tuple[int, float]], bm25_results: list[tuple[int, float]], hyde_results: list[tuple[int, float]], *, rrf_k: int = 60, rrf_scale: float = 10.0, hyde_weight: float = 0.75) -> dict[int, dict]:
    signals: dict[int, dict] = {}

    def ensure(chunk_id: int) -> dict:
        if chunk_id not in signals:
            signals[chunk_id] = {
                "rrf_score": 0.0,
                "faiss_score": 0.0,
                "bm25_score": 0.0,
                "hyde_score": 0.0,
                "faiss_rank": None,
                "bm25_rank": None,
                "hyde_rank": None,
                "in_faiss": False,
                "in_bm25": False,
                "in_hyde": False,
            }
        return signals[chunk_id]

    for rank, (chunk_id, score) in enumerate(faiss_results, start=1):
        chunk_id = int(chunk_id)
        signal = ensure(chunk_id)

        signal["faiss_score"] = float(score)
        signal["faiss_rank"] = rank
        signal["in_faiss"] = True
        signal["rrf_score"] += (1.0 / (rrf_k + rank))

    for rank, (chunk_id, score) in enumerate(bm25_results, start=1,):
        chunk_id = int(chunk_id)
        signal = ensure(chunk_id)

        signal["bm25_score"] = float(score)
        signal["bm25_rank"] = rank
        signal["in_bm25"] = True
        signal["rrf_score"] += (1.0 / (rrf_k + rank))

    for rank, (chunk_id, score) in enumerate(hyde_results, start=1):
        chunk_id = int(chunk_id)
        signal = ensure(chunk_id)

        signal["hyde_score"] = float(score)
        signal["hyde_rank"] = rank
        signal["in_hyde"] = True
        signal["rrf_score"] += (hyde_weight / (rrf_k + rank))

    for signal in signals.values():
        signal["rrf_score"] *= rrf_scale

    return signals

def get_bm25_weights(intent: str) -> tuple[float, float, float, float, float]:
    profile = INTENT_PROFILES.get(
        intent,
        INTENT_PROFILES["general"],
    )

    cfg = profile.get("bm25_weights", {}) or {}

    weights = (
        float(cfg.get("chunk_id", 0.0)),
        float(cfg.get("doc_id", 0.0)),
        float(cfg.get("source", 0.0)),
        float(cfg.get("title", 4.0)),
        float(cfg.get("text", 1.0)),
    )

    if any(weight < 0 for weight in weights):
        raise ValueError(f"BM25 weights must be non-negative, got {weights}")

    return weights

def quote_fts5_phrase(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip().lower())
    value = value.replace('"', '""')
    return f'"{value}"'

def make_fts5_query(user_query: str) -> str:
    raw_tokens = re.findall(r"[A-Za-z0-9_']+", user_query.lower())

    tokens = []
    for t in raw_tokens:
        t = re.sub(r"'s$", "", t)
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

def make_intent_fts5_query(question: str, intent: str) -> str | None:
    lookup_entity, lookup_facets = (extract_lookup_target(question))
    if intent == "lookup":
        if not lookup_entity:
            return None

        entity = quote_fts5_phrase(lookup_entity)

        if "skin" in lookup_facets:
            skin_clause = " OR ".join(
                quote_fts5_phrase(term)
                for term in (
                    "skin",
                    "outfit",
                    "costume",
                    "character outfit",
                )
            )

            return (
                f"(title:{entity} OR text:{entity}) "
                f"AND ({skin_clause})")

        return f"title:{entity}"
    
    if intent == "build":
        build_entity = (
            extract_build_entity(question)
            or (
                extract_entity_terms(question)[0]
                if extract_entity_terms(question)
                else None
            )
        )

        if not build_entity:
            return None

        entity = quote_fts5_phrase(build_entity)

        build_terms = (
            '"build" OR '
            '"weapon" OR '
            '"artifact" OR '
            '"main stats" OR '
            '"substats" OR '
            '"team"'
        )

        return (
            f"(title:{entity} OR text:{entity}) "
            f"AND ({build_terms})")

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
    
def as_bool(x, default: bool = False) -> bool:
    if x is None:
        return default

    if isinstance(x, bool):
        return x

    value = str(x).strip().lower()

    if value in {"1", "true", "yes", "y", "on"}:
        return True

    if value in {"0", "false", "no", "n", "off"}:
        return False

    return default

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

def marker_in_text(question_l: str, marker: str) -> bool:
    marker_l = marker.strip().lower()

    if not marker_l:
        return False
    if " " in marker_l:
        return marker_l in question_l

    return re.search(
        rf"(?<!\w){re.escape(marker_l)}(?!\w)",
        question_l,
    ) is not None


def contains_any_marker(question_l: str, markers) -> bool:
    return any(marker_in_text(question_l, marker) for marker in markers)

def is_build_recommendation_question(question: str) -> bool:
    q = question.lower().replace("’", "'")

    if contains_any_marker(q, BUILD_RECOMMENDATION_MARKERS):
        return True

    if re.search(
        r"\bshould\s+[a-z][a-z' -]{1,40}?\s+"
        r"(?:use|equip|run|build)\b",
        q,
    ):
        return True

    if re.search(
        r"\b(?:weapons?|artifacts?|teams?|stats)\s+for\s+",
        q,
    ):
        return True

    if re.search(
        r"\b[a-z][a-z' -]{1,40}?'s\s+"
        r"(?:best\s+|recommended\s+|signature\s+)?"
        r"(?:weapons?|artifacts?|artifact set|teams?|stats)\b",
        q,
    ):
        return True

    return False

def extract_lookup_entity(question: str) -> str | None:
    entity, _facets = extract_lookup_target(question)
    return entity

def normalize_title_key(value: str) -> str:
    value = value.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()

def primary_page_title(title: str) -> str:
    return re.split(r"\s*[|｜]\s*", title, maxsplit=1)[0].strip()

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

def build_weighted_rrf_signal(channels: dict[str, list[tuple[int, float]]], *, weights: dict[str, float] | None = None, rrf_k: int = 60, rrf_scale: float = 10.0) -> dict[int, dict]:
    weights = weights or {}
    signals: dict[int, dict] = {}

    for channel, results in channels.items():
        weight = float(weights.get(channel, 1.0))

        for rank, (chunk_id, raw_score) in enumerate(results, start=1):
            chunk_id = int(chunk_id)
            signal = signals.setdefault(chunk_id, {
                "rrf_score": 0.0,
            },)
            signal[f"{channel}_score"] = float(raw_score)
            signal[f"{channel}_rank"] = rank
            signal[f"in_{channel}"] = True
            signal["rrf_score"] += (rrf_scale * weight / (rrf_k + rank))

    return signals

def detect_intent(question: str) -> str:
    q = question.lower()
    build_subtypes = detect_build_subtypes(question)
    BUILD_MARKERS = [
        "weapon", "artifact", "build", "damage", "dps", "team", "team comp", "rotation", 
        "stats", "crit", "signature weapon",
        "recommended weapon", "bis weapon", "talent priority", "stat priority", "main stats",
        "substats", "elemental mastery", "energy recharge", "sub-dps",
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
    if is_current_version_question(question):
        return "version"

    if contains_any_marker(q, BIOGRAPHY_MARKERS):
        return "biography"

    if contains_any_marker(q, LOCATION_MARKERS):
        return "location"

    build_entity = extract_build_entity(question)
    if (build_entity or build_subtypes or is_build_recommendation_question(question) or contains_any_marker(q, BUILD_MARKERS)):
        return "build"

    if contains_any_marker(q, MECHANICS_MARKERS):
        return "mechanic"

    if extract_lookup_entity(question):
        return "lookup"

    if contains_any_marker(q, LORE_MARKERS):
        return "lore"

    return "general"

def detect_build_subtypes(question: str) -> set[str]:
    q = question.lower()
    found: set[str] = set()

    for subtype, subtype_cfg in BUILD_SUBTYPE_PROFILES.items():
        markers = subtype_cfg.get("query_markers", ())
        if contains_any_marker(q, markers):
            found.add(subtype)

    return found

def build_grounded_answer_prompt(question: str, context: str, *, intent: str | None =  None, build_subtypes: set[str] | None = None, max_recommendations: int = 5) -> str:
    subtypes = set(build_subtypes or ())
    format_rules = ""
    is_comparison = bool(re.search(r"\b(compare|comparison|versus|vs\.?|difference)\b", question, re.IGNORECASE))

    if intent == "build" and "weapon" in subtypes and is_comparison:
        format_rules = """
    This is a weapon comparison question.

    Answer format:
    1. Compare only the weapons explicitly requested.
    2. Use a compact table with: weapon, Energy Recharge, energy generation, team buffs, and best use case.
    3. State which weapon is preferable for each requested criterion.
    4. Do not introduce additional weapons unless needed for brief context.
    5. Do not infer stats, passives, rankings, or trade-offs not explicitly supported by the context.
    6. If evidence for one criterion is missing, state that the retrieved context does not provide it.
    """
        
    elif intent == "build" and "weapon" in subtypes:
        format_rules = f"""
This is a weapon recommendation question.

Answer format:
1. Give the top recommendation first.
2. Then list up to {max_recommendations} explicitly supported weapon options in ranked order.
3. For each weapon, explain why it is recommended using only evidence from the context.
4. Mention the relevant role, stat, passive, utility, or trade-off only when the context supports it.
5. If the context provides a ranking but no reason, state that it is ranked by the source and do not invent a reason.
6. Even if the question says "best weapon" in the singular, include supported alternatives after the top choice.
7. Do not infer weapon stats, passives, damage, Energy Recharge, Elemental Mastery, or role from prior knowledge.
8. Every explanation must be explicitly supported by the supplied context.
9. If the context contains only a ranked list, say: "Ranked #N by the source; the retrieved context does not provide a reason."
"""
    elif intent == "biography":
        format_rules = """
This is an identity or biography question.

Answer rules:
1. Begin with a direct 2–4 sentence identification of the person or character.
2. Prefer explicit identity, affiliation, role, title, origin, and status statements.
3. Every factual claim must be directly stated in at least one supplied chunk.
4. Do not create a relationship between two entities merely because they occur in the same chunk or discuss similar themes.
5. Do not claim that two entities share a title, identity, role, affiliation, origin, or relationship unless the context explicitly says that they do.
6. Do not convert "introduced in Version X" into "released as playable in Version X."
7. Do not combine separate facts into a stronger claim. For example, evidence that a character is playable plus separate evidence that they were introduced in Version X does not prove that they became playable in Version X.
8. Treat trivia, etymology, notes, and inferred thematic similarities as secondary information.
9. Omit uncertain facts rather than qualifying or guessing them.
10. Do not produce a numbered list unless the question requests one.
"""

    elif intent == "build" and "artifact" in subtypes:
        format_rules = f"""
This is an artifact recommendation question.

Answer format:
1. Give the top artifact set first.
2. Then list up to {max_recommendations} explicitly supported artifact options in ranked order.
3. For each option, explain its use case, set effect, role, or trade-off only when supported by the context.
4. Distinguish full sets from mixed 2-piece combinations when the context does so.
5. If the context provides only a ranking and no explanation, state that clearly instead of inventing a reason.
6. Even if the question says "best artifact set" in the singular, include supported alternatives after the top choice.
7. An artifact may be recommended only when the context explicitly associates that artifact with the requested character.
8. Do not recommend artifacts merely because they appear in a generic artifact mechanics page.
9. Preserve the ranking from the character's build section.
"""

    elif intent == "build" and "team" in subtypes:
        format_rules = f"""
This is a team recommendation question.

List up to {max_recommendations} supported team compositions in ranked order.
For each team, identify the members and briefly explain their roles and synergy using only the context.

Answer format:
1. List a team only when the context explicitly presents those characters together as one team, party, lineup, or team composition.
2. Do not construct a team by combining character names found in separate passages.
3. Do not infer synergy solely from isolated descriptions of individual characters.
4. If no explicit team composition for the requested character appears in the context, state that the retrieved context does not contain a supported team composition.
"""

    elif intent == "build" and "talent" in subtypes:
        format_rules = """
This is a talent-priority question.

Give the priority as an ordered sequence such as:
1. Elemental Skill
2. Elemental Burst
3. Normal Attack

Explain each priority only when the context provides enough evidence.
"""

    return f"""
You are a retrieval-grounded Genshin Impact assistant.

Answer the question using only the supplied context.

General rules:
- Do not invent unsupported facts.
- Treat headings, numbered rankings, bullet lists, tables, and item descriptions as explicit evidence.
- Preserve the ranking order shown in the context.
- Before refusing, inspect all headings, lists, tables, and descriptions.
- Cite supporting chunk IDs where practical.
- If the context supports fewer than {max_recommendations} recommendations, list only those supported.
- If the context contains no answer, say that there is not enough evidence.
- Each factual sentence must be supported by explicit wording in the context.
- Do not infer relationships, equivalence, causation, chronology, or shared titles from co-occurrence.
- Do not merge statements from separate chunks into a new claim unless one chunk explicitly states the resulting relationship.
- Preserve distinctions such as "introduced," "announced," "released," and "became playable."

{format_rules}

Question:
{question}

Context:
{context}
""".strip()

def build_three_way_signal(faiss_results: list[tuple[int, float]], turbovec_results: list[tuple[int, float]], bm25_results: list[tuple[int, float]], *, rrf_k: int = 60, rrf_scale: float = 10.0) -> dict[int, dict]:
    signals: dict[int, dict] = {}

    def ensure(cid: int) -> dict:
        if cid not in signals:
            signals[cid] = {
                "rrf_score": 0.0,
                "faiss_score": 0.0,
                "turbovec_score": 0.0,
                "bm25_score": 0.0,
                "faiss_rank": None,
                "turbovec_rank": None,
                "bm25_rank": None,
                "in_faiss": False,
                "in_turbovec": False,
                "in_bm25": False,
            }
        return signals[cid]

    for rank, (cid, score) in enumerate(faiss_results, start=1):
        cid = int(cid)
        s = ensure(cid)
        s["faiss_score"] = float(score)
        s["faiss_rank"] = rank
        s["in_faiss"] = True
        s["rrf_score"] += 1.0 / (rrf_k + rank)

    for rank, (cid, score) in enumerate(turbovec_results, start=1):
        cid = int(cid)
        s = ensure(cid)
        s["turbovec_score"] = float(score)
        s["turbovec_rank"] = rank
        s["in_turbovec"] = True
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

def merge_context_preserving_seeds(seed_chunks: list[dict], extra_chunks: list[dict], *, max_total: int, max_per_doc: int = 4) -> list[dict]:
    max_total = max(int(max_total), len(seed_chunks))
    output: list[dict] = []
    seen_chunk_ids: set[int] = set()
    doc_counts: dict[int, int] = {}

    for row in seed_chunks:
        chunk_id = int(row["chunk_id"])

        if chunk_id in seen_chunk_ids:
            continue

        output.append(row)
        seen_chunk_ids.add(chunk_id)

        doc_id = int(row["doc_id"])
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

    for row in extra_chunks:
        if len(output) >= max_total:
            break

        chunk_id = int(row["chunk_id"])

        if chunk_id in seen_chunk_ids:
            continue

        doc_id = int(row["doc_id"])

        if doc_counts.get(doc_id, 0) >= max_per_doc:
            continue

        output.append(row)
        seen_chunk_ids.add(chunk_id)
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

    return output

def trim_chunks_to_context_budget(chunks: list[dict], *, max_chunks: int, max_chars: int, max_chars_per_chunk: int) -> list[dict]:
    output: list[dict] = []
    total_chars = 0

    for row in chunks:
        if len(output) >= max_chunks:
            break

        text = str(row.get("text") or "").strip()
        if not text:
            continue

        remaining = max_chars - total_chars
        if remaining < 200:
            break

        allowed = min(len(text), max_chars_per_chunk, remaining)

        copied = dict(row)
        copied["text"] = text[:allowed]

        output.append(copied)
        total_chars += allowed

    return output


def normalized_phrase(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

def has_lookup_phrase(chunks: list[dict], entity: str) -> bool:
    entity_key = normalized_phrase(entity)

    if not entity_key:
        return False

    for row in chunks:
        title_key = normalized_phrase(str(row.get("title") or ""))
        text_key = normalized_phrase(str(row.get("text") or "")[:2500])

        if entity_key in title_key:
            return True

        if entity_key in text_key:
            return True

    return False

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

def normalize_cache_question(question: str) -> str:
    return " ".join((question or "").lower().strip().split())

def stable_json_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def make_retrieval_cache_key(*, question: str, retriever_name: str, backend: str | None, direct_top_k: int, intent: str, subtypes: set[str], db_path: Path, index_meta: dict) -> str:
    db_mtime = int(Path(db_path).stat().st_mtime) if Path(db_path).exists() else 0
    payload = {
        "v": 1,
        "question": normalize_cache_question(question),
        "retriever": retriever_name,
        "backend": backend or "",
        "direct_top_k": int(direct_top_k),
        "intent": intent,
        "subtypes": sorted(subtypes),
        "db_mtime": db_mtime,
        "index_meta": index_meta
    }
    return stable_json_hash(payload)

def retrieval_result_to_cache(result: RetrievalResult) -> dict:
    return {
        "question": result.question,
        "intent": result.intent,
        "build_subtypes": sorted(result.build_subtypes),
        "broad": bool(result.broad),
        "candidate_chunks": result.candidate_chunks,
        "selected_chunks": result.selected_chunks,
        "context": result.context,
        "retrieval_signals": result.retrieval_signals,
        "baseline_label": result.baseline_label,
        "baseline_ord": result.baseline_ord,
        "strict_fts_query": result.strict_fts_query,
        "diagnostics": result.diagnostics,
    }

def retrieval_result_from_cache(payload: dict) -> RetrievalResult:
    return RetrievalResult(
        question=str(payload.get("question", "")),
        intent=str(payload.get("intent", "general")),
        build_subtypes=set(payload.get("build_subtypes", [])),
        broad=bool(payload.get("broad", False)),
        candidate_chunks=list(payload.get("candidate_chunks", [])),
        selected_chunks=list(payload.get("selected_chunks", [])),
        context=str(payload.get("context", "")),
        retrieval_signals=dict(payload.get("retrieval_signals", {})),
        baseline_label=payload.get("baseline_label"),
        baseline_ord=payload.get("baseline_ord"),
        strict_fts_query=payload.get("strict_fts_query"),
        diagnostics=dict(payload.get("diagnostics", {})),
    )

def extract_entity_terms(question: str) -> list[str]:
    q = question.strip().replace("’", "'")

    build_entity_target = (
        r"(?:"
        r"talent priority|talent order|stat priority|"
        r"team compositions?|team comps?|"
        r"artifact sets?|artifacts?|"
        r"weapons?|build|teams?|"
        r"talents?|rotations?|stats"
        r")"
    )

    patterns = [
        rf"\b(?:what\s+(?:is|are)\s+)?"
        rf"(?:the\s+)?"
        rf"(?:best|recommended|signature)\s+"
        rf"{build_entity_target}\s+for\s+"
        rf"([A-Za-z][A-Za-z' -]{{1,40}}?)"
        rf"(?:\?|$)",

        rf"\b(?:what\s+(?:is|are)\s+)?"
        rf"([A-Za-z][A-Za-z' -]{{1,40}}?)'s\s+"
        rf"(?:recommended\s+|best\s+|signature\s+)?"
        rf"{build_entity_target}\b",

        rf"\bwhat\s+(?:is|are)\s+"
        rf"([A-Za-z][A-Za-z' -]{{1,40}}?)\s+"
        rf"(?:recommended|best|signature)\s+"
        rf"{build_entity_target}\b",

        rf"\b{build_entity_target}\s+should\s+"
        rf"([A-Za-z][A-Za-z' -]{{1,40}}?)\s+"
        rf"(?:use|equip|run|build)\b",

        r"\bshould\s+"
        r"([A-Za-z][A-Za-z' -]{1,40}?)\s+"
        r"(?:use|equip|run|build)\b",

        rf"^\s*([A-Z][A-Za-z' -]{{1,40}}?)\s+"
        rf"(?:recommended|best|signature)\s+"
        rf"{build_entity_target}\b",

        r"\bwho\s+(?:is|are|was|were)\s+"
        r"(?:the\s+)?([A-Za-z][A-Za-z' -]{1,40})",
    ]

    for pattern in patterns:
        match = re.search(pattern, q, re.IGNORECASE)
        if not match:
            continue

        entity = match.group(1)

        entity = re.split(
            r"\?|,|\.|\band\b|\bwhere\b|\bwhat\b",
            entity,
            flags=re.IGNORECASE,
        )[0]

        entity = re.sub(r"\s+", " ", entity).strip(" '")
        entity = re.sub(r"^the\s+", "", entity, flags=re.IGNORECASE)
        entity = re.sub(r"'s$", "", entity, flags=re.IGNORECASE)
        entity = re.sub(
            r"\b(?:"
            r"recommended|best|signature|"
            r"weapon|weapons|build|"
            r"artifact|artifacts|"
            r"talent|talents|"
            r"team|teams|"
            r"rotation|rotations|stats"
            r")$",
            "",
            entity,
            flags=re.IGNORECASE,
        ).strip()

        if not entity:
            continue

        main_entity = entity.lower()
        result = [main_entity]
        for token in main_entity.split():
            if token not in result:
                result.append(token)

        return result

    lookup_entity = extract_lookup_entity(question)
    if lookup_entity:
        main_entity = lookup_entity.lower()
        result = [main_entity]

        for token in main_entity.split():
            if token not in result:
                result.append(token)

        return result

    return []

def prefer_entity_seed_chunks(question: str, chunks: list[dict], min_keep: int = 3) -> list[dict]:
    lookup_entity = extract_lookup_entity(question)

    if not lookup_entity:
        return chunks

    entity_key = normalize_title_key(lookup_entity)

    exact_title: list[dict] = []
    text_matches: list[dict] = []
    related_titles: list[dict] = []
    rest: list[dict] = []

    for row in chunks:
        title = str(row.get("title") or "")
        text = str(row.get("text") or "")
        primary_key = normalize_title_key(primary_page_title(title))
        text_key = normalize_title_key(text[:3000])

        if primary_key == entity_key:
            exact_title.append(row)
        elif (primary_key.startswith(entity_key + " ") or entity_key in primary_key):
            related_titles.append(row)
        elif entity_key in text_key:
            text_matches.append(row)
        else:
            rest.append(row)
    return (exact_title + text_matches + related_titles + rest)

def rerank_chunks(question: str, chunks: list[dict], retrieval_signals: dict[int, float | dict], current_version_ord: int | None = None) -> list[dict]:
    q_terms = tokenize(question)
    intent  = detect_intent(question)
    build_subtypes = (detect_build_subtypes(question) if intent == "build" else set())
    profile = INTENT_PROFILES[intent]
    media_exts = [".jpeg", ".jpg", ".png", ".webp", ".mp4", ".svg", ".ico" , ".webm", ".mp3", ".gif", ".wav", ".ogg", ".woff", ".woff2"]
    ranked = []
    priority = profile.get("source_priority", {})
    required = set(priority.get("required", []))
    excluded = set(priority.get("excluded", []))
    entity_terms = extract_entity_terms(question)
    if intent == "lookup":
        lookup_entity, lookup_facets = (extract_lookup_target(question))
    else:
        lookup_entity = None
        lookup_facets = set()
    lookup_key = (normalize_title_key(lookup_entity) if lookup_entity else "")
    question_l = question.lower()
    asks_for_card = contains_any_marker(question_l,("card", "equipment card", "tcg", "genius invokation"))
    asks_for_skin = "skin" in lookup_facets

    is_identity_question = bool(intent == "biography" and re.match(r"^\s*who\s+(?:is|are|was|were)\b", question, re.IGNORECASE))
    identity_entity = (extract_lookup_entity(question) if is_identity_question else None)
    identity_key = (normalize_title_key(identity_entity) if identity_entity else "")

    for row in chunks:
        chunk_id   = int(row["chunk_id"])
        signal = retrieval_signals.get(chunk_id, 0.0)
        if isinstance(signal, dict):
            base_score = float(signal.get("rrf_score", 0.0))
            in_faiss = bool(signal.get("in_faiss", False))
            in_bm25 = bool(signal.get("in_bm25", False))
            faiss_rank = signal.get("faiss_rank")
            bm25_rank = signal.get("bm25_rank")
            in_hyde = bool(signal.get("in_hyde", False))
            hyde_rank = signal.get("hyde_rank")
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
        matched_subtypes: set[str] = set()

        text_terms    = tokenize(text)
        title_terms   = tokenize(title)
        text_overlap  = len(q_terms & text_terms)
        title_overlap = len(q_terms & title_terms)
        combined_l = f"{title_l}\n{text.lower()}"

        weighted_base = base_score * weight
        lexical_bonus = (0.02 * text_overlap + 0.10 * title_overlap)
        tier_bonus = (0.05 if tier == "primary" else 0.02 if tier == "supplementary" else 0.0)
        intent_source_bonus = float(profile.get("source_bonus", {}).get(source, 0.0) - profile.get("source_penalty", {}).get(source, 0.0))

        title_boost_v = float(profile.get("title_boost_v", 0.0))
        intent_title_boost = (title_boost_v if any(m in title_l for m in profile.get("title_boost", [])) else 0.0)

        title_penalties = list(profile.get("title_penalize", []))
        if intent == "build" and "talent" in build_subtypes:
            talent_related = {
                "normal attack",
                "skill",
                "burst",
                "elemental skill",
                "elemental burst",
            }
            title_penalties = [marker for marker in title_penalties if marker not in talent_related]

        for subtype in build_subtypes:
            subtype_cfg = BUILD_SUBTYPE_PROFILES.get(subtype, {})
            section_markers = subtype_cfg.get("section_markers", ())
            if any(marker in combined_l for marker in section_markers):
                matched_subtypes.add(subtype)

        retrieval_bonus = 0.0
        if in_faiss and in_bm25:
            retrieval_bonus += 0.08

        if intent in {"mechanic", "build", "lore", "biography", "location", "lookup", "version", "general"}:
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

        if in_hyde:
            retrieval_bonus += 0.02
        if in_hyde and (in_faiss or in_bm25):
            retrieval_bonus += 0.04
        if (hyde_rank is not None and hyde_rank <= 5):
            retrieval_bonus += 0.03
        elif (hyde_rank is not None and hyde_rank <= 15):
            retrieval_bonus += 0.02
        
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
        if any(marker in title_l for marker in title_penalties):
            penalty += 0.15

        require_any = profile.get("text_require_any", [])
        if require_any:
            text_l = text.lower()
            if not any(r in text_l for r in require_any):
                penalty += float(profile.get("text_require_penalty", 0.0))

        source = row.get("source") or ""
        if source in excluded:
            penalty += 0.95

        recency_bonus = 0.0
        recency_penalty = 0.0

        recency_explicit = is_recency_sensitive_question(question)
        recency_active = recency_explicit or intent in {"build", "mechanic", "version"}

        asks_for_future_version = contains_any_marker(question_l, ("next version", "upcoming version", "future version", "next patch", "upcoming patch", "expected release", "when will",),)
        if recency_active and current_version_ord is not None:
            row_version_ord = row.get("version_ord")
            if row_version_ord is not None:
                try:
                    row_version_ord = int(row_version_ord)
                    if row_version_ord == current_version_ord:
                        recency_bonus += (0.35 if recency_explicit else 0.12)
                    elif row_version_ord > current_version_ord:
                        if asks_for_future_version:
                            recency_bonus += 0.20
                        else:
                            recency_penalty += 0.45
                    else:
                        distance = (current_version_ord - row_version_ord)
                        if distance == 1:
                            recency_bonus += (0.08 if recency_explicit else 0.04)
                        elif distance <= 3:
                            recency_bonus += (0.03 if recency_explicit else 0.01)
                        else:
                            recency_penalty += (min(0.30, 0.04 * distance) if recency_explicit else min(0.12, 0.02 * distance))

                except (TypeError, ValueError):
                    pass

            elif recency_explicit:
                recency_penalty += 0.04
        
        entity_bonus = 0.0
        title_primary = primary_page_title(title)
        title_primary_key = normalize_title_key(title_primary)
        if identity_key:
            if title_primary_key == identity_key:
                # Main page: "Columbina"
                entity_bonus += 1.25

            elif title_primary_key == (identity_key + " profile"):
                # Useful biography subpage.
                entity_bonus += 0.35

            elif title_primary_key.startswith(identity_key + " "):
                entity_bonus += 0.08

            noisy_identity_subpages = (
                "storyline",
                "companion",
                "dressing room",
                "elemental skill",
                "elemental burst",
                "normal attack",
                "constellation",
                " c1",
                " c2",
                " c3",
                " c4",
                " c5",
                " c6",
            )

            if any(marker in title_l for marker in noisy_identity_subpages):
                penalty += 0.25

        if lookup_entity:
            similarity = entity_title_similarity(lookup_entity, title_primary)
            if similarity >= 0.98:
                entity_bonus += 1.00
            elif similarity >= 0.90:
                entity_bonus += 0.65
            elif similarity >= 0.84:
                entity_bonus += 0.35

        if lookup_key:
            if title_primary_key == lookup_key:
                entity_bonus += 1.00
            elif title_primary_key.startswith(lookup_key + " "):
                entity_bonus += 0.10
            elif lookup_key in title_primary_key:
                entity_bonus += 0.05

            if not asks_for_card and "equipment card" in title_l:
                penalty += 0.80

            if not asks_for_skin and any(marker in title_l for marker in ("dynamic skin", "lustrous skin")):
                penalty += 0.55

            if any(marker in title_l for marker in ("/change history", "/gallery", "change history")):
                penalty += 0.45
        
        elif entity_terms:
            main_entity = entity_terms[0]
            title_norm = normalize_title_key(title)
            if title_norm == main_entity:
                entity_bonus += 0.40
            elif title_norm.startswith(main_entity + " "):
                entity_bonus += 0.30
            elif main_entity in title_l:
                entity_bonus += 0.15

            if intent == "build":
                if "build" in title_l:
                    if main_entity in title_l:
                        entity_bonus += 0.25
                    else:
                        penalty += 0.50

            if intent == "biography":
                bad_profile_titles = (
                    "avatar",
                    "namecard",
                    "fan art contest",
                    "quest item",
                    "normal attack",
                    "constellation",
                    "utility passive",
                    "taking pictures",
                    "change history",
                )
                if any(x in title_l for x in bad_profile_titles):
                    entity_bonus -= 0.20

        if intent == "build" and entity_terms:
            main_entity = entity_terms[0]
            text_l = text.lower()

            entity_in_title = main_entity in title_l
            entity_in_text = main_entity in text_l[:3000]

            if not entity_in_title and not entity_in_text:
                penalty += 1.00

            if "build" in title_l:
                if entity_in_title:
                    entity_bonus += 0.35
                else:
                    penalty += 1.25

            if entity_in_title and matched_subtypes:
                retrieval_bonus += 0.35

        if intent == "build" and build_subtypes:
            if matched_subtypes:
                best_bonus = max(float(BUILD_SUBTYPE_PROFILES[subtype].get("bonus", 0.40)) for subtype in matched_subtypes)
                retrieval_bonus += best_bonus
                if len(matched_subtypes) > 1:
                    retrieval_bonus += 0.08 * (len(matched_subtypes) - 1)
            else:
                missing_penalty = min(float(BUILD_SUBTYPE_PROFILES[subtype].get("missing_penalty", 0.35))for subtype in build_subtypes)
                penalty += missing_penalty

        if "skin" in lookup_facets:
            skin_markers = (
                "skin",
                "outfit",
                "costume",
                "character outfit",
                "attire",
                "appearance",
                "dynamic skin",
                "lustrous skin",
            )

            has_skin_evidence = any(marker in combined_l for marker in skin_markers)

            if has_skin_evidence:
                retrieval_bonus += 0.45
            else:
                penalty += 0.20

            if lookup_entity:
                entity_key = normalize_title_key(lookup_entity)
                title_key = normalize_title_key(title)
                text_key = normalize_title_key(text[:3000])
                if (entity_key in title_key or entity_key in text_key):
                    retrieval_bonus += 0.30
                else:
                    penalty += 0.25

        final_score = (
            weighted_base
            + lexical_bonus
            + entity_bonus
            + tier_bonus
            + intent_source_bonus
            + intent_title_boost
            + retrieval_bonus
            + recency_bonus
            - recency_penalty
            - penalty
        )
        row["_rerank_score"] = float(final_score)
        ranked.append((final_score, row))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in ranked]