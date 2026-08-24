from __future__ import annotations

import json
import logging
import re
from typing import Any

from .utils_graph import entity_key
from qna.generators import generate

log = logging.getLogger(__name__)

ALLOWED_ENTITY_TYPES={
    "character",
    "deity",
    "faction",
    "nation",
    "region",
    "location",
    "organization",
    "species",
    "item",
    "weapon",
    "artifact",
    "quest",
    "event",
    "concept",
    "unknown",
}

ALLOWED_RELATION_TYPES={
    "PARENT_OF",
    "CHILD_OF",
    "SIBLING_OF",
    "SPOUSE_OF",
    "ANCESTOR_OF",

    "FRIEND_OF",
    "ALLY_OF",
    "ENEMY_OF",
    "RIVAL_OF",
    "COMPANION_OF",

    "MENTOR_OF",
    "STUDENT_OF",
    "DISCIPLE_OF",

    "MEMBER_OF",
    "LEADER_OF",
    "FOUNDER_OF",
    "AFFILIATED_WITH",
    "SERVES",
    "WORKS_FOR",

    "CREATED",
    "CREATED_BY",
    "OWNS",
    "OWNED_BY",
    "WIELDS",
    "WIELDED_BY",

    "LOCATED_IN",
    "RESIDES_IN",
    "ORIGINATES_FROM",
    "RULES",
    "PROTECTS",

    "FOUGHT",
    "DEFEATED",
    "KILLED",
    "OPPOSES",

    "WORSHIPS",
    "WORSHIPPED_BY",

    "SUCCESSOR_OF",
    "PREDECESSOR_OF",

    "PARTICIPATED_IN",
    "INVOLVED_IN",

    "ASSOCIATED_WITH",
    "RELATED_TO",
}

def build_extraction_prompt(title: str, text: str) -> str:
    return f"""
You extract a factual knowledge graph from Genshin Impact wiki text.

Article title:
{title}

Text:
{text}

Return JSON only with this exact structure:
{{
  "entities": [
    {{
      "name": "canonical entity name",
      "type": "character|deity|faction|nation|region|location|affiliations|organization|species|item|weapon|artifact|quest|event|concept|unknown",
      "aliases": []
    }}
  ],
  "relationships": [
    {{
      "source": "entity name",
      "target": "entity name",
      "type": "RELATION_TYPE",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- Extract only facts explicitly supported by the supplied text.
- Do not use outside Genshin knowledge.
- Do not invent relationships.
- Both source and target must appear in the entities list.
- Use canonical entity names when possible.
- Relationship types should be short uppercase forms such as FRIEND_OF, MEMBER_OF, SERVES, CREATED_BY, LOCATED_IN, ASSOCIATED_WITH, ENEMY_OF, SIBLING_OF, PARENT_OF, ENVOY_OF.
- Confidence must be between 0.0 and 1.0.
- If there are no supported relationships, return an empty relationships list.
- Do not include Markdown fences.
""".strip()

def parse_extraction_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Extractor returned no JSON object")
        data = json.loads(raw[start:end+1])

    if not isinstance(data, dict):
        raise ValueError("Extractor output must be a JSON object")
    return data

def normalize_extraction(data: dict[str, Any]) -> dict[str, Any]:
    entities = []
    relationships = []

    for row in data.get("entities", []):
        if not isinstance(row, dict):
            continue

        name = str(row.get("name") or "").strip()
        entity_type = str(row.get("type") or "unknown").strip().lower()

        if not name:
            continue
        if entity_type not in ALLOWED_ENTITY_TYPES:
            entity_type = "unknown"

        aliases = row.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []

        aliases = [str(alias).strip() for alias in aliases if str(alias).strip()]
        entities.append({
            "name":name,
            "type":entity_type,
            "aliases":aliases,
        })

    known_keys={entity_key(row["name"]) for row in entities if entity_key(row["name"])}

    for row in data.get("relationships", []):
        if not isinstance(row, dict):
            continue

        source = str(row.get("source") or "").strip()
        target = str(row.get("target") or "").strip()
        relation_type=str(row.get("type") or "RELATED_TO").strip().upper().replace(" ","_")
        if relation_type not in ALLOWED_RELATION_TYPES:
            relation_type="RELATED_TO"

        if not source or not target or not relation_type:
            continue

        source_key=entity_key(source)
        target_key=entity_key(target)
        if not source_key or not target_key:
            continue
        if source_key not in known_keys:
            continue
        if target_key not in known_keys:
            continue
        if source_key==target_key:
            continue

        try:
            confidence = float(row.get("confidence",0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        relationships.append({
            "source": source,
            "target": target,
            "type": relation_type,
            "confidence": confidence,
        })

    return {
        "entities": entities,
        "relationships": relationships,
    }

def extract_graph_from_chunk(cfg: dict[str, Any], *, title: str, text: str) -> dict[str, Any]:
    ncfg = cfg.get("neo4j", {}) or {}
    num_predict = int(ncfg.get("extraction_num_predict", 2048))
    num_ctx = int(ncfg.get("extraction_num_ctx",4096))
    model = str(ncfg.get("extraction_model", "qwen3.6:27b"))
    temperature = float(ncfg.get("extraction_temperature", 0.0))
    prompt = build_extraction_prompt(title, text)
    raw = generate(cfg, prompt, model_override=model, options_override={
        "temperature": temperature,
        "num_ctx": num_ctx,
        "num_predict": num_predict}, 
    think_override=False).strip()

    if not raw:
        return {
            "entities": [],
            "relationships": []
        }

    try:
        data=parse_extraction_json(raw)
    except json.JSONDecodeError:
        log.warning("[GRAPH] JSON parse failed; retrying with larger output budget title=%r old_num_predict=%d", title, num_predict,)
        retry_predict=min(num_predict * 2,4096)
        retry_ctx=max(num_ctx,retry_predict + 2048)
        raw=generate(cfg, prompt, model_override=model, options_override={
        "temperature":temperature,
        "num_ctx":retry_ctx,
        "num_predict":retry_predict})

        data=parse_extraction_json(raw)
    return normalize_extraction(data)