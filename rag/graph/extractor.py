from __future__ import annotations

import json
import logging
import re
from typing import Any

from qna.generators import generate

log = logging.getLogger(__name__)

ALLOWED_ENTITY_TYPES = {
    "character",
    "deity",
    "faction",
    "nation",
    "region",
    "location",
    "organization",
    "affiliations",
    "species",
    "item",
    "weapon",
    "artifact",
    "quest",
    "event",
    "concept",
    "unknown",
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

    known_names = {["name"].casefold() for row in entities}

    for row in data.get("relationships", []):
        if not isinstance(row, dict):
            continue

        source = str(row.get("source") or "").strip()
        target = str(row.get("target") or "").strip()
        relation_type = str(row.get("type") or "").strip().upper()

        if not source or not target or not relation_type:
            continue
        if source.casefold() not in known_names:
            continue
        if target.casefold() not in known_names:
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
    model = str(ncfg.get("extraction_model", "qwen3.6:27b"))
    prompt = build_extraction_prompt(title, text)
    raw = generate(cfg, prompt, model_override=model, options_override={
        "temperature": float(ncfg.get("extraction_temperature", 0.0))
    }, think_override=False).strip()

    if not raw:
        return {
            "entities": [],
            "relationships": []
        }

    data = parse_extraction_json(raw)
    return normalize_extraction(data)