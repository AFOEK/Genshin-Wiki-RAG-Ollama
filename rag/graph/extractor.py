from __future__ import annotations

import json
import logging
import re
from typing import Any

from .utils_graph import entity_key
from qna.generators import generate

log = logging.getLogger(__name__)

ALLOWED_ENTITY_TYPES = {
    # -------------------------------------------------------------------------
    # People / sentient beings
    # -------------------------------------------------------------------------
    "character",
    "playable_character",
    "npc",
    "deity",
    "archon",
    "adeptus",
    "spirit",
    "human",
    "species",
    "creature",
    "animal",
    "monster",
    "enemy",
    "boss",
    "construct",
    "dragon",

    # -------------------------------------------------------------------------
    # Political / social / institutional
    # -------------------------------------------------------------------------
    "faction",
    "organization",
    "affiliation",
    "nation",
    "government",
    "military",
    "clan",
    "family",
    "tribe",
    "guild",
    "company",
    "academy",
    "religion",
    "cult",

    # -------------------------------------------------------------------------
    # Geography / world structure
    # -------------------------------------------------------------------------
    "world",
    "realm",
    "nation",
    "region",
    "subregion",
    "area",
    "location",
    "city",
    "village",
    "settlement",
    "landmark",
    "point_of_interest",
    "domain",
    "dungeon",
    "ruin",
    "temple",
    "palace",
    "building",
    "island",
    "mountain",
    "forest",
    "river",
    "lake",
    "sea",

    # -------------------------------------------------------------------------
    # Equipment / inventory objects
    # -------------------------------------------------------------------------
    "item",
    "weapon",
    "weapon_type",
    "artifact",
    "artifact_set",
    "artifact_piece",
    "gadget",
    "tool",
    "equipment",

    # -------------------------------------------------------------------------
    # Materials / resources
    # -------------------------------------------------------------------------
    "material",
    "resource",
    "ore",
    "plant",
    "ingredient",
    "local_specialty",
    "enemy_drop",
    "character_ascension_material",
    "weapon_ascension_material",
    "talent_material",

    # -------------------------------------------------------------------------
    # Food / cooking
    # -------------------------------------------------------------------------
    "food",
    "dish",
    "recipe",
    "special_dish",
    "ingredient",

    # -------------------------------------------------------------------------
    # Economy
    # -------------------------------------------------------------------------
    "currency",
    "shop",
    "merchant",
    "reward",

    # -------------------------------------------------------------------------
    # Character progression / mechanics
    # -------------------------------------------------------------------------
    "talent",
    "skill",
    "ability",
    "constellation",
    "ascension",
    "level",
    "stat",
    "buff",
    "debuff",

    # -------------------------------------------------------------------------
    # Elemental / supernatural systems
    # -------------------------------------------------------------------------
    "element",
    "elemental_reaction",
    "vision",
    "delusion",
    "gnosis",
    "energy",
    "power",
    "curse",
    "blessing",

    # -------------------------------------------------------------------------
    # Quests / commissions / story content
    # -------------------------------------------------------------------------
    "quest",
    "archon_quest",
    "story_quest",
    "world_quest",
    "commission",
    "hangout_event",
    "quest_series",
    "quest_chapter",
    "story_chapter",

    # -------------------------------------------------------------------------
    # Events / activities
    # -------------------------------------------------------------------------
    "event",
    "event_series",
    "activity",
    "challenge",
    "trial",
    "game_mode",

    # -------------------------------------------------------------------------
    # Achievements
    # -------------------------------------------------------------------------
    "achievement",
    "achievement_series",
    "achievement_category",

    # -------------------------------------------------------------------------
    # Combat
    # -------------------------------------------------------------------------
    "enemy",
    "boss",
    "enemy_group",
    "combat_encounter",
    "attack",
    "status_effect",

    # -------------------------------------------------------------------------
    # Furnishings / Serenitea Pot
    # -------------------------------------------------------------------------
    "furnishing",
    "furnishing_set",
    "building",
    "decoration",

    # -------------------------------------------------------------------------
    # Collectibles / exploration
    # -------------------------------------------------------------------------
    "collectible",
    "oculus",
    "chest",
    "key_item",
    "sigil",
    "token",

    # -------------------------------------------------------------------------
    # Books / documents / textual media
    # -------------------------------------------------------------------------
    "book",
    "book_series",
    "document",
    "letter",
    "note",
    "diary",
    "record",
    "inscription",
    "text",

    # -------------------------------------------------------------------------
    # Other media
    # -------------------------------------------------------------------------
    "soundtrack",
    "song",
    "album",
    "cutscene",
    "trailer",
    "voice_line",

    # -------------------------------------------------------------------------
    # Character cosmetics
    # -------------------------------------------------------------------------
    "outfit",
    "namecard",

    # -------------------------------------------------------------------------
    # History / lore
    # -------------------------------------------------------------------------
    "historical_event",
    "war",
    "battle",
    "incident",
    "era",
    "period",
    "dynasty",
    "civilization",
    "culture",
    "tradition",
    "ritual",
    "festival",
    "legend",
    "myth",
    "prophecy",

    # -------------------------------------------------------------------------
    # Identity / roles / abstract lore
    # -------------------------------------------------------------------------
    "title",
    "role",
    "occupation",
    "rank",
    "concept",
    "ideology",
    "law",
    "language",
    "script",

    # -------------------------------------------------------------------------
    # Fallback
    # -------------------------------------------------------------------------
    "unknown",
}

RELATION_GROUP_TYPES = {
    "family": {
        "PARENT_OF","CHILD_OF","SIBLING_OF","SPOUSE_OF",
        "ANCESTOR_OF","DESCENDANT_OF","RELATIVE_OF",
        "ADOPTED_BY","GUARDIAN_OF","WARD_OF",
    },
    "social": {
        "FRIEND_OF","ALLY_OF","ENEMY_OF","RIVAL_OF","COMPANION_OF",
        "ACQUAINTANCE_OF","PARTNER_OF","LOVER_OF","BETRAYED",
        "TRUSTS","DISLIKES","RESPECTS","FEARS",
    },
    "teaching": {
        "MENTOR_OF","TEACHER_OF","STUDENT_OF","DISCIPLE_OF",
        "MASTER_OF","SLAVE_OF",
    },
    "organization": {
        "ENVOY_OF","COMMANDS","COMMANDED_BY","SUBORDINATE_OF",
        "SUPERVISES","GOVERNS","GOVERNED_BY","REPRESENTS",
        "REPRESENTED_BY","APPOINTED_BY","MEMBER_OF","LEADER_OF",
        "FOUNDER_OF","AFFILIATED_WITH","SERVES","WORKS_FOR",
        "EMPLOYED_BY","EMPLOYS","HOLDS_POSITION","SUCCEEDED_BY",
        "RULES","PROTECTS",
    },
    "creation": {
        "CREATED","CREATED_BY","OWNS","OWNED_BY",
        "WIELDS","WIELDED_BY","DISCOVERED","DISCOVERED_BY",
        "INVENTED","INVENTED_BY","FORGED","FORGED_BY",
        "BUILT","BUILT_BY","USES","USED_BY",
        "CARRIES","CARRIED_BY","CREATION_OF",
    },
    "location": {
        "LOCATED_IN","RESIDES_IN","LIVES_IN","ORIGINATES_FROM",
        "PART_OF","CONTAINS","NEAR","BORDERS","CONNECTED_TO",
        "TRAVELS_TO","BORN_IN","DIED_IN","VISITED",
        "RULES","PROTECTS","TAKES_PLACE_IN","OCCURRED_IN",
    },
    "combat": {
        "ENEMY_OF","FOUGHT","DEFEATED","KILLED","OPPOSES",
        "ATTACKED","DEFENDED","SAVED","CAPTURED","IMPRISONED",
        "ESCAPED_FROM","SEALED","SEALED_BY",
        "DESTROYED","DESTROYED_BY",
    },
    "religion": {
        "WORSHIPS","WORSHIPPED_BY","HAS_GNOSIS",
    },
    "element": {
        "USES_ELEMENT","ASSOCIATED_WITH_ELEMENT","GRANTS_ELEMENT",
        "RESONATES_WITH","HAS_VISION","HAS_DELUSION","HAS_GNOSIS",
    },
    "event": {
        "PARTICIPATED_IN","INVOLVED_IN","APPEARS_IN","FEATURED_IN",
        "TAKES_PLACE_IN","OCCURRED_IN","OCCURRED_DURING",
        "TRIGGERS","TRIGGERED_BY","PRECEDES","FOLLOWS",
        "CONTEMPORARY_OF","CAUSES","CAUSED_BY","RESULTED_IN",
    },
    "chronology": {
        "SUCCESSOR_OF","PREDECESSOR_OF","SUCCEEDED_BY",
        "PRECEDES","FOLLOWS","CONTEMPORARY_OF",
        "CAUSES","CAUSED_BY","RESULTED_IN",
    },
    "quest": {
        "STARTS_QUEST","APPEARS_IN","FEATURED_IN",
        "REQUIRED_FOR","REWARDS","REWARDED_BY",
        "UNLOCKS","UNLOCKED_BY","TRIGGERS","TRIGGERED_BY",
        "TAKES_PLACE_IN","PARTICIPATED_IN","INVOLVED_IN",
    },
    "items": {
        "MATERIAL_FOR","CRAFTED_FROM","CRAFTS_INTO",
        "OBTAINED_FROM","DROPPED_BY","PURCHASED_FROM",
        "SOLD_BY","REQUIRED_BY","USED_FOR",
        "ASCENDS","USED_TO_ASCEND","UPGRADES_TALENT",
        "HAS_TALENT","HAS_CONSTELLATION","USES_WEAPON_TYPE",
        "USES","USED_BY","CARRIES","CARRIED_BY",
        "OWNS","OWNED_BY","WIELDS","WIELDED_BY",
    },
    "identity": {
        "IDENTITY_OF","KNOWN_AS","HAS_TITLE",
        "FORM_OF","INCARNATION_OF","CREATION_OF",
    },
    "generic": {
        "ASSOCIATED_WITH","RELATED_TO",
    },
}

RELATION_GROUP_RULES = {
    "family":
        "Family relations require explicitly stated kinship; matching surnames or clans are insufficient.",
    "social":
        "Friendship, alliance, rivalry, romance, trust, or hostility must be explicitly supported.",
    "teaching":
        "MENTOR_OF/TEACHER_OF/STUDENT_OF/DISCIPLE_OF are teaching relations; MASTER_OF is not automatically mentorship.",
    "organization":
        "MEMBER_OF requires membership; AFFILIATED_WITH is weaker; SERVES does not automatically mean employment.",
    "creation":
        "Distinguish CREATED, INVENTED, FORGED and BUILT; distinguish ownership, carrying, use and wielding.",
    "location":
        "Distinguish residence, origin, birthplace, temporary presence, structural containment and geographic location.",
    "combat":
        "Do not upgrade opposition to combat, combat to defeat, or defeat to killing without explicit evidence.",
    "religion":
        "Worship relations require explicit worship; merely being a deity does not establish worship.",
    "element":
        "Distinguish active elemental use, elemental association, Vision, Delusion and Gnosis possession.",
    "event":
        "Mere mention in an event is not participation; causality requires explicit causal evidence.",
    "chronology":
        "PRECEDES/FOLLOWS are chronological only; succession and causality require explicit support.",
    "quest":
        "Distinguish appearing in, participating in, starting, unlocking, rewarding and locating a quest.",
    "items":
        "Distinguish ownership, use, drops, acquisition, crafting, rewards and progression requirements.",
    "identity":
        "Aliases normally belong in aliases; IDENTITY_OF/FORM_OF/INCARNATION_OF require explicit identity claims.",
}

ENTITY_TYPE_ALIASES = {
    "achievements": "achievement",
    "achievement_set": "achievement_series",
    "artifactset": "artifact_set",
    "worldquest": "world_quest",
    "non_playable_character": "npc",
}

ALLOWED_RELATION_TYPES = {
    # -------------------------------------------------------------------------
    # Family and kinship
    # -------------------------------------------------------------------------
    "PARENT_OF",
    "CHILD_OF",
    "SIBLING_OF",
    "SPOUSE_OF",
    "ANCESTOR_OF",
    "DESCENDANT_OF",
    "RELATIVE_OF",
    "ADOPTED_BY",
    "GUARDIAN_OF",
    "WARD_OF",

    # -------------------------------------------------------------------------
    # Friendship, alliance, rivalry, and companionship
    # -------------------------------------------------------------------------
    "FRIEND_OF",
    "ALLY_OF",
    "ENEMY_OF",
    "RIVAL_OF",
    "COMPANION_OF",

    # -------------------------------------------------------------------------
    # Teaching, mentorship, service, and master-dependent relationships
    # -------------------------------------------------------------------------
    "MENTOR_OF",
    "TEACHER_OF",
    "STUDENT_OF",
    "DISCIPLE_OF",
    "ENVOY_OF",
    "MASTER_OF",
    "SLAVE_OF",

    # -------------------------------------------------------------------------
    # Personal and interpersonal relationships
    # -------------------------------------------------------------------------
    "ACQUAINTANCE_OF",
    "PARTNER_OF",
    "LOVER_OF",
    "BETRAYED",
    "TRUSTS",
    "DISLIKES",
    "RESPECTS",
    "FEARS",

    # -------------------------------------------------------------------------
    # Command, hierarchy, governance, and representation
    # -------------------------------------------------------------------------
    "COMMANDS",
    "COMMANDED_BY",
    "SUBORDINATE_OF",
    "SUPERVISES",
    "GOVERNS",
    "GOVERNED_BY",
    "REPRESENTS",
    "REPRESENTED_BY",
    "APPOINTED_BY",

    # -------------------------------------------------------------------------
    # Organization membership and institutional affiliation
    # -------------------------------------------------------------------------
    "MEMBER_OF",
    "LEADER_OF",
    "FOUNDER_OF",
    "AFFILIATED_WITH",
    "SERVES",
    "WORKS_FOR",

    # -------------------------------------------------------------------------
    # Creation, ownership, and weapon possession
    # -------------------------------------------------------------------------
    "CREATED",
    "CREATED_BY",
    "OWNS",
    "OWNED_BY",
    "WIELDS",
    "WIELDED_BY",

    # -------------------------------------------------------------------------
    # Geography, residence, travel, origin, and territorial relationships
    # -------------------------------------------------------------------------
    "LOCATED_IN",
    "RESIDES_IN",
    "LIVES_IN",
    "ORIGINATES_FROM",
    "PART_OF",
    "CONTAINS",
    "NEAR",
    "BORDERS",
    "CONNECTED_TO",
    "TRAVELS_TO",
    "BORN_IN",
    "DIED_IN",
    "VISITED",
    "RULES",
    "PROTECTS",

    # -------------------------------------------------------------------------
    # Employment, occupation, and formal positions
    # -------------------------------------------------------------------------
    "EMPLOYED_BY",
    "EMPLOYS",
    "HOLDS_POSITION",
    "SUCCEEDED_BY",

    # -------------------------------------------------------------------------
    # Conflict and combat
    # -------------------------------------------------------------------------
    "FOUGHT",
    "DEFEATED",
    "KILLED",
    "OPPOSES",

    # -------------------------------------------------------------------------
    # Religion and worship
    # -------------------------------------------------------------------------
    "WORSHIPS",
    "WORSHIPPED_BY",

    # -------------------------------------------------------------------------
    # Succession and predecessor relationships
    # -------------------------------------------------------------------------
    "SUCCESSOR_OF",
    "PREDECESSOR_OF",

    # -------------------------------------------------------------------------
    # Participation and involvement
    # -------------------------------------------------------------------------
    "PARTICIPATED_IN",
    "INVOLVED_IN",

    # -------------------------------------------------------------------------
    # Generic semantic fallback relationships
    # -------------------------------------------------------------------------
    "ASSOCIATED_WITH",
    "RELATED_TO",

    # -------------------------------------------------------------------------
    # Elemental abilities and elemental associations
    # -------------------------------------------------------------------------
    "USES_ELEMENT",
    "ASSOCIATED_WITH_ELEMENT",
    "GRANTS_ELEMENT",
    "RESONATES_WITH",

    # -------------------------------------------------------------------------
    # Genshin-specific supernatural objects / powers
    # -------------------------------------------------------------------------
    "HAS_VISION",
    "HAS_DELUSION",
    "HAS_GNOSIS",

    # -------------------------------------------------------------------------
    # Combat actions, protection, imprisonment, sealing, and destruction
    # -------------------------------------------------------------------------
    "ATTACKED",
    "DEFENDED",
    "SAVED",
    "CAPTURED",
    "IMPRISONED",
    "ESCAPED_FROM",
    "SEALED",
    "SEALED_BY",
    "DESTROYED",
    "DESTROYED_BY",

    # -------------------------------------------------------------------------
    # Discovery, invention, construction, usage, and possession
    # -------------------------------------------------------------------------
    "DISCOVERED",
    "DISCOVERED_BY",
    "INVENTED",
    "INVENTED_BY",
    "FORGED",
    "FORGED_BY",
    "BUILT",
    "BUILT_BY",
    "USES",
    "USED_BY",
    "CARRIES",
    "CARRIED_BY",

    # -------------------------------------------------------------------------
    # Quest, event, reward, trigger, and appearance relationships
    # -------------------------------------------------------------------------
    "STARTS_QUEST",
    "APPEARS_IN",
    "FEATURED_IN",
    "REQUIRED_FOR",
    "REWARDS",
    "REWARDED_BY",
    "UNLOCKS",
    "UNLOCKED_BY",
    "TRIGGERS",
    "TRIGGERED_BY",
    "TAKES_PLACE_IN",
    "OCCURRED_IN",
    "OCCURRED_DURING",

    # -------------------------------------------------------------------------
    # Chronology, causality, and historical relationships
    # -------------------------------------------------------------------------
    "PRECEDES",
    "FOLLOWS",
    "CONTEMPORARY_OF",
    "CAUSES",
    "CAUSED_BY",
    "RESULTED_IN",

    # -------------------------------------------------------------------------
    # Materials, crafting, acquisition, drops, shops, and usage
    # -------------------------------------------------------------------------
    "MATERIAL_FOR",
    "CRAFTED_FROM",
    "CRAFTS_INTO",
    "OBTAINED_FROM",
    "DROPPED_BY",
    "PURCHASED_FROM",
    "SOLD_BY",
    "REQUIRED_BY",
    "USED_FOR",

    # -------------------------------------------------------------------------
    # Character progression and gameplay mechanics
    # -------------------------------------------------------------------------
    "ASCENDS",
    "USED_TO_ASCEND",
    "UPGRADES_TALENT",
    "HAS_TALENT",
    "HAS_CONSTELLATION",
    "USES_WEAPON_TYPE",

    # -------------------------------------------------------------------------
    # Identity, aliases, titles, forms, incarnations, and origin
    # -------------------------------------------------------------------------
    "IDENTITY_OF",
    "KNOWN_AS",
    "HAS_TITLE",
    "FORM_OF",
    "INCARNATION_OF",
    "CREATION_OF",
}

ALLOWED_ENTITY_TYPE_SET = set(ALLOWED_ENTITY_TYPES)
ALLOWED_RELATION_TYPE_SET = set(ALLOWED_RELATION_TYPES)

entity_types = "|".join(sorted(ALLOWED_ENTITY_TYPE_SET))
relation_types = ", ".join(sorted(ALLOWED_RELATION_TYPE_SET))

def relation_types_for_groups(groups: tuple[str,...] | list[str] | None, *, full_ontology: bool=False) -> list[str]:
    if full_ontology:
        return sorted(ALLOWED_RELATION_TYPE_SET)

    selected=set(RELATION_GROUP_TYPES["generic"])

    for group in groups or ():
        selected.update(RELATION_GROUP_TYPES.get(group, ()))

    selected &= ALLOWED_RELATION_TYPE_SET
    return sorted(selected)


def relation_rules_for_groups(groups: tuple[str,...] | list[str] | None, *, full_ontology: bool=False) -> str:
    if full_ontology:
        selected_groups=RELATION_GROUP_RULES.keys()
    else:
        selected_groups=groups or ()

    lines=[f"- {RELATION_GROUP_RULES[group]}" for group in selected_groups if group in RELATION_GROUP_RULES]
    return "\n".join(lines)

def build_extraction_prompt(title: str, text: str, *, relation_groups: tuple[str,...] | list[str] | None=None, full_ontology: bool=False) -> str:
    allowed_relations=relation_types_for_groups(relation_groups, full_ontology=full_ontology)
    relation_types=", ".join(allowed_relations)
    group_rules=relation_rules_for_groups(relation_groups, full_ontology=full_ontology)

    return f"""
Extract a factual knowledge graph from the supplied Genshin Impact wiki text.

Title: {title}

Text:
{text}

Allowed entity types:
{entity_types}

Allowed relationship types:
{relation_types}

CRITICAL:
- Relationship type MUST exactly match one of the allowed relationship types above.
- NEVER invent, paraphrase, conjugate, pluralize, or create relationship types.
- If no allowed relationship accurately represents a fact, omit that fact.
- Use only information explicitly supported by the supplied text.
- Do not use outside knowledge.

Return compact JSON only:
{{"entities":[{{"name":"canonical name","type":"entity_type","aliases":[]}}],
"relationships":[{{"source":"entity name","target":"entity name","type":"RELATION_TYPE","confidence":0.0}}]}}

Rules:
- Both relationship endpoints must appear in "entities".
- Use canonical entity names when possible.
- Prefer specific relationships over ASSOCIATED_WITH or RELATED_TO.
- Mere co-occurrence does not establish a relationship.
- Do not infer friendship, family, membership, ownership, causality, location, or allegiance from proximity alone.
- Respect negation, uncertainty, speculation, and historical context.
- Distinguish gameplay mechanics from in-universe lore.
- Do not output duplicate relationships.
- Do not output both a relationship and its inverse for the same fact.
- Never output self-relations.
- Prefer at most 25 entities and 20 relationships.
- If additional important explicit facts exist, they may be included.
- Never exceed 40 entities or 30 relationships.
- Include at most 3 useful aliases per entity.
- Confidence 0.95-1.00 = explicit/direct.
- Confidence 0.90-0.94 = clearly supported but indirect.
- Confidence 0.85-0.89 = supported with minor interpretation.
- Below 0.85 = omit the relationship.

Relevant semantic rules:
{group_rules or "- Apply the relationship definitions literally."}

If no supported relationships exist, return an empty "relationships" list.
If no supported entities exist, return empty "entities" and "relationships" lists.
Do not include Markdown or explanatory text.
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

def normalize_extraction(data: dict[str, Any], min_confidence: float = 0.85) -> dict[str, Any]:
    entities = []
    relationships = []

    for row in data.get("entities", []):
        if not isinstance(row, dict):
            continue

        name = str(row.get("name") or "").strip()
        entity_type=str(row.get("type") or "unknown").strip().lower()
        entity_type=re.sub(r"[\s\-]+","_",entity_type)
        entity_type=ENTITY_TYPE_ALIASES.get(entity_type,entity_type)

        if not name:
            continue
        
        if entity_type not in ALLOWED_ENTITY_TYPE_SET:
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
        relation_type=str(row.get("type") or "").strip().upper().replace(" ","_")
        if relation_type not in ALLOWED_RELATION_TYPE_SET:
            log.info("[GRAPH] rejected unsupported relation type=%r source=%r target=%r", relation_type, source, target)
            continue

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
            confidence = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))
        if confidence < min_confidence:
            log.info("[GRAPH] rejected low-confidence relation source=%r type=%s target=%r confidence=%.3f threshold=%.3f", source, relation_type, target, confidence, min_confidence,)
            continue

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

def run_graph_model(cfg: dict[str, Any], *, prompt: str, model: str, temperature: float, num_ctx: int, num_predict: int) -> str:
    return generate(
        cfg, prompt, model_override=model,
        options_override={
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        think_override=False,
    ).strip()


def extract_graph_from_chunk(cfg: dict[str, Any], *, title: str, text: str, filter_score: int = 0, filter_groups: tuple[str, ...] = ()) -> dict[str, Any]:
    ncfg = cfg.get("neo4j", {}) or {}
    primary_model = str(ncfg.get("extraction_model", "qwen3.5:9b"))
    fallback_model = str(ncfg.get("fallback_model", "qwen3.8:27b"))
    temperature = float(ncfg.get("extraction_temperature", 0.0))
    min_confidence = float(ncfg.get("min_relation_confidence", 0.85))
    primary_ctx = int(ncfg.get("extraction_num_ctx", 8192))
    primary_predict = int(ncfg.get("extraction_num_predict", 2048))
    fallback_ctx = int(ncfg.get("fallback_num_ctx", 16384))
    fallback_predict = int(ncfg.get("fallback_num_predict", 4096))
    fallback_on_empty = bool(ncfg.get("fallback_on_empty", True))
    fallback_min_score = int(ncfg.get("fallback_min_score", 3))
    primary_prompt=build_extraction_prompt(title, text, relation_groups=filter_groups, full_ontology=False)
    fallback_prompt=build_extraction_prompt(title, text, relation_groups=filter_groups, full_ontology=True)

    try:
        raw = run_graph_model(cfg, prompt=primary_prompt, model=primary_model, temperature=temperature, num_ctx=primary_ctx, num_predict=primary_predict,)
        if not raw:
            raise ValueError("Primary extractor returned empty output")

        extraction = normalize_extraction(parse_extraction_json(raw), min_confidence)

    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("[GRAPH] primary extractor failed title=%r model=%s score=%d error=%s; using fallback=%s", title, primary_model, filter_score, exc, fallback_model,)
        return run_fallback_graph_model(cfg, prompt=fallback_prompt, title=title, model=fallback_model, temperature=temperature, num_ctx=fallback_ctx, num_predict=fallback_predict, min_confidence=min_confidence,)

    if (fallback_on_empty and not extraction["relationships"] and filter_score >= fallback_min_score):
        log.info("[GRAPH] primary returned no accepted relations title=%r model=%s score=%d; fallback=%s", title, primary_model, filter_score, fallback_model,)
        return run_fallback_graph_model(cfg, prompt=fallback_prompt, title=title, model=fallback_model, temperature=temperature, num_ctx=fallback_ctx, num_predict=fallback_predict, min_confidence=min_confidence,)

    return extraction


def run_fallback_graph_model(cfg: dict[str, Any], *, prompt: str, title: str, model: str, temperature: float, num_ctx: int, num_predict: int, min_confidence: float) -> dict[str, Any]:
    raw = run_graph_model(cfg, prompt=prompt, model=model, temperature=temperature, num_ctx=num_ctx, num_predict=num_predict,)
    if not raw:
        raise ValueError(f"Fallback extractor returned empty output: {model}")

    extraction = normalize_extraction(parse_extraction_json(raw), min_confidence)
    log.info("[GRAPH] fallback completed title=%r model=%s entities=%d relationships=%d", title, model, len(extraction["entities"]), len(extraction["relationships"]),)
    return extraction