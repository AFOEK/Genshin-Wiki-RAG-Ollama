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
      "type": "{entity_types}",
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
- Do not infer a relationship merely because it seems plausible.
- Both source and target must appear in the entities list.
- Use canonical entity names when possible.
- Do not create new relationship types.
- The relationship "type" MUST be exactly one of the allowed relation types.
- Allowed relationship types: {relation_types}
- Prefer the most specific supported relationship type instead of ASSOCIATED_WITH or RELATED_TO.
- Use ASSOCIATED_WITH or RELATED_TO only when no more specific allowed type accurately represents the explicit relationship.
- Do not output duplicate relationships between the same source, target, and type.
- Do not output both a relation and its inverse unless explicitly necessary.
- Confidence must be between 0.0 and 1.0 and should reflect how directly the supplied text supports the relationship.
- Relationship direction is significant: source TYPE target must read naturally.
- When inverse relation types exist, emit only the direction most directly expressed by the text.
- Do not emit both A CREATED B and B CREATED_BY A for the same fact.

Semantic disambiguation:
- PRECEDES and FOLLOWS are chronological relations only. FOLLOWS means occurs after, not "is a follower of".
- MENTOR_OF, STUDENT_OF, and DISCIPLE_OF describe teaching relationships.
- MASTER_OF and SLAVE_OF describe explicit master/slave or master/servant relationships, not teaching relationships.
- PARTNER_OF means an explicitly stated partnership. Do not use it automatically for friendship or romance.
- LOVER_OF means an explicitly stated romantic relationship.
- MEMBER_OF means formal membership in a faction or organization.
- WORKS_FOR and EMPLOYED_BY mean employment or work relationships.
- SERVES means explicit service or allegiance and does not automatically imply employment.
- AFFILIATED_WITH is weaker than MEMBER_OF and should be used only when formal membership is not stated.
- SUBORDINATE_OF describes an explicit hierarchical command relationship.
- LOCATED_IN means one entity is geographically situated inside another location.
- PART_OF means structural, administrative, organizational, or compositional inclusion.
- CONTAINS is the inverse structural relation of PART_OF.
- RESIDES_IN and LIVES_IN describe residence, not birthplace or origin.
- ORIGINATES_FROM describes provenance or origin and should not be used merely because an entity currently lives somewhere.
- BORN_IN and DIED_IN apply only when birth or death location is explicitly stated.
- CREATED/CREATED_BY describe general creation.
- INVENTED/INVENTED_BY describe invention.
- FORGED/FORGED_BY describe forging.
- BUILT/BUILT_BY describe construction.
- Do not substitute these creation relations for one another when the text provides a more specific description.
- OWNS/OWNED_BY describe ownership.
- WIELDS/WIELDED_BY describe active weapon use.
- CARRIES/CARRIED_BY describe possession or carrying without necessarily implying ownership or weapon use.
- USES/USED_BY describe explicit functional use.
- HAS_VISION, HAS_DELUSION, and HAS_GNOSIS apply only when possession is explicitly supported.
- USES_ELEMENT means an entity actively uses an element.
- ASSOCIATED_WITH_ELEMENT is for an explicit elemental association that is not necessarily active use.
- RULES means governing authority over a place or people.
- GOVERNS/GOVERNED_BY describes formal governance.
- PROTECTS means explicit protection or guardianship, not simply being friendly or allied.
- FOUGHT means participation in combat against another entity.
- DEFEATED means explicit victory.
- KILLED requires explicit causation of death.
- OPPOSES describes explicit opposition and does not necessarily imply combat.
- APPEARS_IN and FEATURED_IN describe presence in a quest, event, work, or story.
- PARTICIPATED_IN and INVOLVED_IN require actual involvement, not merely being mentioned.
- TAKES_PLACE_IN and OCCURRED_IN describe event or quest location.
- OCCURRED_DURING describes temporal containment within another event or period.
- CAUSES and CAUSED_BY require an explicitly supported causal relationship, not mere temporal sequence.
- RESULTED_IN requires the text to describe the result as a consequence.
- KNOWN_AS is for a stated alternate name or title only when representing it as a relationship is useful; ordinary alternate names should preferably go in aliases.
- IDENTITY_OF means two named entities are explicitly revealed or stated to be the same identity.
- FORM_OF means one entity is a distinct form or manifestation of another.
- INCARNATION_OF requires an explicit incarnation or embodiment relationship.
- If there are no supported relationships, return an empty "relationships" list.
- If there are no supported entities, return empty "entities" and "relationships" lists.
- Do not include Markdown fences or explanatory text.
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
        if relation_type not in ALLOWED_RELATION_TYPES:
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

def extract_graph_from_chunk(cfg: dict[str, Any], *, title: str, text: str) -> dict[str, Any]:
    ncfg = cfg.get("neo4j", {}) or {}
    num_predict = int(ncfg.get("extraction_num_predict", 2048))
    num_ctx = int(ncfg.get("extraction_num_ctx",4096))
    model = str(ncfg.get("extraction_model", "qwen3.6:27b"))
    temperature = float(ncfg.get("extraction_temperature", 0.0))
    min_confidence = float(ncfg.get("min_relation_confidence", 0.85))
    prompt = build_extraction_prompt(title, text)
    raw = generate(cfg, prompt, model_override=model, options_override={
        "temperature": temperature,
        "num_ctx": num_ctx,
        "num_predict": num_predict
    }, 
    think_override=False).strip()

    if not raw:
        return {
            "entities": [],
            "relationships": []
        }

    try:
        data=parse_extraction_json(raw)
    except (json.JSONDecodeError, ValueError):
        log.warning("[GRAPH] JSON parse failed; retrying with larger output budget title=%r old_num_predict=%d", title, num_predict,)
        retry_predict=min(num_predict * 2, 4096)
        retry_ctx=max(num_ctx, retry_predict + 2048)
        raw=generate(cfg, prompt, model_override=model, options_override={
            "temperature":temperature,
            "num_ctx":retry_ctx,
            "num_predict":retry_predict
        }, think_override=False).strip()
        data=parse_extraction_json(raw)
    return normalize_extraction(data, min_confidence)