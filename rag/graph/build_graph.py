import sqlite3
import logging

from .utils import entity_key
from .extractor import extract_graph_from_chunk 

log = logging.getLogger(__name__)

RELATION_MARKERS=(
    "friend",
    "friends",
    "ally",
    "allies",
    "enemy",
    "enemies",
    "brother",
    "sister",
    "mother",
    "father",
    "parent",
    "child",
    "leader",
    "member",
    "affiliation",
    "associated",
    "serves",
    "served",
    "envoy",
    "disciple",
    "master",
    "follower",
    "created by",
    "worship",
    "archon",
    "god",
)

def likely_graph_chunk(text:str)->bool:
    lowered=text.casefold()
    return any(
        marker in lowered
        for marker in RELATION_MARKERS
    )

def iter_genshin_wiki_chunks(conn: sqlite3.Connection):
    cur = conn.execute("""
    SELECT
        d.doc_id,
        d.title,
        d.url,
        d.source,
        c.chunk_id,
        c.text
    FROM docs d
    JOIN chunks c ON c.doc_id = d.doc_id
    WHERE d.source='genshin_wiki'
    AND COALESCE(d.status, 1)=1
    AND c.is_active=1
    ORDER BY d.doc_id, c.chunk_index
    """)

    for row in cur:
        yield dict(row)

def upsert_chunk(client, row: dict) -> None:
    client.query("""
    MERGE (d:Document {doc_id:$doc_id})
    SET d.title=$title,
        d.url=$url,
        d.source=$source

    MERGE (c.Chunk {chunk_id:$chunk_id})
    SET c.doc_id=$doc_id,
        c.source=$source

    MERGE (d)-[:HAS_CHUNK]->(c)
    """,
    doc_id=int(row["doc_id"]),
    chunk_id=int(row["chunk_id"]),
    title=str(row["title"]),
    url=str(row["url"]),
    source=str(row["source"]))

def upsert_entity(client, name: str, entity_type: str = "unknown", aliases: list[str] | None = None) -> None:
    key = entity_key(name)
    if not key:
        return

    client.query("""
        MERGE (r:Entity {key:$key})
        ON CREATE SET
            e.name=$name,
            e.type=$entity_type,
            e.aliases=$aliases
        ON MATCH SET
            e.name=coalesce(e.name, $name),
            e.type=CASE
                WHEN e.type IS NULL OR e.type='unknown'
                THEN $entity_type
                ELSE e.type
            ENF
    """,
    key=key, name=name, entity_type=entity_type, aliases=aliases)

def link_entity_to_chunk(client, name: str, chunk_id: int) -> None:
    key = entity_key(name=name)
    if not key:
        return

    client.query("""
        MATCH (e:Entity {key:$key})
        MATCH (c:Chunk {chunk_id:$chunk_id})
        MATCH (e)-[:MENTIONED_IN]->(c)
    """, key=key, chunk_id=int(chunk_id))

def upsert_relationship(client, source: str, target: str, relation_type: str, chunk_id: int, confidence: float=1.0) -> None:
    source_key = entity_key(source)
    target_key = entity_key(target)
    if not source_key or not target_key:
        return
    if source_key == target_key:
        return

    relation_type = relation_type.strip().upper().replace(" ", "_")
    if not relation_type:
        relation_type = "RELATED_TO"

    client.query("""
        MATCH (a:Entity {key:$source_key})
        MATCH (b:Entity {key:$target_key})

        MERGE (a)-[r:RELATION {
            relation_type:$relation_type,
            evidence_chunk_id:$chunk_id
        }]->(b)

        SET r.confidence=$confidence,
            r.source='genshin_wiki'
    """,
    source_key=source_key,
    target_key=target_key,
    relation_type=relation_type,
    chunk_id=int(chunk_id),
    confidence=float(confidence))


def process_graph_chunk(cfg: dict, client, row: dict) -> dict:
    upsert_chunk(client, row)
    extraction = extract_graph_from_chunk(cfg, title=str(row["title"]), text=str(row["text"]))
    chunk_id = int(row["chunk_id"])

    for entity in extraction["entities"]:
        upsert_entity(client, name=entity["name"], entity_type=entity["type"], aliases=entity["aliases"])
        link_entity_to_chunk(client, entity["name"], chunk_id)

    for relationship in extraction["relationships"]:
        upsert_relationship(client, source=relationship["source"], target=relationship["target"], relation_type=relationship["type"], chunk_id=chunk_id, confidence=relationship["confidence"])

def build_graph(cfg: dict, conn, client, limit: int | None = None) -> None:
    processed=0
    entity_count=0
    relationship_count=0

    for row in iter_genshin_wiki_chunks(conn):
        try:
            if not likely_graph_chunk(str(row["text"])):
                continue
            
            extraction=process_graph_chunk(cfg, client, row)
        except Exception:
            log.exception("[GRAPH] failed chunk_id=%s title=%r", row["chunk_id"], row["title"])
            continue

        processed += 1
        entity_count += len(extraction["entities"])
        relationship_count += len(extraction["relationships"])

        log.info("[GRAPH] chunk=%s title=%r entities=%d relationships=%d", row["chunk_id"], row["title"], len(extraction["entities"]), len(extraction["relationships"]),)

        if limit is not None and processed>=limit:
            break

    log.info("[GRAPH] done chunks=%d entities=%d relationships=%d", processed, entity_count, relationship_count)