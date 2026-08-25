import sqlite3
import logging
import argparse
import yaml
import hashlib
import json
import inspect
import re

from dotenv import load_dotenv
from pathlib import Path

from .utils_graph import entity_key
from .extractor import extract_graph_from_chunk
from .neo4j_client import Neo4jClient
from .schema import ensure_schema
from core.paths import resolve_db_path
from utils.logging_setup import setup_logging

log = logging.getLogger(__name__)

RELATION_MARKER_GROUPS={
    "family":(
        "father", "mother", "parent", "son", "daughter",
        "child", "brother", "sister", "sibling", "husband",
        "wife", "spouse", "ancestor", "descendant", "twin",
    ),
    "social":(
        "friend", "friends", "companion", "ally", "allies",
        "partner", "rival", "enemy", "enemies",
    ),
    "mentorship":(
        "master", "mentor", "teacher", "student",
        "disciple", "apprentice", "follower",
    ),
    "affiliation":(
        "affiliation", "affiliated with", "member", "member of",
        "belongs to", "joined", "leader", "founder", "founded",
        "commander", "captain", "general", "chief", "director",
        "boss"
    ),
    "service":(
        "serves", "served", "serves under", "served under",
        "works for", "worked for", "subordinate", "envoy",
        "agent", "representative", "retainer",
    ),
    "creation":(
        "created", "created by", "creator", "made by",
        "forged by", "invented", "inventor", "owner",
        "owned by", "wielder", "wielded by",
    ),
    "location":(
        "located in", "located at", "resides in", "lives in",
        "native of", "originated in", "rules", "ruled",
        "governs", "governed", "protects", "guardian",
    ),
    "conflict":(
        "fought", "fought against", "battle", "defeated",
        "defeated by", "killed", "killed by", "opposed",
        "opposes", "betrayed",
    ),
    "divine":(
        "god", "goddess", "deity", "archon", "worship",
        "worships", "worshipped", "divine", "celestia",
    ),
    "association":(
        "associated with", "connected to", "related to",
        "relationship", "successor", "predecessor",
        "succeeded", "replaced",
    ),
    "event":(
        "participated in", "involved in", "appeared in",
        "takes part in", "during the",
    ),
}

RELATION_MARKERS=tuple(marker for markers in RELATION_MARKER_GROUPS.values() for marker in markers)
RELATION_PATTERN=re.compile(r"\b(?:" + "|".join(re.escape(marker) for marker in sorted(RELATION_MARKERS, key=len, reverse=True)) + r")\b", flags=re.IGNORECASE)

def graph_source_hash(row:dict)->str:
    payload={
        "title":str(row.get("title") or ""),
        "url":str(row.get("url") or ""),
        "text":str(row.get("text") or "")}
    raw=json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",",":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def file_sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def graph_extractor_signature(cfg: dict) -> str:
    ncfg=cfg.get("neo4j",{}) or {}
    graph_dir=Path(__file__).resolve().parent

    payload={
        "extractor_sha256":file_sha256(graph_dir/"extractor.py"),
        "utils_sha256":file_sha256(graph_dir/"utils_graph.py"),
        "schema_sha256":file_sha256(graph_dir/"schema.py"),
        "prepare_extraction_sha256":hashlib.sha256(
            inspect.getsource(prepare_graph_extraction).encode("utf-8")
        ).hexdigest(),
        "relation_markers":RELATION_MARKERS,
        "model":str(ncfg.get("extraction_model", "qwen3.6:27b")),
        "temperature":float(ncfg.get("extraction_temperature",0.0)),
        "source":str(ncfg.get("source", "genshin_wiki")),
    }

    raw=json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",",":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def batched_rows(rows, batch_size: int):
    batch=[]
    for row in rows:
        batch.append(row)
        if len(batch)>=batch_size:
            yield batch
            batch=[]

    if batch:
        yield batch

def get_graph_chunk_states(client, chunk_ids: list[int]) -> dict[int, dict]:
    if not chunk_ids:
        return {}

    rows=client.query("""
        UNWIND $chunk_ids AS chunk_id

        MATCH (c:Chunk {chunk_id:chunk_id})

        RETURN
            c.chunk_id AS chunk_id,
            c.content_hash AS content_hash,
            c.extractor_signature AS extractor_signature
    """,
        chunk_ids=[int(chunk_id) for chunk_id in chunk_ids],)

    return {
        int(row["chunk_id"]):{
            "content_hash":row.get("content_hash"),
            "extractor_signature":row.get("extractor_signature")} for row in rows}

def prepare_graph_extraction(extraction: dict) -> tuple[list[dict], list[dict]]:
    entities_by_key={}

    for entity in extraction.get("entities",[]):
        name=str(entity.get("name") or "").strip()
        key=entity_key(name)

        if not key:
            continue

        entity_type=str(entity.get("type") or "unknown").strip().lower()
        aliases=[
            str(alias).strip()
            for alias in entity.get("aliases", [])
            if str(alias).strip()
        ]

        existing=entities_by_key.get(key)

        if existing is None:
            entities_by_key[key]={
                "key":key,
                "name":name,
                "type":entity_type,
                "aliases":aliases,
            }
        elif existing["type"]=="unknown" and entity_type!="unknown":
            existing["type"]=entity_type

    relationships=[]

    for relationship in extraction.get("relationships",[]):
        source=str(relationship.get("source") or "").strip()
        target=str(relationship.get("target") or "").strip()
        source_key=entity_key(source)
        target_key=entity_key(target)

        if not source_key or not target_key:
            continue
        if source_key==target_key:
            continue

        relation_type=str(relationship.get("type") or "RELATED_TO").strip().upper().replace(" ", "_")

        try:
            confidence=float(relationship.get("confidence",0.0))
        except (TypeError,ValueError):
            confidence=0.0

        relationships.append({
            "source_key":source_key,
            "target_key":target_key,
            "relation_type":relation_type,
            "confidence":max(0.0,min(1.0,confidence)),
        })

    return list(entities_by_key.values()),relationships

def relation_marker_hits(text: str) -> set[str]:
    lowered=text.casefold()
    hits=set()
    for group,markers in RELATION_MARKER_GROUPS.items():
        if any(re.search(rf"\b{re.escape(marker)}\b", lowered) for marker in markers):
            hits.add(group)

    return hits

def likely_graph_chunk(text: str) -> bool:
    return bool(relation_marker_hits(text))

def replace_graph_chunk(client, row:dict, extraction:dict, *, content_hash:str, extractor_signature:str) -> None:
    entities, relationships=prepare_graph_extraction(extraction)

    client.query("""
        MERGE (d:Document {doc_id:$doc_id})
        SET d.title=$title,
            d.url=$url,
            d.source=$source

        MERGE (c:Chunk {chunk_id:$chunk_id})
        SET c.doc_id=$doc_id,
            c.source=$source

        MERGE (d)-[:HAS_CHUNK]->(c)

        WITH c

        CALL (c) {
            MATCH (:Entity)-[m:MENTIONED_IN]->(c)
            DELETE m
        }

        CALL () {
            MATCH ()-[r:RELATION]->()
            WHERE r.evidence_chunk_id=$chunk_id
            DELETE r
        }

        CALL (c) {
            UNWIND $entities AS entity

            MERGE (e:Entity {key:entity.key})

            ON CREATE SET
                e.name=entity.name,
                e.type=entity.type,
                e.aliases=entity.aliases

            ON MATCH SET
                e.name=coalesce(e.name,entity.name),
                e.type=CASE
                    WHEN e.type IS NULL OR e.type='unknown'
                    THEN entity.type
                    ELSE e.type
                END

            MERGE (e)-[:MENTIONED_IN]->(c)
        }

        CALL () {
            UNWIND $relationships AS relation

            MATCH (a:Entity {key:relation.source_key})
            MATCH (b:Entity {key:relation.target_key})

            MERGE (a)-[r:RELATION {
                relation_type:relation.relation_type,
                evidence_chunk_id:$chunk_id
            }]->(b)

            SET r.confidence=relation.confidence,
                r.source=$source
        }

        SET c.content_hash=$content_hash,
            c.extractor_signature=$extractor_signature,
            c.graph_updated_at=datetime()
    """,
        doc_id=int(row["doc_id"]),
        chunk_id=int(row["chunk_id"]),
        title=str(row["title"]),
        url=str(row["url"]),
        source=str(row["source"]),
        entities=entities,
        relationships=relationships,
        content_hash=content_hash,
        extractor_signature=extractor_signature)

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

    MERGE (c:Chunk {chunk_id:$chunk_id})
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
        MERGE (e:Entity {key:$key})
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
            END,
            e.aliases=CASE
                WHEN e.aliases IS NULL
                THEN $aliases
                ELSE e.aliases
            END
    """,
    key=key, name=name, entity_type=entity_type, aliases=aliases)

def link_entity_to_chunk(client, name: str, chunk_id: int) -> None:
    key = entity_key(name=name)
    if not key:
        return

    client.query("""
        MATCH (e:Entity {key:$key})
        MATCH (c:Chunk {chunk_id:$chunk_id})
        MERGE (e)-[:MENTIONED_IN]->(c)
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


def process_graph_chunk(cfg: dict, client, row: dict, *, content_hash: str, extractor_signature: str) -> dict:
        extraction=extract_graph_from_chunk(cfg, title=str(row["title"]), text=str(row["text"]),)
        replace_graph_chunk(client, row, extraction, content_hash=content_hash, extractor_signature=extractor_signature,)
        return extraction

def delete_graph_chunks(client, chunk_ids:list[int]) -> None:
    if not chunk_ids:
        return

    client.query("""
        UNWIND $chunk_ids AS chunk_id

        CALL (chunk_id) {
            MATCH ()-[r:RELATION]->()
            WHERE r.evidence_chunk_id=chunk_id
            DELETE r
        }

        WITH DISTINCT chunk_id

        MATCH (c:Chunk {chunk_id:chunk_id})
        DETACH DELETE c
    """,
        chunk_ids=[int(chunk_id) for chunk_id in chunk_ids])

def ensure_pipeline_meta_table(conn:sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_meta (
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(namespace,key)
        )
    """)
    conn.commit()

def get_pipeline_meta(conn: sqlite3.Connection, namespace: str, key: str) -> str | None:
    row=conn.execute("""
        SELECT value
        FROM pipeline_meta
        WHERE namespace=? AND key=?
    """,(namespace, key)).fetchone()

    if row is None:
        return None
    return str(row["value"])

def set_pipeline_meta(conn: sqlite3.Connection, namespace: str, key: str, value: str) -> None:
    conn.execute("""
        INSERT INTO pipeline_meta(
            namespace,key,value,updated_at
        )
        VALUES(
            ?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now')
        )
        ON CONFLICT(namespace,key) DO UPDATE SET
            value=excluded.value,
            updated_at=excluded.updated_at
    """,(namespace, key, value))
    conn.commit()

def build_graph(cfg: dict, conn, client, limit: int | None = None, force: bool = False, prune: bool = True) -> None:
    ensure_pipeline_meta_table(conn)
    ncfg=cfg.get("neo4j",{}) or {}
    batch_size=int(ncfg.get("sync_batch_size",500))
    extractor_sig=graph_extractor_signature(cfg)
    previous_sig=get_pipeline_meta(conn, "neo4j", "extractor_signature")
    if previous_sig is None:
        log.info("[GRAPH] initial extractor signature=%s", extractor_sig[:12])
    elif previous_sig!=extractor_sig:
        log.warning("[GRAPH] extractor changed old=%s new=%s", previous_sig[:12], extractor_sig[:12])
    else:
        log.info("[GRAPH] extractor unchanged signature=%s", extractor_sig[:12])

    set_pipeline_meta(conn, "neo4j", "sync_status", "running",)

    scanned=0
    processed=0
    unchanged=0
    ineligible=0
    removed_ineligible=0
    failed=0
    stop=False

    for batch in batched_rows(iter_genshin_wiki_chunks(conn), batch_size):
        chunk_ids=[int(row["chunk_id"]) for row in batch]
        existing_states=get_graph_chunk_states(client, chunk_ids)
        for row in batch:
            scanned+=1
            chunk_id=int(row["chunk_id"])
            text=str(row["text"])
            content_hash=graph_source_hash(row)
            state=existing_states.get(chunk_id)
            hits=relation_marker_hits(text)
            log.info("[GRAPH] chunk=%s relation_markers=%s", chunk_id, sorted(hits))
            if not hits:
                ineligible+=1
                if state is not None:
                    delete_graph_chunks(client, [chunk_id])
                    removed_ineligible+=1
                continue

            if (
                not force
                and state is not None
                and state.get("content_hash") == content_hash
                and state.get("extractor_signature") == extractor_sig
            ):
                unchanged+=1
                continue

            try:
                extraction=process_graph_chunk(cfg, client, row, content_hash=content_hash, extractor_signature=extractor_sig)
            except Exception:
                failed+=1
                log.exception("[GRAPH] failed chunk_id=%s title=%r", chunk_id, row["title"])
                continue

            processed+=1
            log.info("[GRAPH] updated chunk=%s title=%r entities=%d relationships=%d", chunk_id, row["title"], len(extraction["entities"]), len(extraction["relationships"]))
            if limit is not None and processed >= limit:
                stop=True
                break

        if stop:
            break

    stale_removed=0
    if prune and limit is None:
        stale_removed=prune_stale_graph_chunks(conn, client, batch_size=batch_size)
        prune_graph_orphans(client)
    elif prune and limit is not None:
        log.info("[GRAPH] stale pruning skipped because this is a limited test run")

    if limit is None and failed==0:
        set_pipeline_meta(conn, "neo4j", "extractor_signature", extractor_sig)
        set_pipeline_meta(conn, "neo4j", "extraction_model", str(ncfg.get("extraction_model", "qwen3.6:27b")))
        set_pipeline_meta(conn, "neo4j", "sync_status", "success")
        log.info("[GRAPH] full synchronization committed signature=%s", extractor_sig[:12])
    elif limit is not None:
        set_pipeline_meta(conn, "neo4j", "sync_status", "limited")
        log.info("[GRAPH] limited run; global extractor signature not committed")
    else:
        set_pipeline_meta(conn, "neo4j", "sync_status", "partial")
        log.warning("[GRAPH] synchronization incomplete failures=%d", failed)

    log.info("[GRAPH] done scanned=%d updated=%d unchanged=%d ineligible=%d removed_ineligible=%d stale_removed=%d failed=%d", scanned, processed, unchanged, ineligible, removed_ineligible, stale_removed, failed)

def sqlite_active_chunk_ids(conn:sqlite3.Connection, chunk_ids:list[int]) -> set[int]:
    if not chunk_ids:
        return set()

    placeholders=",".join("?" for _ in chunk_ids)
    cur=conn.execute(
        f"""
        SELECT c.chunk_id
        FROM chunks c
        JOIN docs d ON d.doc_id=c.doc_id
        WHERE c.chunk_id IN ({placeholders})
          AND d.source='genshin_wiki'
          AND COALESCE(d.status,1)=1
          AND c.is_active=1
        """,
        chunk_ids)

    return {int(row["chunk_id"]) for row in cur}

def iter_neo4j_chunk_id_batches(client, batch_size: int=1000):
    after_chunk_id=-1
    while True:
        rows = client.query("""
            MATCH (c:Chunk)
            WHERE c.source='genshin_wiki'
              AND c.chunk_id>$after_chunk_id

            RETURN c.chunk_id AS chunk_id
            ORDER BY c.chunk_id
            LIMIT $limit
        """, after_chunk_id=int(after_chunk_id), limit=int(batch_size))

        if not rows:
            break

        chunk_ids=[int(row["chunk_id"]) for row in rows]
        yield chunk_ids
        after_chunk_id=chunk_ids[-1]

def prune_stale_graph_chunks(conn: sqlite3.Connection, client, *, batch_size: int=1000) -> int:
    removed=0
    for graph_chunk_ids in iter_neo4j_chunk_id_batches(client, batch_size):
        active_ids=sqlite_active_chunk_ids(conn, graph_chunk_ids)
        stale_ids=[chunk_id for chunk_id in graph_chunk_ids if chunk_id not in active_ids]
        if not stale_ids:
            continue

        delete_graph_chunks(client, stale_ids,)
        removed+=len(stale_ids)
        log.info("[GRAPH] pruned stale chunks=%d", len(stale_ids),)

    return removed

def prune_graph_orphans(client) -> None:
    client.query("""
        MATCH (d:Document)
        WHERE d.source='genshin_wiki'
          AND NOT EXISTS {
              MATCH (d)-[:HAS_CHUNK]->(:Chunk)
          }
        DELETE d
    """)

    client.query("""
        MATCH (e:Entity)
        WHERE NOT EXISTS {
            MATCH (e)-[:MENTIONED_IN]->(:Chunk)
        }
        AND NOT EXISTS {
            MATCH (e)-[:RELATION]-(:Entity)
        }
        DELETE e
    """)