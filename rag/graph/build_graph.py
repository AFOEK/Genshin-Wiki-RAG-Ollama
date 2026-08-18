import sqlite3

from .utils import entity_key

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