from __future__ import annotations
from .neo4j_client import Neo4jClient

SCHEMA_QUERIES = [
    """
    CREATE CONSTRAINT entity_key IF NOT EXISTS
    FOR (e:Entity)
    REQUIRE e.key IS UNIQUE
    """,
    """
    CREATE CONSTRAINT document_id IF NOT EXISTS
    FOR (d:Document)
    REQUIRE d.doc_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT chunk_id IF NOT EXISTS
    FOR (c:Chunk)
    REQUIRE c.chunk_id IS UNIQUE
    """,
]

def ensure_schema(client: Neo4jClient) -> None:
    for query in SCHEMA_QUERIES:
        client.query(query)