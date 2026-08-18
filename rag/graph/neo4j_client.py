from __future__ import annotations
import os
import logging
from typing import LiteralString, cast, Any
from neo4j import GraphDatabase

log = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self, cfg: dict) -> None:
        ncfg = cfg.get("neo4j", {}) or {}
        self.database = str(ncfg.get("database", "neo4j"))
        password_env = str(ncfg.get("password_env", "NEO4J_PASSWORD"))
        password = os.environ.get(password_env)

        if not password:
            raise RuntimeError(f"[NEO4J] Password environment variable not set {password_env}")

        self.driver = GraphDatabase.driver(
            str(ncfg.get("uri", "neo4j://localhost:7687")),
            auth = (str(ncfg.get("user", "neo4j")), password)
        )

        self.driver.verify_connectivity()

    def query(self, cypher: str, **params) -> list[dict[str, Any]]:
        records, summary, keys = self.driver.execute_query(
            cast(LiteralString , cypher), parameters_ = params, database_= self.database
        )

        return [dict(record) for record in records]

    def close(self):
        self.driver.close()