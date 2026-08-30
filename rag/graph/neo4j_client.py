from __future__ import annotations

import os
import logging

from typing import LiteralString, cast, Any
from neo4j import GraphDatabase

log=logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self, cfg: dict) -> None:
        ncfg=cfg.get("neo4j",{}) or {}
        uri=os.getenv("NEO4J_URI", str(ncfg.get("uri","bolt://localhost:7688")),)
        user=os.getenv("NEO4J_USER", str(ncfg.get("user","neo4j")),)
        self.database=os.getenv("NEO4J_DATABASE", str(ncfg.get("database","neo4j")),)
        password_env=str(ncfg.get("password_env","NEO4J_PASSWORD"))
        password=os.getenv(password_env)

        if not password:
            raise RuntimeError(f"[NEO4J] Password environment variable not set: {password_env}")

        log.info("[NEO4J] Connecting uri=%s user=%s database=%s", uri, user, self.database,)
        self.driver=GraphDatabase.driver(uri, auth=(user,password),)
        self.driver.verify_connectivity()
        log.info("[NEO4J] Connected uri=%s user=%s database=%s", uri, user, self.database,)

    def query(self, cypher: str, **params,) -> list[dict[str,Any]]:
        records,summary,keys=self.driver.execute_query(cast(LiteralString,cypher), parameters_=params, database_=self.database,)
        return [record.data() for record in records]

    def close(self):
        self.driver.close()