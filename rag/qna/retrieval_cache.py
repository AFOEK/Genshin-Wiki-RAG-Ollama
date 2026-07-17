from __future__ import annotations

import json
import logging
import sqlite3
import time
import threading

from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

QUERY = """
PRAGMA journal_mode=WAL;
PRAGMA wal_autocheckpoint=2000;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS retrieval_cache (
    cache_key TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_retrieval_cache_expires ON retrieval_cache(expires_at);
"""

class RetrievalCache:
    def __init__(self, path: Path, *, ttl_seconds: int = 86400, max_entries: int = 50000):
        self.path = Path(path)
        self.ttl_seconds = int(ttl_seconds)
        self.max_entries = int(max_entries)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), timeout=60.0, check_same_thread=False)
        with self._lock:
            self.conn.executescript(QUERY)
            self.conn.commit()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            row = self.conn.execute(
                """
                SELECT payload, expires_at
                FROM retrieval_cache
                WHERE cache_key = ?
                """, (cache_key,)).fetchone()

            if row is None:
                return None

            payload, expires_at = row

            if float(expires_at) < now:
                self.conn.execute(
                    """
                    DELETE FROM retrieval_cache
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                )
                self.conn.commit()
                return None

            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                self.conn.execute(
                    """
                    DELETE FROM retrieval_cache
                    WHERE cache_key = ?
                    """,
                    (cache_key,))
                self.conn.commit()
                return None
        
    def set(self, cache_key: str, payload: dict[str, Any],) -> None:
        now = time.time()
        expires_at = now + self.ttl_seconds
        encoded_payload = json.dumps(payload, ensure_ascii=False)

        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO retrieval_cache(
                    cache_key,
                    created_at,
                    expires_at,
                    payload
                )
                VALUES (?, ?, ?, ?)
                """,
                (cache_key, now, expires_at, encoded_payload,))
            self._prune_locked(now)
            self.conn.commit()

    def _prune_locked(self, now: float) -> None:
        self.conn.execute(
            """
            DELETE FROM retrieval_cache
            WHERE expires_at < ?
            """, (now,))
        self.conn.execute(
            """
            DELETE FROM retrieval_cache
            WHERE cache_key IN (
                SELECT cache_key
                FROM retrieval_cache
                ORDER BY created_at ASC
                LIMIT MAX(
                    (
                        SELECT COUNT(*)
                        FROM retrieval_cache
                    ) - ?,
                    0
                )
            )
            """, (self.max_entries,))


    def prune(self) -> None:
        with self._lock:
            self._prune_locked(time.time())
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()
