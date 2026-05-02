import sqlite3
import logging
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = """
PRAGMA foreign_keys=ON;
PRAGMA journal_mode=WAL;
PRAGMA wal_autocheckpoint=2000;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=100000;
PRAGMA temp_store=MEMORY;
PRAGMA cache_size=-50000;
PRAGMA mmap_size=1500000000;

CREATE TABLE IF NOT EXISTS docs (
    doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    url TEXT UNIQUE,
    title TEXT,
    tier TEXT DEFAULT 'primary',
    weight REAL DEFAULT 1.0,
    status INTEGER DEFAULT 1,
    fetched_at TEXT,
    raw_hash TEXT,
    norm_hash TEXT,

    last_modified TEXT,
    etag TEXT,

    raw_zst BLOB,
    raw_len INTEGER,
    raw_zst_len INTEGER
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL,
    chunk_index INTEGER,
    text TEXT,

    text_zst BLOB,
    text_len INTEGER,
    text_zst_len INTEGER,

    chunk_hash TEXT,
    is_active INTEGER DEFAULT 1,
    UNIQUE(doc_id, chunk_index),

    FOREIGN KEY (doc_id) REFERENCES docs(doc_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER PRIMARY KEY,
    dims INTEGER,
    vector BLOB,

    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS fts_dirty_docs (
        doc_id INTEGER PRIMARY KEY,
        reason TEXT,
        marked_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

CREATE TABLE IF NOT EXISTS chunk_parents (
        parent_id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER NOT NULL,
        parent_index INTEGER NOT NULL,
        start_chunk_index INTEGER NOT NULL,
        end_chunk_index INTEGER NOT NULL,
        is_active INTEGER DEFAULT 1,
        UNIQUE(doc_id, parent_index),

        FOREIGN KEY (doc_id) REFERENCES docs(doc_id)
            ON DELETE CASCADE
            ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS chunk_parent_map (
    chunk_id INTEGER PRIMARY KEY,
    parent_id INTEGER NOT NULL,

    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,

    FOREIGN KEY (parent_id) REFERENCES chunk_parents(parent_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS parent_dirty_docs (
    doc_id INTEGER PRIMARY KEY,
    reason TEXT,
    marked_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_docs_source ON docs(source);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_docs_source_raw_hash ON docs(source, raw_hash);
CREATE INDEX IF NOT EXISTS idx_docs_last_modified ON docs(last_modified);
CREATE INDEX IF NOT EXISTS idx_docs_etag ON docs(etag);
CREATE INDEX IF NOT EXISTS idx_chunks_active ON chunks(doc_id, is_active);
CREATE INDEX IF NOT EXISTS idx_docs_status ON docs(status);
CREATE INDEX IF NOT EXISTS idx_chunk_parents_doc ON chunk_parents(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunk_parent_map_parent ON chunk_parent_map(parent_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_active_parent ON chunks(doc_id, is_active, chunk_index);

CREATE VIEW IF NOT EXISTS v_docs_by_source AS
SELECT
    source,
    COUNT(*) AS doc_count
FROM docs
GROUP BY source;

CREATE VIEW IF NOT EXISTS v_active_chunks_by_source AS
SELECT
    d.source,
    COUNT(*) AS active_chunk_count
FROM chunks c
JOIN docs d ON d.doc_id = c.doc_id
WHERE c.is_active = 1
GROUP BY d.source;

CREATE VIEW IF NOT EXISTS v_embeddings_by_source AS
SELECT
    d.source,
    COUNT(*) AS embedded_active_chunk_count
FROM chunks c
JOIN docs d ON d.doc_id = c.doc_id
JOIN embeddings e ON e.chunk_id = c.chunk_id
WHERE c.is_active = 1
GROUP BY d.source;

CREATE VIEW IF NOT EXISTS v_missing_embeddings_by_source AS
SELECT
    d.source,
    COUNT(*) AS missing_embedding_count
FROM chunks c
JOIN docs d ON d.doc_id = c.doc_id
LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
WHERE c.is_active = 1
  AND e.chunk_id IS NULL
GROUP BY d.source;

CREATE VIEW IF NOT EXISTS v_source_summary AS
SELECT
    d.source,
    COUNT(DISTINCT d.doc_id) AS doc_count,
    COUNT(DISTINCT CASE WHEN c.is_active = 1 THEN c.chunk_id END) AS active_chunk_count,
    COUNT(DISTINCT CASE WHEN c.is_active = 1 AND e.chunk_id IS NOT NULL THEN c.chunk_id END) AS embedded_active_chunk_count,
    COUNT(DISTINCT CASE WHEN c.is_active = 1 AND e.chunk_id IS NULL THEN c.chunk_id END) AS missing_embedding_count,
    COUNT(DISTINCT CASE
    WHEN COALESCE(d.status, 1) = 1
     AND NOT EXISTS (
        SELECT 1
        FROM chunks c2
        WHERE c2.doc_id = d.doc_id
        AND c2.is_active = 1
        )
        THEN d.doc_id
    END) AS docs_with_no_active_chunks
FROM docs d
LEFT JOIN chunks c ON c.doc_id = d.doc_id
LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
GROUP BY d.source;

CREATE VIEW IF NOT EXISTS v_docs_with_missing_embeddings AS
SELECT
    d.source,
    d.doc_id,
    d.title,
    d.url,
    COUNT(c.chunk_id) AS missing_embedding_chunks
FROM docs d
JOIN chunks c ON c.doc_id = d.doc_id
LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
WHERE c.is_active = 1
  AND e.chunk_id IS NULL
GROUP BY d.source, d.doc_id, d.title, d.url;

CREATE VIEW IF NOT EXISTS v_docs_with_no_active_chunks AS
SELECT
    d.source,
    d.doc_id,
    d.title,
    d.url,
    0 AS active_chunk_count
FROM docs d
LEFT JOIN chunks c
    ON c.doc_id = d.doc_id
   AND c.is_active = 1
WHERE c.chunk_id IS NULL
  AND COALESCE(d.status, 1) = 1;

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    doc_id UNINDEXED,
    source UNINDEXED,
    title,
    text,
    tokenize='unicode61'
);
"""

def read_only_connect(path: str) -> sqlite3.Connection:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    log.info(f"[INFO] Read-only sqlite db connected at {p}")
    return conn

def init_db(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(p), timeout=120.0, isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.executescript(SCHEMA)

        log.info(f"[INFO] Initialized sqlite db schema at {p}")
    finally:
        conn.close()

def connect(path: str, init_schema: bool = False) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(p), timeout=60.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")

    if init_schema:
        conn.executescript(SCHEMA)

    log.info(f"[INFO] Connected to sqlite db at {p}")
    return conn