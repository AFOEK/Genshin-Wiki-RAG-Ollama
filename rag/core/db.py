import sqlite3
import logging
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=10000;
PRAGMA temp_store=MEMORY;
PRAGMA cache_size=-65000;
PRAGMA mmap_size=2500000000;

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

CREATE INDEX IF NOT EXISTS idx_docs_source ON docs(source);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_docs_source_raw_hash ON docs(source, raw_hash);
"""

def connect(path: str) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=60.0, isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(SCHEMA)
    log.info(f"Connected to sqlite db at {p}")
    return conn