from __future__ import annotations

import random
import sqlite3
import logging
from dataclasses import dataclass
from typing import Optional

from utils.hashing import sha256_text
from utils.codec import zstd_decompress_text

log = logging.getLogger(__name__)

@dataclass
class IntegrityFailure:
    kind: str
    id: str
    url: str
    reason: str
    got_hash: Optional[str] = None
    expected_hash : Optional[str] = None
    error: Optional[str] = None

@dataclass
class IntegrityReport:
    docs_total: int
    docs_checked: int
    docs_ok: int

    chunks_total: int
    chunks_checked: int
    chunks_ok: int

    failures: list[IntegrityFailure]

@dataclass
class CompressionStats:
    docs_rows: int
    docs_avg_raw: Optional[float]
    docs_avg_zst: Optional[float]
    docs_ratio: Optional[float]

    chunks_rows: int
    chunks_avg_raw: Optional[float]
    chunks_avg_zst: Optional[float]
    chunks_ratio: Optional[float]

def sample(rows: list, samples: Optional[int], rng: random.Random) -> list:
    if samples is None:
        return rows
    if samples <= 0:
        return []
    if len(rows) <= samples:
        return rows
    return rng.sample(rows, samples)

def audit_integrity(conn: sqlite3.Connection, sample_docs: Optional[int] = 50, sample_chunks: Optional[int] = 200, seed: int = 1337, active_chunks_only: bool = True) -> IntegrityReport:
    rng = random.Random(seed)
    cur = conn.cursor()
    failures: list[IntegrityFailure] = []

    cur.execute("""
    SELECT doc_id, url, raw_zst, raw_hash FROM docs WHERE raw_zst IS NOT NULL OR raw_hash IS NOT NULL
    """)
    doc_rows = cur.fetchall()
    docs_total = len(doc_rows)

    docs_rows_checked = []
    for doc_id, url, raw_zst, raw_hash in doc_rows:
        if raw_zst in None:
            failures.append(IntegrityFailure("docs", doc_id, url , "missing_blob_doc"))
            continue
        if raw_hash is None:
            failures.append(IntegrityFailure("docs", doc_id, url, "missing_hash_doc"))
            continue
        docs_rows_checked.append((doc_id, url, raw_zst, raw_hash))

    doc_pick = sample(docs_rows_checked, sample_docs, rng)

    docs_ok = 0

    for doc_id, url, raw_zst, raw_hash  in doc_pick:
        try:
            text = zstd_decompress_text(raw_zst)
            got = sha256_text(text)
            if got == raw_hash:
                docs_ok += 1
            else:
                failures.append(IntegrityFailure("docs", doc_id, url, "hash_mismatch_doc", got, raw_hash))

        except Exception as e:
            log.exception(f"Integrity failure detected for {doc_id}, {url} due to decompress error")
            failures.append(IntegrityFailure("docs", doc_id, url, "decompress_error_doc", error=str(e)))

    docs_checked = len(doc_pick)

    chunk_where = "c.text_zst IS NOT NULL OR c.chunk_hash IS NOT NULL"
    if active_chunks_only:
        chunk_where += " AND c.is_active=1"

    cur.execute(f"""
    SELECT c.chunk_id, d.url, c.text_zst, c.chunk_hash
    FROM chunks c
    JOIN docs d ON d.doc_id = c.doc_id
    WHERE {chunk_where}
    """)
    chunk_rows = cur.fetchall()
    chunks_total = len(chunk_rows)

    chunk_rows_checked = []
    for chunk_id, url, text_zst, chunk_hash in chunk_rows:
        if text_zst is None:
            failures.append(IntegrityFailure("chunks", chunk_id, url, "missing_blob_chunks"))
            continue
        if chunk_hash is None:
            failures.append(IntegrityFailure("chunks", chunk_id, url, "missing_hash_chunk"))
            continue
        chunk_rows_checked.append((chunk_id, url, text_zst, chunk_hash))
    
    chunk_picks = sample(chunk_rows_checked, sample_chunks, rng)
    chunks_ok = 0

    for chunk_id, url, text_zst, chunk_hash in chunk_picks:
        try:
            text = zstd_decompress_text(text_zst)
            got = sha256_text(text)
            if got == chunk_hash:
                chunks_ok += 1
            else:
                failures.append(IntegrityFailure("chunks", chunk_id, url, "hash_mismatch_chunk", got, chunk_hash))
        except Exception as e:
            log.exception(f"Integrity failure detected for {chunk_id}, {url} due to decompress error")
            failures.append(IntegrityFailure("chunks", chunk_id, url, "decompress_error_chunk", error=str(e)))
    
    chunk_checked = len(chunk_picks)

    return IntegrityReport(
        docs_total=docs_total,
        docs_checked=docs_checked,
        docs_ok=docs_ok,
        chunks_total=chunks_total,
        chunks_checked=chunk_checked,
        chunks_ok=chunks_ok,
        failures=failures
    )

def compression_stats(conn: sqlite3.Connection, active_chunks_only: bool = True) -> CompressionStats:
    cur = conn.cursor()
    cur.execute("""
    SELECT COUNT(*), AVG(raw_len), AVG(raw_zst_len)
    FROM docs
    WHERE raw_zst IS NOT NULL AND raw_len IS NOT NULL AND raw_zst_len IS NOT NULL
    """)
    d_rows, d_avg_raw, d_avg_zst = cur.fetchone()

    d_ratio = None
    if d_rows and d_avg_raw and d_avg_zst and d_avg_raw > 0:
        d_ratio = float(d_avg_zst) / float(d_avg_raw)

    chunk_where = "text_zst IS NOT NULL AND text_len IS NOT NULL and text_zst_len IS NOT NULL"
    if active_chunks_only:
        chunk_where += " AND is_active=1"
    
    cur.execute(f"""
    SELECT COUNT(*), AVG(text_len), AVG(text_zst_len)
    FROM chunks
    WHERE {chunk_where}
    """)
    c_rows, c_avg_raw, c_avg_zst = cur.fetchone()

    c_ratio = None
    if c_rows and c_avg_raw and c_avg_zst and c_avg_raw > 0:
        c_ratio = float(c_avg_zst) / float(c_avg_raw)

    return CompressionStats(
        docs_rows=int(d_rows or 0),
        docs_avg_raw=float(d_avg_raw) if d_avg_raw is not None else None,
        docs_avg_zst=float(d_avg_zst) if d_avg_zst is not None else None,
        docs_ratio=d_ratio,

        chunks_rows=int(c_rows or 0),
        chunks_avg_raw=float(c_avg_raw) if c_avg_raw is not None else None,
        chunks_avg_zst=float(c_avg_zst) if c_avg_zst is not None else None,
        chunks_ratio=c_ratio,
    )