from __future__ import annotations

import random
import sqlite3
import logging
import json
import faiss
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from utils.hashing import sha256_text
from utils.codec import zstd_decompress_text
from core.paths import resolve_db_path, resolve_faiss_dir
from core.db import read_only_connect

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

    orphan_chunks: int = 0
    orphan_embeddings: int = 0
    missing_embeddings: int = 0
    docs_missing_chunks: int = 0

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

@dataclass
class FaissAuditReport:
    index_total: int
    ids_total: int
    sqlite_active_embeds: int
    dims: int
    failures: list[str]

def sample(rows: list, samples: Optional[int], rng: random.Random) -> list:
    if samples is None:
        return rows
    if samples <= 0:
        return []
    if len(rows) <= samples:
        return rows
    return rng.sample(rows, samples)

def audit_integrity(conn: sqlite3.Connection, sample_docs: Optional[int] = 50, sample_chunks: Optional[int] = 200, seed: int = 1337, active_chunks_only: bool = True, check_orphans: bool = True, check_missing_embeddings: bool = True, max_orphan_failures: int = 200, max_missing_embedding_failures: int = 200) -> IntegrityReport:
    rng = random.Random(seed)
    cur = conn.cursor()
    failures: list[IntegrityFailure] = []
    orphan_chunks = 0
    orphan_embeddings = 0
    missing_embeddings = 0
    docs_missing_chunks = 0

    cur.execute("""
    SELECT doc_id, url, raw_zst, raw_hash FROM docs WHERE raw_zst IS NOT NULL OR raw_hash IS NOT NULL
    """)
    doc_rows = cur.fetchall()
    docs_total = len(doc_rows)

    docs_rows_checked = []
    for doc_id, url, raw_zst, raw_hash in doc_rows:
        if raw_zst is None:
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

    if check_orphans:
        cur.execute("""
        SELECT c.chunk_id, c.doc_id
        FROM chunks c
        LEFT JOIN docs d ON d.doc_id = c.doc_id
        WHERE d.doc_id IS NULL
        """)
        rows = cur.fetchall()
        orphan_chunks = len(rows)
        for chunk_id, doc_id in rows[:max_orphan_failures]:
            failures.append(
                IntegrityFailure(
                    kind="orphan",
                    id=str(chunk_id),
                    url="",
                    reason=f"Orphaned chunk: doc_id={doc_id}"
                )
            )

        cur.execute("""
        SELECT e.chunk_id
        FROM embeddings e
        LEFT JOIN chunks c ON c.chunk_id = e.chunk_id
        WHERE c.chunk_id IS NULL
        """)
        rows = cur.fetchall()
        orphan_embeddings = len(rows)
        for (chunk_id,) in rows[:max_orphan_failures]:
            failures.append(
                IntegrityFailure(
                    kind="orphan",
                    id=str(chunk_id),
                    url="",
                    reason="orphan_embedding: chunk_id missing"
                )
            )

    if check_missing_embeddings:
        cur.execute("""
        SELECT d.doc_id, d.url
        FROM docs d
        LEFT JOIN chunks c ON c.doc_id = d.doc_id AND c.is_active=1
        WHERE d.status = 1
        GROUP BY d.doc_id
        HAVING COUNT(c.chunk_id)=0
        """)
        rows = cur.fetchall()
        docs_missing_chunks = len(rows)
        for doc_id, url in rows[:max_orphan_failures]:
            failures.append(
                IntegrityFailure(
                    kind="docs",
                    id=str(doc_id),
                    url=url,
                    reason="doc_has_no_active_chunks"
                )
            )
        cur.execute("""
        SELECT c.chunk_id, d.url
        FROM chunks c
        JOIN docs d ON d.doc_id = c.doc_id
        LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
        WHERE c.is_active=1 AND e.chunk_id IS NULL
        """)
        rows = cur.fetchall()
        missing_embeddings = len(rows)
        for chunk_id, url in rows[:max_missing_embedding_failures]:
            failures.append(
                IntegrityFailure(
                    kind="chunks",
                    id=str(chunk_id),
                    url=url,
                    reason="active_chunk_missing_embedding"
                )
            )

    return IntegrityReport(
        docs_total=docs_total,
        docs_checked=docs_checked,
        docs_ok=docs_ok,
        chunks_total=chunks_total,
        chunks_checked=chunk_checked,
        chunks_ok=chunks_ok,
        failures=failures,
        orphan_chunks=orphan_chunks,
        orphan_embeddings=orphan_embeddings,
        missing_embeddings=missing_embeddings,
        docs_missing_chunks=docs_missing_chunks,
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

def audit_faiss_against_sqlite(cfg: dict, *, index_dir: str | None = None, sample_self_test: int = 200,) -> FaissAuditReport:
    failures: list[str] = []

    db_path = resolve_db_path(cfg)
    faiss_root = resolve_faiss_dir(cfg)
    current = Path(index_dir) if index_dir else (faiss_root / "current")

    index_path = current / "index.faiss"
    ids_path = current / "ids.npy"
    meta_path = current / "meta.json"

    for p in (index_path, ids_path, meta_path):
        if not p.exists():
            failures.append(f"missing_file: {p}")

    if failures:
        return FaissAuditReport(
            index_total=0,
            ids_total=0,
            sqlite_active_embeds=0,
            dims=0,
            failures=failures,
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_path))
    ids = np.load(str(ids_path))

    if len(ids) != index.ntotal:
        failures.append(f"ids_len_mismatch: ids={len(ids)} index.ntotal={index.ntotal}")

    d = int(meta.get("dims", 0) or 0)
    if d <= 0:
        failures.append("meta_dims_invalid")

    conn = read_only_connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        SELECT e.dims AS dims
        FROM embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        WHERE c.is_active=1
        LIMIT 1
    """)
    r = cur.fetchone()
    if r is None:
        failures.append("sqlite_has_no_active_embeddings")
        sqlite_d = 0
    else:
        sqlite_d = int(r["dims"])
        if d and sqlite_d != d:
            failures.append(f"dims_mismatch: sqlite={sqlite_d} meta={d}")

    cur.execute("""
        SELECT COUNT(*)
        FROM embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        WHERE c.is_active=1
    """)
    sqlite_active_embeds = int(cur.fetchone()[0] or 0)

    
    cur.execute("""
        SELECT e.chunk_id
        FROM embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        WHERE c.is_active=1
        ORDER BY e.chunk_id
    """)
    sqlite_ids = [int(x[0]) for x in cur.fetchall()]

    faiss_ids = ids.astype(np.int64, copy=False).tolist()

    if len(sqlite_ids) != len(faiss_ids):
        failures.append(f"count_mismatch: sqlite_active={len(sqlite_ids)} faiss={len(faiss_ids)}")
    else:
        if sqlite_ids != faiss_ids:
            failures.append("id_order_or_content_mismatch_between_sqlite_and_faiss")

    if hasattr(index, "reconstruct") and index.ntotal > 0 and d > 0:
        n = int(index.ntotal)
        take = min(sample_self_test, n)
        step = max(1, n // take)
        q = np.zeros((d,), dtype=np.float32)

        for i in range(0, n, step):
            index.reconstruct(i, q)
            k = min(5, n)
            D, I = index.search(q.reshape(1, -1), k)
            got_positions = [int(x) for x in I[0] if int(x) >= 0]
            top_score = float(D[0, 0]) if D.size else float("-inf")
            if i in got_positions:
                pass
            elif top_score >= 0.999999:
                pass
            else:
                failures.append(
                    f"self_test_failed_at={i} got_topk={got_positions} score={top_score}"
                )
                break
            take -= 1
            if take <= 0:
                break
    
    faiss_cfg = cfg.get("faiss", {}) or {}
    cfg_metric = str(faiss_cfg.get("metric", "cosine")).strip().lower()
    cfg_index_mode = str(faiss_cfg.get("index_mode", "flat_ip")).strip().lower()

    meta_metric = str(meta.get("metric", "")).strip().lower()
    meta_index = str(meta.get("faiss_index", "")).strip().lower()

    if meta_metric and meta_metric != cfg_metric:
        failures.append(f"metric_mismatch: cfg={cfg_metric} meta={meta_metric}")

    if meta_index and meta_index != cfg_index_mode:
        failures.append(f"index_mode_mismatch: cfg={cfg_index_mode} meta={meta_index}")

    return FaissAuditReport(
        index_total=int(index.ntotal),
        ids_total=int(len(ids)),
        sqlite_active_embeds=sqlite_active_embeds,
        dims=int(d or sqlite_d or 0),
        failures=failures,
    )