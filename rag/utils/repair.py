from __future__ import annotations
import logging
import sqlite3
from typing import Callable

from utils.codec import zstd_decompress_text
from core.pipeline import process_document

log = logging.getLogger(__name__)

def find_docs_with_no_active_chunks(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.cursor()
    cur.execute("""
    SELECT d.doc_id, d.source, d.url, d.title, d.tier, d.weight, d.raw_zst
        FROM docs d
        LEFT JOIN chunks c ON c.doc_id = d.doc_id AND c.is_active=1
        GROUP BY d.doc_id
        HAVING COUNT(c.chunk_id)=0
    """)
    return [dict(r) for r in cur.fetchall()]

def find_active_chunks_missing_embeddings(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.cursor()
    cur.execute("""
    SELECT c.chunk_id, c.doc_id, d.source, d.url, d.title, c.text
        FROM chunks c
        JOIN docs d ON d.doc_id = c.doc_id
        LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
        WHERE c.is_active=1 AND e.chunk_id IS NULL
    """)
    return [dict(r) for r in cur.fetchall()]

def repair_doc_from_archived_raw(conn: sqlite3.Connection, embed_fn: Callable, cfg: dict, doc_row:dict) -> bool:
    raw_zst = doc_row.get("raw_zst")
    if not raw_zst:
        log.warning("[REPAIR] doc_id=%s url=%s has no raw compressed; cannot repair from archive", doc_row["doc_id"], doc_row["url"])
        return False
    
    try:
        raw_txt = zstd_decompress_text(raw_zst)
        process_document(conn,
            embed_fn,
            cfg,
            doc_row["source"],
            doc_row["url"],
            doc_row["title"],
            raw_txt,
            tier=doc_row.get("tier", "primary"),
            weight=float(doc_row.get("weight", 1.0)),
            do_embed=True)
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*)
            FROM chunks
            WHERE doc_id=? AND is_active=1
        """, (doc_row["doc_id"],))
        active_chunks = int(cur.fetchone()[0] or 0)
        if active_chunks == 0:
            log.warning("[REPAIR] doc_id=%s url=%s still has no active chunks after repair", doc_row["doc_id"], doc_row["url"])
        log.info("[REPAIR] repaired from archived raw doc_id=%s url=%s", doc_row["doc_id"], doc_row["url"])
        return True
    except Exception:
        log.exception("[REPAIR] failed archived-raw repair doc_id=%s url=%s", doc_row["doc_id"], doc_row["url"])
        return False
    
def repair_missing_embeddings(conn: sqlite3.Connection, embed_fn: Callable, cfg: dict, rows: list[dict]) -> int:
    repaired = 0
    max_chars = int(cfg.get("pipeline", {}).get("max_embed_chars", 1800))
    cur = conn.cursor()

    for row in rows:
        chunk_id = int(row["chunk_id"])
        txt = row["text"] or ""
        safe_txt = txt[:max_chars] if len(txt) > max_chars else txt
        try:
            vec, dims = embed_fn(safe_txt)
            cur.execute("INSERT OR REPLACE INTO embeddings(chunk_id, dims, vector) VALUES (?, ?, ?)", (chunk_id, dims, vec))
            repaired += 1
        except Exception:
            log.exception("[REPAIR] failed embedding repair chunk_id=%s doc_id=%s url=%s", chunk_id, row["doc_id"], row["url"])
    
    conn.commit()
    return repaired

def repair_database(conn: sqlite3.Connection, embed_fn: Callable, cfg: dict) -> dict:
    report = {
        "docs_no_active_chunks_found": 0,
        "docs_no_active_chunks_repaired": 0,
        "chunks_missing_embeddings_found": 0,
        "chunks_missing_embeddings_repaired": 0
    }
    broken_docs = find_docs_with_no_active_chunks(conn)
    report["docs_no_active_chunks_found"] = len(broken_docs)
    for row in broken_docs:
        ok = repair_doc_from_archived_raw(conn, embed_fn, cfg, row)
        if ok:
            report["docs_no_active_chunks_repaired"] += 1
    
    missing_embed = find_active_chunks_missing_embeddings(conn)
    report["chunks_missing_embeddings_found"] = len(missing_embed)

    repaired = repair_missing_embeddings(conn, embed_fn, cfg, missing_embed)
    report["chunks_missing_embeddings_repaired"] = repaired
    return report