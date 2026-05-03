import sqlite3
import logging

log = logging.getLogger(__name__)

def rebuild_chunks_fts(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    log.info("[FTS5] Creating new chunks records")
    cur.execute("BEGIN IMMEDIATE")
    try:
        cur.execute("DELETE FROM chunks_fts")
        cur.execute("""
            INSERT INTO chunks_fts(rowid, chunk_id, doc_id, source, title, text)
            SELECT
                c.chunk_id,
                c.chunk_id,
                c.doc_id,
                d.source,
                d.title,
                c.text
            FROM chunks c
            JOIN docs d ON d.doc_id = c.doc_id
            WHERE c.is_active = 1
              AND COALESCE(d.status, 1) = 1
        """)
        inserted = cur.rowcount if cur.rowcount is not None else 0
        cur.execute("DELETE FROM fts_dirty_docs")
        conn.commit()
        return {"fts_rows_inserted": int(inserted or 0)}
    except Exception:
        conn.rollback()
        raise

def mark_fts_dirty_docs(conn: sqlite3.Connection, doc_id: int, reason: str = "changed") -> None:
    conn.execute("""
    INSERT INTO fts_dirty_docs(doc_id, reason, marked_at)
    VALUES (?, ?, CURRENT_TIMESTAMP) ON CONFLICT(doc_id) DO
    UPDATE SET 
    reason=excluded.reason,
    marked_at=CURRENT_TIMESTAMP
    """, (int(doc_id), reason))
    log.info("[FTS5] Marked dirty FTS5 docs with reasons %s", reason)

def mark_all_active_docs_dirty(conn: sqlite3.Connection, reason:str = "initial") -> int:
    cur = conn.cursor()
    cur.execute("""
    INSERT OR IGNORE INTO fts_dirty_docs(doc_id, reason, marked_at)
    SELECT DISTINCT d.doc_id, ?, CURRENT_TIMESTAMP
    FROM docs d
    JOIN chunks c ON c.doc_id = d.doc_id
    WHERE c.is_active = 1 AND COALESCE(d.status, 1) = 1            
    """, (reason,))
    n = cur.rowcount if cur.rowcount is not None else 0
    log.info("[FTS5] %d row marked dirty", n)
    conn.commit()
    return int(n or 0)

def sync_dirty_chunks_fts(conn: sqlite3.Connection, batch_size: int=500) -> dict:
    cur = conn.cursor()
    total_docs = 0
    total_inserted = 0

    while True:
        dirty = cur.execute("""
        SELECT doc_id
        FROM fts_dirty_docs
        ORDER BY marked_at
        LIMIT ?
        """, (batch_size)).fetchall()

        if not dirty:
            break

        docs_ids = [int(r[0]) for r in dirty]
        cur.execute("BEGIN IMMEDIATE")
        try:
            for doc_id in docs_ids:
                cur.execute("DELETE FROM chunks_fts WHERE doc_id=?", (doc_id,))

                cur.execute("""
                INSERT INTO chunks_fts(rowid, chunk_id, doc_id, source, title, text)
                SELECT
                    c.chunk_id,
                    c.chunk_id,
                    c.doc_id,
                    d.source,
                    d.title,
                    c.text
                FROM chunks c
                JOIN docs d on d.doc_id = c.doc_id
                WHERE c.doc_id = ? AND c.is_active = 1 AND COALESCE(d.status, 1) = 1
                """, (doc_id,))
                inserted = cur.rowcount
                if inserted and inserted > 0:
                    total_inserted += inserted

                cur.execute("DELETE FROM fts_dirty_docs WHERE doc_id=?", (doc_id,))
                
            conn.commit()
            total_docs += len(docs_ids)
            log.info("[FTS5] Synced dirty docs batch=%d total_docs=%d total_inserted=%d", len(docs_ids), total_docs, total_inserted)
        except Exception:
            conn.rollback()
            raise
    
    return {
        "dirty_docs_synced": total_docs,
        "fts_rows_inserted": total_inserted,
    }