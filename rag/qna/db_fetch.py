from __future__ import annotations
import sqlite3

def fetch_chunks(conn: sqlite3.Connection, chunk_ids: list[int]) -> list[dict]:
    if not chunk_ids:
        return []

    placeholders = ",".join("?" for _ in chunk_ids)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            c.chunk_id,
            c.chunk_index,
            c.text,
            d.doc_id,
            d.source,
            d.url,
            d.title,
            d.tier,
            d.weight
        FROM chunks c
        JOIN docs d ON d.doc_id = c.doc_id
        WHERE c.chunk_id IN ({placeholders})
    """, chunk_ids)

    rows = [dict(r) for r in cur.fetchall()]
    row_map = {int(r["chunk_id"]): r for r in rows}
    return [row_map[cid] for cid in chunk_ids if cid in row_map]