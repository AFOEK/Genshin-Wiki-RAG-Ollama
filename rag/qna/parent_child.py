from __future__ import annotations

import logging
import sqlite3

log = logging.getLogger(__name__)

def fetch_parent_context_chunks(conn: sqlite3.Connection, seed_chunks: list[dict], *, max_parents: int = 8, max_total_chunks: int = 32) -> list[dict]:
    if not seed_chunks:
        return []

    max_parents = max(1, int(max_parents))
    max_total_chunks = max(1, int(max_total_chunks))

    cur = conn.cursor()
    seed_ids = [int(r["chunk_id"]) for r in seed_chunks]
    seed_set = set(seed_ids)

    parent_ids: list[int] = []
    seen_parent_ids: set[int] = set()

    for cid in seed_ids:
        row = cur.execute("""
        SELECT parent_id FROM chunk_parent_map WHERE chunk_id = ?
        """, (cid,)).fetchone()

        if row is None:
            continue

        pid = int(row["parent_id"] if isinstance(row, sqlite3.Row) else row[0])
        if pid in seen_parent_ids:
            continue

        seen_parent_ids.add(pid)
        parent_ids.append(pid)

        if len(parent_ids) >= max_parents:
            break

    if not parent_ids:
        log.warning("[PARENT] No parent mappings found; falling back to seed chunks")
        return seed_chunks[:max_total_chunks]

    expanded: list[dict] = []
    seen_chunk_ids: set[int] = set()

    for pid in parent_ids:
        rows = cur.execute("""
        SELECT
            c.chunk_id, c.doc_id, c.chunk_index, c.text,
            d.source, d.url, d.title, d.tier, d.weight,
            p.parent_id, p.parent_index
        FROM chunk_parent_map m
        JOIN chunks c ON c.chunk_id = m.chunk_id
        JOIN docs d ON d.doc_id = c.doc_id
        JOIN chunk_parents p ON p.parent_id = m.parent_id
        WHERE m.parent_id = ?
          AND c.is_active = 1
          AND COALESCE(d.status, 1) = 1
        ORDER BY c.chunk_index
        """, (pid,)).fetchall()

        for row in rows:
            item = dict(row)
            cid = int(item["chunk_id"])

            if cid in seen_chunk_ids:
                continue

            item["is_seed_chunk"] = 1 if cid in seed_set else 0
            expanded.append(item)
            seen_chunk_ids.add(cid)

            if len(expanded) >= max_total_chunks:
                log.info("[PARENT] Reached max_total_chunks=%d", max_total_chunks)
                return expanded

    return expanded