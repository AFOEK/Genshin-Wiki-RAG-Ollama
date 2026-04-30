from __future__ import annotations

import sqlite3
import logging

log = logging.getLogger(__name__)

def expand_context_windows(conn: sqlite3.Connection, seed_chunks: list[dict], *, before: int = 1, after: int = 1, max_total_chunks: int = 30) -> list[dict]:
    if not seed_chunks:
        return []
    
    before = max(0, int(before))
    after = max(0, int(after))

    max_total_chunks = max(1, int(max_total_chunks))

    cur = conn.cursor()
    expanded: list[dict] = []
    seen_chunks_ids: set[int] = set()

    seed_ids = [int(r["chunk_id"]) for r in seed_chunks]
    placeholder = ','.join("?" for _ in seed_ids)
    cur.execute(f"""
    SELECT chunk_id, doc_id, chunk_index
    FROM chunks
    WHERE chunk_id IN ({placeholder})
    """, (seed_ids,))
    pos_by_chunk_id = {int(r["chunk_id"]): (int(r["doc_id"]), int(r["chunk_index"])) for r in cur.fetchall()}

    for seed in seed_chunks:
        seed_id = int(seed["chunk_id"])
        pos = pos_by_chunk_id.get("seed_id")
        if pos is None:
            continue

        doc_id, chunk_index = pos
        start_idx = max(0, chunk_index - before)
        end_idx = chunk_index + after
        cur.execute("""
        SELECT
            c.chunk_id,
            c.doc_id,
            c.chunk_index,
            c.text,
            d.source,
            d.url,
            d.title,
            d.tier,
            d.weight
        FROM chunks c
        JOIN docs d ON d.doc_id = c.doc_id
        WHERE c.doc_id = ? AND c.is_active = 1 AND c.chunk_index
        BETWEEN ? AND ?
        AND COALESCE(d.status, 1) = 1
        ORDER BY c.chunk_index
        """, (doc_id, start_idx, end_idx))
        for row in cur.fetchall():
            cid = int(row["chunk_id"])
            if cid in seen_chunks_ids:
                continue

            item = dict(row)
            item["is_seed_chunk"] = 1 if cid == seed_id else 0
            item["seed_chunk_id"] = seed_id
            expanded.append(item)
            seen_chunks_ids.add(cid)
            if len(expanded) >= max_total_chunks:
                log.info("[CTX_EXPAND] Reaced max_total_chunks=%d", max_total_chunks)
                return expanded
    return expanded