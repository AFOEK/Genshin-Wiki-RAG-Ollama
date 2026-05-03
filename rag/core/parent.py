from __future__ import annotations

import logging
import sqlite3

log = logging.getLogger(__name__)

def mark_parent_dirty_doc(
    conn: sqlite3.Connection,
    doc_id: int,
    reason: str = "chunks_changed",
) -> None:
    log.info("[PARENT] Marking dirty parent docs")
    conn.execute(
        """
        INSERT INTO parent_dirty_docs(doc_id, reason, marked_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(doc_id) DO UPDATE SET
            reason=excluded.reason,
            marked_at=CURRENT_TIMESTAMP
        """,
        (int(doc_id), reason),
    )


def mark_all_active_docs_parent_dirty(
    conn: sqlite3.Connection,
    reason: str = "initial_parent_build",
) -> int:
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO parent_dirty_docs(doc_id, reason, marked_at)
        SELECT DISTINCT d.doc_id, ?, CURRENT_TIMESTAMP
        FROM docs d
        JOIN chunks c ON c.doc_id = d.doc_id
        WHERE c.is_active = 1
          AND COALESCE(d.status, 1) = 1
    """, (reason,))
    n = int(cur.rowcount or 0)
    conn.commit()
    log.info("[PARENT] Dirty docs counts %d", n)
    return n

def rebuild_parent_for_doc(
    cur: sqlite3.Cursor,
    doc_id: int,
    *,
    children_per_parent: int,
) -> tuple[int, int]:
    cur.execute(
        """
        DELETE FROM chunk_parent_map
        WHERE parent_id IN (
            SELECT parent_id
            FROM chunk_parents
            WHERE doc_id = ?
        )
        """,
        (doc_id,),
    )
    cur.execute(
        "DELETE FROM chunk_parents WHERE doc_id = ?",
        (doc_id,),
    )

    cur.execute(
        """
        SELECT COUNT(*)
        FROM chunks c
        JOIN docs d ON d.doc_id = c.doc_id
        WHERE c.doc_id = ?
          AND c.is_active = 1
          AND COALESCE(d.status, 1) = 1
        """,
        (doc_id,),
    )
    active_count = int(cur.fetchone()[0] or 0)
    if active_count == 0:
        return 0, 0

    cur.execute("DROP TABLE IF EXISTS temp_parent_doc_chunks")
    cur.execute(
        """
        CREATE TEMP TABLE temp_parent_doc_chunks AS
        SELECT
            c.chunk_id,
            c.doc_id,
            c.chunk_index,
            CAST(c.chunk_index / ? AS INTEGER) AS parent_index
        FROM chunks c
        JOIN docs d ON d.doc_id = c.doc_id
        WHERE c.doc_id = ?
          AND c.is_active = 1
          AND COALESCE(d.status, 1) = 1
        """,
        (children_per_parent, doc_id),
    )

    cur.execute(
        """
        INSERT INTO chunk_parents(
            doc_id,
            parent_index,
            start_chunk_index,
            end_chunk_index,
            is_active
        )
        SELECT
            doc_id,
            parent_index,
            MIN(chunk_index),
            MAX(chunk_index),
            1
        FROM temp_parent_doc_chunks
        GROUP BY doc_id, parent_index
        """
    )
    parents_inserted = int(cur.rowcount or 0)

    cur.execute(
        """
        INSERT INTO chunk_parent_map(chunk_id, parent_id)
        SELECT
            a.chunk_id,
            p.parent_id
        FROM temp_parent_doc_chunks a
        JOIN chunk_parents p
          ON p.doc_id = a.doc_id
         AND p.parent_index = a.parent_index
        """
    )
    maps_inserted = int(cur.rowcount or 0)

    cur.execute("DROP TABLE IF EXISTS temp_parent_doc_chunks")
    log.info("[PARENT] parents_inserted: %d, maps_inserted: %d", parents_inserted, maps_inserted)

    return parents_inserted, maps_inserted

def sync_dirty_parent_docs(
    conn: sqlite3.Connection,
    *,
    children_per_parent: int = 4,
    batch_size: int = 500,
) -> dict:
    children_per_parent = max(1, int(children_per_parent))
    batch_size = max(1, int(batch_size))

    cur = conn.cursor()

    total_docs = 0
    total_parents = 0
    total_maps = 0
    log.info("[PARENT] Syncing dirty parent docs")
    while True:
        rows = cur.execute(
            """
            SELECT doc_id
            FROM parent_dirty_docs
            ORDER BY marked_at
            LIMIT ?
            """,
            (batch_size,),
        ).fetchall()

        if not rows:
            break

        doc_ids = [int(r[0]) for r in rows]

        cur.execute("BEGIN IMMEDIATE")
        try:
            for doc_id in doc_ids:
                p_count, m_count = rebuild_parent_for_doc(
                    cur,
                    doc_id,
                    children_per_parent=children_per_parent,
                )

                total_parents += p_count
                total_maps += m_count

                cur.execute(
                    "DELETE FROM parent_dirty_docs WHERE doc_id = ?",
                    (doc_id,),
                )

            conn.commit()
            total_docs += len(doc_ids)

            log.info(
                "[PARENT] synced dirty docs batch=%d total_docs=%d parents=%d mapped_chunks=%d",
                len(doc_ids),
                total_docs,
                total_parents,
                total_maps,
            )

        except Exception:
            conn.rollback()
            raise
    
    log.info("[PARENT] Sync done dirty_docs=%d parents=%d mapped_chunks=%d", total_docs, total_parents, total_maps)

    return {
        "dirty_docs_synced": total_docs,
        "parents_inserted": total_parents,
        "mapped_chunks": total_maps,
        "children_per_parent": children_per_parent,
    }

def rebuild_parent_map(conn: sqlite3.Connection, *, childern_per_parent: int= 4) -> dict:
    childern_per_parent = max(1, childern_per_parent)
    cur = conn.cursor()
    log.info("[PARENT] Rebuild parent-child map")
    cur.execute("BEGIN IMMEDIATE")
    try:
        cur.execute("DELETE FROM chunk_parent_map")
        cur.execute("DELETE FROM chunk_parents")

        cur.execute("DROP TABLE IF EXISTS temp_parent_active")
        cur.execute("""
        CREATE TEMP TABLE temp_parent_active AS
        SELECT
            c.chunk_id,
            c.doc_id,
            c.chunk_index,
            CAST(c.chunk_index / ? AS INTEGER) AS parent_index
        FROM chunks c
        JOIN docs d ON d.doc_id = c.doc_id
        WHERE c.is_active = 1
        AND COALESCE(d.status, 1) = 1
        """, (childern_per_parent,))

        cur.execute("""
            INSERT INTO chunk_parents(
                doc_id,
                parent_index,
                start_chunk_index,
                end_chunk_index,
                is_active
            )
            SELECT
                doc_id,
                parent_index,
                MIN(chunk_index),
                MAX(chunk_index),
                1
            FROM temp_parent_active
            GROUP BY doc_id, parent_index
        """)

        parent_count = int(cur.rowcount or 0)

        cur.execute("""
            INSERT INTO chunk_parent_map(chunk_id, parent_id)
            SELECT
                a.chunk_id,
                p.parent_id
            FROM temp_parent_active a
            JOIN chunk_parents p
              ON p.doc_id = a.doc_id
             AND p.parent_index = a.parent_index
        """)

        map_count = int(cur.rowcount or 0)

        cur.execute("DROP TABLE IF EXISTS temp_parent_active")
        conn.commit()

        log.info("[PARENT] Rebuild parent map parents=%d, mapped_chunks=%d, children_per_parents=%d", parent_count, map_count, childern_per_parent)

        return{
            "parents": parent_count,
            "mapped_chunks": map_count,
            "childern_per_parent": childern_per_parent
        }
    except Exception:
        conn.rollback()
        raise