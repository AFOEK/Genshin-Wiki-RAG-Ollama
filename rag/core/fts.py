def rebuild_chunks_fts(conn):
    cur = conn.cursor()
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
    """)
    conn.commit()