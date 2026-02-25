import logging
from datetime import datetime, timezone

from utils.hashing import sha256_text
from utils.textproc import normalize, chunk_text
from utils.codec import zstd_compress_text
from utils.clean_fandom import clean_fandom_text

log = logging.getLogger(__name__)

def defang_tables(s: str) -> str:
    lines = s.splitlines()
    table_lines = sum(1 for l in lines if "|" in l)
    pipe_count = s.count("|")
    if table_lines > 10 or pipe_count > 80:
        lines = [l.replace("|", " ") for l in lines]
        s = "\n".join(lines)
    else:
        s = "\n".join(lines)
    if s.count("[[")> 20:
        s = s.replace("[[", " ").replace("]]", " ")
    if s.count("{{") > 10:
        s = s.replace("{{", " ").replace("}}", " ")
    return s 

def process_document(conn, embed_fn, config, source, url, title, raw_text, tier="primary", weight=1.0):
    cur = conn.cursor()
    cur.execute("SELECT doc_id, raw_hash FROM docs WHERE url=?", (url,))
    row = cur.fetchone()
    raw_hash = sha256_text(raw_text)
    if row:
        doc_id_existing, old_raw_hash = row
        if old_raw_hash == raw_hash:
            cur.execute("SELECT COUNT(*) FROM chunks WHERE doc_id=? AND is_active=1", (doc_id_existing,))
            active_chunks = int(cur.fetchone()[0] or 0)
            if active_chunks > 0:
                cur.execute("""
                    SELECT COUNT(*)
                    FROM chunks c
                    LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
                    WHERE c.doc_id=? AND c.is_active=1 AND e.chunk_id IS NULL
                """, (doc_id_existing,))
                missing_emb = int(cur.fetchone()[0] or 0)
                if missing_emb == 0:
                    log.info(f"SKIP {url} (doc+chunks+embeddings already complete)")
                    return
                log.warning(f"Embeddings missing for {missing_emb} chunks, embedding-only pass for {url}")
                MAX_EMBED_CHARS = int(config.get("pipeline", {}).get("max_embed_chars", 1800))
                MIN_EMBED_CHARS = int(config.get("pipeline", {}).get("min_embed_chars", 800))
                cur.execute("""
                    SELECT c.chunk_id, c.text
                    FROM chunks c
                    LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
                    WHERE c.doc_id=? AND c.is_active=1 AND e.chunk_id IS NULL
                """, (doc_id_existing,))
                rows = cur.fetchall()
                for cid, txt in rows:
                    safe_txt = txt[:MAX_EMBED_CHARS] if len(txt) > MAX_EMBED_CHARS else txt
                    safe_txt = defang_tables(safe_txt)
                    vec = dims = None
                    last_err = None
                    for attempt in range(6):
                        try:
                            vec, dims = embed_fn(safe_txt)
                            break
                        except Exception as e:
                            last_err = e
                            log.exception("Embed retry %d/6 chunk_id=%s", attempt+1, cid)
                            if len(safe_txt) <= MIN_EMBED_CHARS:
                                break
                            safe_txt = safe_txt[: max(MIN_EMBED_CHARS, len(safe_txt)//4)]

                    if vec is None or dims is None:
                        log.warning(f"embed failed chunk_id={cid} final_len={len(safe_txt)} err={last_err}")
                        continue

                    cur.execute(
                        "INSERT OR REPLACE INTO embeddings(chunk_id, dims, vector) VALUES(?, ?, ?)",
                        (cid, dims, vec)
                    )
                conn.commit()
                return
        log.warning(f"REBUILD {url} (content changed)")

    log.info(f"Processing document for {title} with {url}")
    cleaned = raw_text
    if source in ("genshin_wiki", "fandom_api", "wiki"):
        cleaned = clean_fandom_text(raw_text)
    norm = normalize(cleaned)
    norm_hash = sha256_text(norm)
    archive_raw = bool(config.get("pipeline", {}).get("archive_raw", False))
    raw_zst = raw_len = raw_zst_len = None
    if archive_raw:
        raw_len = len(raw_text)
        raw_zst = zstd_compress_text(raw_text)
        raw_zst_len = len(raw_zst)
    
    cur.execute("""
    INSERT INTO docs(source, url, title, fetched_at, raw_hash, norm_hash, tier, weight, raw_zst, raw_len, raw_zst_len)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(url) DO UPDATE SET
        title=excluded.title,
        fetched_at=excluded.fetched_at,
        raw_hash=excluded.raw_hash,
        norm_hash=excluded.norm_hash,
        tier=excluded.tier,
        weight=excluded.weight,
        raw_zst=excluded.raw_zst,
        raw_len=excluded.raw_len,
        raw_zst_len=excluded.raw_zst_len
    """, (source, url, title, datetime.now(timezone.utc).isoformat(), raw_hash, norm_hash, tier, weight, raw_zst, raw_len, raw_zst_len))

    conn.commit()

    cur.execute("SELECT doc_id FROM docs WHERE url=?", (url,))
    doc_id = cur.fetchone()[0]

    log.info(f"RAW LEN: {len(raw_text)}, NORM LEN: {len(norm)}, title: {title}")

    chunks = chunk_text(norm, config["pipeline"]["chunk_size"], config["pipeline"]["chunk_overlap"])
    cur.execute("UPDATE chunks SET is_active=0 WHERE doc_id=?", (doc_id,))

    for i, c in enumerate(chunks):
        chash = sha256_text(c)
        czst = zstd_compress_text(c)
        clen = len(c)
        czlen = len(czst)

        cur.execute("""
        INSERT INTO chunks(doc_id, chunk_index, text, text_zst, text_len, text_zst_len, chunk_hash, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(doc_id, chunk_index) DO UPDATE SET
            text=excluded.text,
            text_zst=excluded.text_zst,
            text_len=excluded.text_len,
            text_zst_len=excluded.text_zst_len,
            chunk_hash=excluded.chunk_hash,
            is_active=1
        """, (doc_id, i, c, czst, clen, czlen, chash))

    conn.commit()

    cur.execute("""
    SELECT chunk_id, text FROM chunks
    WHERE doc_id=? AND is_active=1
    AND chunk_id NOT IN (SELECT chunk_id FROM embeddings)
    """, (doc_id,))

    rows = cur.fetchall()
    MAX_EMBED_CHARS = int(config.get("pipeline", {}).get("max_embed_chars", 1800))
    MIN_EMBED_CHARS = int(config.get("pipeline", {}).get("min_embed_chars", 800))

    for cid, txt in rows:
        safe_txt = txt[:MAX_EMBED_CHARS] if len(txt) > MAX_EMBED_CHARS else txt
        safe_txt = defang_tables(safe_txt)

        vec = None
        dims = None
        last_err = None

        for i in range(8):
            try:
                vec, dims = embed_fn(safe_txt)
                break
            except Exception as e:
                log.exception("Too long chunks token retrying %d/8", i)
                last_err = e
                if len(safe_txt) <= MIN_EMBED_CHARS:
                    break
                safe_txt = safe_txt[: max(MIN_EMBED_CHARS, len(safe_txt)//4)]

        if vec is None or dims is None:
            log.warning(f"[WARN] embed failed chunk_id={cid} orig_len={len(txt)} final_len={len(safe_txt)} err={last_err}")
            continue

        cur.execute(
            "INSERT OR REPLACE INTO embeddings(chunk_id, dims, vector) VALUES(?, ?, ?)",
            (cid, dims, vec)
        )
    
    conn.commit()