import logging
from datetime import datetime, timezone
from tqdm import tqdm

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
    log.info(f"Processing document for {title} with {url}")
    cleaned = raw_text

    if source in ("genshin_wiki", "fandom_api", "wiki"):
        cleaned = clean_fandom_text(raw_text)
    
    norm = normalize(cleaned)
    raw_hash = sha256_text(raw_text)
    norm_hash = sha256_text(norm)

    archive_raw = bool(config.get("pipeline", {}).get("archive_raw", False))
    raw_zst = None
    raw_len = None
    raw_zst_len = None

    if archive_raw:
        raw_len = len(raw_text)
        raw_zst = zstd_compress_text(raw_text)
        raw_zst_len = len(raw_zst)
        log.info(f"Compressed document using zstd with {raw_zst_len}")

    cur = conn.cursor()
    cur.execute("SELECT doc_id, norm_hash FROM docs WHERE url=?", (url,))
    row = cur.fetchone()

    if row:
        doc_id_exisiting, old_norm_hash = row

        cur.execute("SELECT COUNT(*) FROM chunks where doc_id=? AND is_active=?",(doc_id_exisiting, 1))
        active_chunks = cur.fetchone()[0]

        if old_norm_hash == norm_hash and active_chunks > 0:
            log.info("SKIP, chunk exist")
            return
        else:
            log.warning("REBUILD, content changed or unchange but chunks missing")
    
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

        for i in range(6):
            try:
                vec, dims = embed_fn(safe_txt)
                break
            except Exception as e:
                log.exception(f"Too long chunks token retrying {i}/6")
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