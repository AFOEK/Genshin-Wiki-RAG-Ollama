import logging
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any
from collections.abc import Callable

from utils.hashing import sha256_text
from utils.textproc import normalize, chunk_text
from utils.codec import zstd_compress_text
from utils.clean_fandom import clean_fandom_text
from utils.versioning import extract_version_signal
from core.parent import mark_parent_dirty_doc
from core.splade import mark_splade_dirty_doc
from core.fts import mark_fts_dirty_docs

EmbedFn = Callable[[str], tuple[bytes, int]]

log = logging.getLogger(__name__)

MOVED_URL_SOURCE = set()

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

def process_document(conn: sqlite3.Connection, embed_fn: EmbedFn, config: dict[str, Any], source: str, url: str, title: str, raw_text: str, *, tier: str = "primary", weight: float =1.0, do_embed: bool=True, last_modified: str | None = None, etag: str | None =None) -> list[tuple[int, str]]:
    raw_text = str(raw_text or "").strip()

    if not raw_text:
        log.warning("[PIPELINE] Refusing empty documents: source: %s, url: %s", source, url)
        return []

    if (source == "game8" and len(raw_text) < 800):
        log.warning("[GAME8] Refusing undersized document url: %s chars: %d", url, len(raw_text))
        return []
    
    cur = conn.cursor()
    source_cfg = next(
        (
            item
            for item in config.get("sources", [])
            if item.get("name") == source
        ),
        {},
    )
    force_rebuild = bool(source_cfg.get("force_rebuild", False))
    try:
        cur.execute("BEGIN IMMEDIATE")
        raw_hash = sha256_text(raw_text)
        version_label, version_ord = extract_version_signal(title, raw_text, config, source=source)
        doc_changed = False
        cur.execute("SELECT doc_id, raw_hash FROM docs WHERE url=?", (url,))
        row = cur.fetchone()
        if row is None and source in MOVED_URL_SOURCE:
            cur.execute(
                "SELECT doc_id, url, raw_hash FROM docs WHERE source=? AND raw_hash=? ORDER BY doc_id DESC LIMIT 1",
                (source, raw_hash))
            moved = cur.fetchone()
            if moved:
                doc_id_existing, old_url, old_hash_raw = moved
                log.info("[INFO] URL moved detected: %s -> %s", old_url, url)
                cur.execute("DELETE FROM docs WHERE url=? AND doc_id<>?", (url, doc_id_existing))
                log.info("[INFO] DELETED DUPLICATE RECORDS url=%s and keep_doc_id=%s", url, doc_id_existing)
                cur.execute(
                    """
                    UPDATE docs
                    SET url=?,
                        title=?,
                        fetched_at=?,
                        tier=?,
                        weight=?,
                        last_modified=?,
                        etag=?,
                        version_label=?,
                        version_ord=?
                    WHERE doc_id=?
                    """,
                    (url, title, datetime.now(timezone.utc).isoformat(), tier, weight, last_modified, etag, version_label, version_ord, doc_id_existing),
                )
                mark_fts_dirty_docs(conn, doc_id_existing, reason="url_moved_metadata")
                row = (doc_id_existing, old_hash_raw)
        if row:
            doc_id_existing, old_raw_hash = row
            if (old_raw_hash == raw_hash and not force_rebuild):
                cur.execute("SELECT COUNT(*) FROM chunks WHERE doc_id=? AND is_active=1", (doc_id_existing,))
                active_chunks = int(cur.fetchone()[0] or 0)
                if active_chunks > 0:
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM chunks c
                        LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
                        WHERE c.doc_id=? AND c.is_active=1 AND e.chunk_id IS NULL
                        """,
                        (doc_id_existing,),
                    )
                    missing_emb = int(cur.fetchone()[0] or 0)
                    if missing_emb == 0:
                        cur.execute(
                            """
                            UPDATE docs
                            SET title=?, fetched_at=?, tier=?, weight=?, last_modified=?, etag=?, version_label=?, version_ord=?
                            WHERE doc_id=?
                            """,
                            (title, datetime.now(timezone.utc).isoformat(), tier, weight, last_modified, etag, version_label, version_ord, doc_id_existing))
                        mark_fts_dirty_docs(conn, doc_id_existing, reason="metadata_refresh")
                        conn.commit()
                        log.info("SKIP %s (doc+chunks+embeddings already complete)", url)
                        return []
                    log.warning("Embeddings missing for %d chunks, embedding-only pass for %s", missing_emb, url)

                    MAX_EMBED_CHARS = int(config.get("pipeline", {}).get("max_embed_chars", 1800))
                    MIN_EMBED_CHARS = int(config.get("pipeline", {}).get("min_embed_chars", 800))
                    cur.execute(
                        """
                        SELECT c.chunk_id, c.text
                        FROM chunks c
                        LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
                        WHERE c.doc_id=? AND c.is_active=1 AND e.chunk_id IS NULL
                        ORDER BY c.chunk_index
                        """,
                        (doc_id_existing,),
                    )
                    rows = cur.fetchall()
                    if not do_embed:
                        conn.commit()
                        return rows
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
                                log.exception("Embed retry %d/6 chunk_id=%s", attempt + 1, cid)
                                if len(safe_txt) <= MIN_EMBED_CHARS:
                                    break
                                safe_txt = safe_txt[: max(MIN_EMBED_CHARS, len(safe_txt) // 4)]

                        if vec is None or dims is None:
                            log.warning("embed failed chunk_id=%s final_len=%d err=%s", cid, len(safe_txt), last_err)
                            continue
                        cur.execute(
                            "INSERT OR REPLACE INTO embeddings(chunk_id, dims, vector) VALUES(?, ?, ?)",
                            (cid, dims, vec),
                        )
                    conn.commit()
                    return []
                log.warning("REBUILD %s (unchanged raw, but no active chunks)", url)
                doc_changed = True
            elif(old_raw_hash == raw_hash and force_rebuild):
                log.warning("[REBUILD] forced source=%s url=%s", source, url)
                doc_changed = True
            else:
                log.warning("[WARN] REBUILD %s (content changed)", url)
                doc_changed = True
        log.info("Processing document title=%s url=%s", title, url)

        cleaned = raw_text
        if source in ("genshin_wiki", "fandom_api", "wiki"):
            cleaned = clean_fandom_text(raw_text)

        norm = normalize(cleaned)
        norm_hash = sha256_text(norm)

        if source == "game8":
            cur.execute(
                """
                SELECT
                    doc_id,
                    url,
                    title
                FROM docs
                WHERE source = 'game8'
                AND norm_hash = ?
                AND url <> ?
                AND status = 1
                LIMIT 1
                """,
                (norm_hash, url,),)

            duplicate = cur.fetchone()
            if duplicate is not None:
                duplicate_doc_id = int(duplicate[0])
                duplicate_url = str(duplicate[1])
                duplicate_title = str(duplicate[2] or "")
                log.warning("[GAME8] Rejecting duplicate body " "url=%s duplicate_doc_id=%d " "duplicate_url=%s " "duplicate_title=%r", url, duplicate_doc_id, duplicate_url, duplicate_title,)
                conn.rollback()
                return []
            
        chunks = chunk_text(norm, config["pipeline"]["chunk_size"], config["pipeline"]["chunk_overlap"])

        global_filter_cfg = config.get("filters", {}) or {}
        global_deny_pattern = (global_filter_cfg.get("chunk_deny_text_regex") or global_filter_cfg.get("deny_text_regex"))

        source_cfg = next(
            (
                source_cfg
                for source_cfg in config.get("sources", [])
                if source_cfg.get("name") == source
            ),
            {},
        )
        source_deny_pattern = (source_cfg.get("chunk_deny_text_regex") or source_cfg.get("deny_text_regex"))
        deny_patterns = [pattern for pattern in (global_deny_pattern, source_deny_pattern) if pattern]

        deny_text_re = (
            re.compile(
                "|".join(
                    f"(?:{pattern})"
                    for pattern in deny_patterns
                ),
                re.IGNORECASE,
            )
            if deny_patterns
            else None
        )

        original_chunk_count = len(chunks)
        filtered_chunks: list[tuple[int, str]] = []

        for original_index, chunk in enumerate(chunks):
            chunk_value = str(chunk or "").strip()

            if not chunk_value:
                continue

            if (deny_text_re is not None and deny_text_re.search(chunk_value)):
                log.debug("[CHUNK_FILTER] rejected source=%s url=%s chunk_index=%d preview=%r", source, url, original_index, chunk_value[:160],)
                continue

            filtered_chunks.append((original_index, chunk_value))

        chunks = filtered_chunks

        log.info("[CHUNK_FILTER] source=%s total=%d accepted=%d rejected=%d", source, original_chunk_count, len(chunks), original_chunk_count - len(chunks),)

        archive_raw = bool(config.get("pipeline", {}).get("archive_raw", False))
        raw_zst = raw_len = raw_zst_len = None
        if archive_raw:
            raw_len = len(raw_text)
            raw_zst = zstd_compress_text(raw_text)
            raw_zst_len = len(raw_zst)

        if not chunks:
            log.warning("[PIPELINE] No usable chunks producedm preserving previous document source=%s title=%s url=%s", source, title, url,)
            conn.rollback()
            return []

        cur.execute(
            """
            INSERT INTO docs(source, url, title, fetched_at, raw_hash, norm_hash, tier, weight, last_modified, etag, version_label, version_ord, raw_zst, raw_len, raw_zst_len)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                title=excluded.title,
                fetched_at=excluded.fetched_at,
                raw_hash=excluded.raw_hash,
                norm_hash=excluded.norm_hash,
                tier=excluded.tier,
                weight=excluded.weight,
                last_modified=excluded.last_modified,
                etag=excluded.etag,
                version_label=excluded.version_label,
                version_ord=excluded.version_ord,
                raw_zst=excluded.raw_zst,
                raw_len=excluded.raw_len,
                raw_zst_len=excluded.raw_zst_len,
                status=1
            """,
            (
                source,
                url,
                title,
                datetime.now(timezone.utc).isoformat(),
                raw_hash,
                norm_hash,
                tier,
                weight,
                last_modified,
                etag,
                version_label,
                version_ord,
                raw_zst,
                raw_len,
                raw_zst_len,
            ),
        )
        cur.execute("SELECT doc_id FROM docs WHERE url=?", (url,))
        doc_id = cur.fetchone()[0]
        cur.execute("UPDATE chunks SET is_active=0 WHERE doc_id=?", (doc_id,))
        for i, c in chunks:
            chash = sha256_text(c)
            czst = zstd_compress_text(c)
            clen = len(c)
            czlen = len(czst)
            cur.execute(
                """
                INSERT INTO chunks(doc_id, chunk_index, text, text_zst, text_len, text_zst_len, chunk_hash, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(doc_id, chunk_index) DO UPDATE SET
                    text=excluded.text,
                    text_zst=excluded.text_zst,
                    text_len=excluded.text_len,
                    text_zst_len=excluded.text_zst_len,
                    chunk_hash=excluded.chunk_hash,
                    is_active=1
                """,
                (doc_id, i, c, czst, clen, czlen, chash),
            )
        mark_fts_dirty_docs(conn, doc_id, reason="chunks_changed")
        mark_parent_dirty_doc(conn, doc_id, reason="chunks_changed")
        mark_splade_dirty_doc(conn, doc_id, reason="chunks_changed")
        if doc_changed:
            cur.execute(
                """
                DELETE FROM embeddings
                WHERE chunk_id IN (
                    SELECT chunk_id
                    FROM chunks
                    WHERE doc_id=? AND is_active=1
                )
                """,
                (doc_id,),
            )
            log.info("[INFO] Cleared stale embeddings for rebuilt doc_id=%s url=%s", doc_id, url)

        cur.execute(
            """
            SELECT c.chunk_id, c.text
            FROM chunks c
            LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
            WHERE c.doc_id=? AND c.is_active=1 AND e.chunk_id IS NULL
            ORDER BY c.chunk_index
            """,
            (doc_id,),
        )
        rows = cur.fetchall()

        if not do_embed:
            conn.commit()
            return rows

        MAX_EMBED_CHARS = int(config.get("pipeline", {}).get("max_embed_chars", 1800))
        MIN_EMBED_CHARS = int(config.get("pipeline", {}).get("min_embed_chars", 800))

        for cid, txt in rows:
            safe_txt = txt[:MAX_EMBED_CHARS] if len(txt) > MAX_EMBED_CHARS else txt
            safe_txt = defang_tables(safe_txt)
            vec = dims = None
            last_err = None

            for attempt in range(8):
                try:
                    vec, dims = embed_fn(safe_txt)
                    break
                except Exception as e:
                    last_err = e
                    log.exception("Embed retry %d/8 chunk_id=%s", attempt + 1, cid)
                    if len(safe_txt) <= MIN_EMBED_CHARS:
                        break
                    safe_txt = safe_txt[: max(MIN_EMBED_CHARS, len(safe_txt) // 4)]

            if vec is None or dims is None:
                log.warning("[INFO] embed failed chunk_id=%s orig_len=%d final_len=%d err=%s", cid, len(txt), len(safe_txt), last_err)
                continue

            cur.execute(
                "INSERT OR REPLACE INTO embeddings(chunk_id, dims, vector) VALUES(?, ?, ?)",
                (cid, dims, vec),
            )

        conn.commit()
        return []

    except Exception:
        conn.rollback()
        raise