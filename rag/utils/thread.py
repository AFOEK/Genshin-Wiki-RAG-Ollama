import threading, queue, logging
from core.pipeline import process_document
from core.db import connect

log = logging.getLogger(__name__)
STOP = object()

def safe_process(conn, embed_fn, cfg, name, url, title, text, tier, weight, log):
    try:
        process_document(conn, embed_fn, cfg, name, url, title, text, tier=tier, weight=weight)
        return True
    except Exception:
        log.exception("process_document failed source=%s title=%s url=%s", name, title, url)
        return False

def producer(source_name: str, docs_iter, out_q: queue.Queue):
    produced = 0
    try:
        for url, title, text in docs_iter:
            out_q.put((source_name, url, title, text))
            produced += 1
            if produced % 200 == 0:
                log.info(f"[{source_name}] produced={produced} queue_size = {out_q.qsize()}")
    except Exception:
        log.exception(f"[{source_name}] producer crashed")
    finally:
        out_q.put((source_name, STOP, STOP, STOP))
        log.info(f"[{source_name}] finished produced={produced}")

def ingest_consumer(num_producers: int, in_q: queue.Queue, db_path, embed_fn, cfg, filters, tier_map, weight_map):
    conn = connect(db_path)
    finished = 0
    processed = 0
    skipped = 0
    failed = 0

    while finished < num_producers:
        src, url, title, text = in_q.get()
        try:
            if url is STOP:
                finished += 1
                continue

            if not filters.url_allowed(url) or not filters.text_allowed(text):
                skipped += 1
                continue

            tier = tier_map.get(src, "primary")
            weight = weight_map.get(src, 1.0)
            ok = safe_process(conn, embed_fn, cfg, src, url, title, text, tier, weight, log)
            
            if ok:
                processed += 1
            else:
                failed += 1

            if processed and  processed % 200 == 0:
                log.info(f"[Ingest] processed={processed} skipped={skipped} failed={failed} queue_size={in_q.qsize()}")

        except Exception:
            log.exception(f"[Ingest] failed src={src} title={title} url={url}")
        finally:
            try:
                conn.close()
            except Exception:
                pass
            in_q.task_done()

    log.info(f"[Ingest] DONE processed={processed} skipped={skipped} failed={failed}")