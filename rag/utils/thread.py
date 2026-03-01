import threading, queue, logging, time
from dataclasses import dataclass
from typing import Callable
from core.pipeline import process_document, defang_tables
from core.db import connect

log = logging.getLogger(__name__)
STOP = object()

@dataclass(frozen=True)
class EmbedJob:
    chunk_id: int
    text: str

@dataclass(frozen=True)
class EmbedResult:
    chunk_id: int
    dims: int | None
    vec: bytes | None

def safe_process(conn, embed_fn, cfg, name, url, title, text, tier, weight, log):
    try:
        process_document(conn, embed_fn, cfg, name, url, title, text, tier=tier, weight=weight)
        return True
    except Exception:
        log.exception("[THREAD] process_document failed source=%s title=%s url=%s", name, title, url)
        return False

def producer(source_name: str, docs_iter, out_q: queue.Queue):
    produced = 0
    try:
        for url, title, text in docs_iter:
            if out_q.full():
                log.warning("[PRODUCER] [%s] doc queue FULL; waiting...", source_name)
            out_q.put((source_name, url, title, text))
            produced += 1
            if produced % out_q.maxsize == 0:
                log.info("[PRODUCER] [%s] produced=%d q=%d", source_name, produced, out_q.qsize())
    except Exception:
        log.exception("[PRODUCER] [%s] crashed", source_name)
    finally:
        out_q.put((source_name, STOP, STOP, STOP))
        log.info("[PRODUCER] [%s] finished produced=%d", source_name, produced)

def embed_worker(embed_fn: Callable[[str], tuple[bytes, int]], embed_q: queue.Queue, res_q: queue.Queue, cfg: dict, worker_id: int):
    max_chars = int(cfg.get("pipeline", {}).get("max_embed_chars", 1800))
    min_chars = int(cfg.get("pipeline", {}).get("min_embed_chars", 800))
    while True:
        job = embed_q.get()
        try:
            if job is STOP:
                return
            assert isinstance(job, EmbedJob)
            txt = job.text
            safe_txt = txt[:max_chars] if len(txt) > max_chars else txt
            safe_txt = defang_tables(safe_txt)
            vec = dims = None
            last_err = None

            for attempt in range(8):
                try:
                    vec, dims = embed_fn(safe_txt)
                    break
                except Exception as e:
                    last_err = e
                    log.warning("[EMBED-%d] retry %d/8 chunk_id=%s len=%d err=%s",
                                worker_id, attempt + 1, job.chunk_id, len(safe_txt), type(e).__name__)
                    if len(safe_txt) <= min_chars:
                        break
                    safe_txt = safe_txt[:max(min_chars, len(safe_txt) // 4)]

            if vec is None or dims is None:
                log.error("[EMBED-%d] failed chunk_id=%s final_len=%d err=%r",
                          worker_id, job.chunk_id, len(safe_txt), last_err)
                res_q.put(EmbedResult(chunk_id=job.chunk_id, dims=None, vec=None))
                continue
            res_q.put(EmbedResult(chunk_id=job.chunk_id, dims=dims, vec=vec))
        finally:
            embed_q.task_done()

def ingest_consumer(num_producers: int,
                    doc_q: queue.Queue,
                    db_path: str,
                    embed_fn: Callable[[str], tuple[bytes, int]],
                    cfg: dict,
                    filters,
                    tier_map: dict,
                    weight_map: dict,
                    embed_workers: int = 2,
                    embed_queue_size: int = 200):

    log.info("[INGEST] start db_path=%s producers=%d embed_workers=%d",
             db_path, num_producers, embed_workers)

    conn = connect(db_path)

    embed_q: queue.Queue = queue.Queue(maxsize=embed_queue_size)
    embed_res_q: queue.Queue = queue.Queue()

    workers = []
    for wid in range(embed_workers):
        t = threading.Thread(
            target=embed_worker,
            args=(embed_fn, embed_q, embed_res_q, cfg, wid),
            daemon=False,
        )
        t.start()
        workers.append(t)

    finished = 0
    processed = skipped = failed = 0
    pending_embeds = 0
    last_commit = time.monotonic()
    inserted_since_commit = 0

    def drain_results(max_n: int = 200) -> int:
        nonlocal pending_embeds, last_commit, inserted_since_commit
        drained = 0
        batch = []

        while drained < max_n:
            try:
                res = embed_res_q.get_nowait()
            except queue.Empty:
                break
            try:
                if res.vec is not None and res.dims is not None:
                    batch.append((res.chunk_id, res.dims, res.vec))
                    inserted_since_commit += 1
                pending_embeds -= 1
            finally:
                embed_res_q.task_done()
            drained += 1

        if batch:
            cur = conn.cursor()
            cur.executemany(
                "INSERT OR REPLACE INTO embeddings(chunk_id, dims, vector) VALUES(?, ?, ?)",
                batch
            )

        now = time.monotonic()
        if inserted_since_commit >= 2500 or (inserted_since_commit > 0 and now - last_commit >= 3.0):
            conn.commit()
            last_commit = now
            inserted_since_commit = 0

        return drained
    
    try:
        while finished < num_producers:
            try:
                src, url, title, text = doc_q.get(timeout=15)
            except queue.Empty:
                log.info("[INGEST] idle finished=%d/%d doc_q=%d pending=%d embed_q=%d res_q=%d",
                         finished, num_producers, doc_q.qsize(), pending_embeds,
                         embed_q.qsize(), embed_res_q.qsize())
                drain_results(600)
                continue
            try:
                if url is STOP:
                    finished += 1
                    continue

                if not filters.url_allowed(url) or not filters.text_allowed(text):
                    skipped += 1
                    continue
                tier = tier_map.get(src, "primary")
                weight = weight_map.get(src, 1.0)
                chunks_to_embed = process_document(
                    conn, embed_fn, cfg,
                    src, url, title, text,
                    tier=tier, weight=weight,
                    do_embed=False
                ) or []

                processed += 1

                for chunk_id, chunk_text in chunks_to_embed:
                    if embed_q.full():
                        log.warning("[INGEST] embed queue FULL; waiting chunk_id=%s", chunk_id)
                    while True:
                        try:
                            embed_q.put(EmbedJob(chunk_id=chunk_id, text=chunk_text), timeout=1.0)
                            pending_embeds += 1
                            break
                        except queue.Full:
                            drain_results(max_n=100)
                            time.sleep(0.01)

                drain_results(600)

                total = processed + skipped + failed
                if total and total % embed_queue_size == 0:
                    log.info("[INGEST] processed=%d skipped=%d failed=%d doc_q=%d pending=%d embed_q=%d res_q=%d",
                             processed, skipped, failed, doc_q.qsize(), pending_embeds,
                             embed_q.qsize(), embed_res_q.qsize())

                if processed and processed % 650 == 0:
                    conn.commit()

            except Exception:
                failed += 1
                log.exception("[INGEST] failed src=%s url=%s", src, url)

            finally:
                doc_q.task_done()

        log.info("[INGEST] producers finished; waiting embed jobs pending=%d", pending_embeds)
        embed_q.join()
        drain_results(10_000)
        embed_res_q.join()
        conn.commit()

    finally:
        for _ in workers:
            embed_q.put(STOP)
        for t in workers:
            t.join()

        try:
            conn.close()
        except Exception:
            pass

    log.info("[INGEST] DONE processed=%d skipped=%d failed=%d", processed, skipped, failed)