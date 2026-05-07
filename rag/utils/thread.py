import threading, queue, logging, time
from dataclasses import dataclass
from typing import Callable
from core.pipeline import process_document, defang_tables
from core.db import connect
from core.embed import NonRetryableEmbedError

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

def embed_batch_resilient(embed_fn, prepared_jobs, min_chars, worker_id):
    if not prepared_jobs:
        return []

    texts = [txt for _, txt in prepared_jobs]

    try:
        batch_results = embed_fn(texts)
        if len(batch_results) != len(prepared_jobs):
            raise RuntimeError(
                f"batch result count mismatch: got={len(batch_results)} expected={len(prepared_jobs)}"
            )

        out = []
        for (job, _), (vec, dims) in zip(prepared_jobs, batch_results):
            out.append(EmbedResult(chunk_id=job.chunk_id, dims=dims, vec=vec))
        return out

    except NonRetryableEmbedError as e:
        if len(prepared_jobs) == 1:
            job, cur_txt = prepared_jobs[0]
            last_err = e

            for attempt in range(8):
                try:
                    vec, dims = embed_fn(cur_txt)
                    return [EmbedResult(chunk_id=job.chunk_id, dims=dims, vec=vec)]
                except NonRetryableEmbedError as ex:
                    last_err = ex
                    log.warning(
                        "[EMBED-%d] shrink retry %d/8 chunk_id=%s len=%d err=%s",
                        worker_id, attempt + 1, job.chunk_id, len(cur_txt), type(ex).__name__
                    )
                    if len(cur_txt) <= min_chars:
                        break
                    cur_txt = cur_txt[:max(min_chars, len(cur_txt) // 2)]
                except Exception as ex:
                    last_err = ex
                    log.warning(
                        "[EMBED-%d] single retry %d/8 chunk_id=%s len=%d err=%s",
                        worker_id, attempt + 1, job.chunk_id, len(cur_txt), type(ex).__name__
                    )
                    if len(cur_txt) <= min_chars:
                        break
                    cur_txt = cur_txt[:max(min_chars, len(cur_txt) // 2)]

            return [EmbedResult(chunk_id=job.chunk_id, dims=None, vec=None)]

        mid = len(prepared_jobs) // 2
        left = embed_batch_resilient(embed_fn, prepared_jobs[:mid], min_chars, worker_id)
        right = embed_batch_resilient(embed_fn, prepared_jobs[mid:], min_chars, worker_id)
        return left + right

    except Exception as e:
        if len(prepared_jobs) == 1:
            job, cur_txt = prepared_jobs[0]
            log.warning("[EMBED-%d] non-retryable failure chunk_id=%s err=%s", worker_id, job.chunk_id, e)
            return [EmbedResult(chunk_id=job.chunk_id, dims=None, vec=None)]

        mid = len(prepared_jobs) // 2
        left = embed_batch_resilient(embed_fn, prepared_jobs[:mid], min_chars, worker_id)
        right = embed_batch_resilient(embed_fn, prepared_jobs[mid:], min_chars, worker_id)
        return left + right

def producer(source_name: str, docs_iter, out_q: queue.Queue, source_filters=None):
    produced = 0
    try:
        for item in docs_iter:
            if len(item) == 5:
                url, title, text, last_modified, etag = item
            elif len(item) == 3:
                url, title, text = item
                last_modified = etag = None
            else:
                log.warning("[PRODUCER] [%s] unexpected item shape %d, skipping", source_name, len(item))
                continue

            if source_filters:
                if not source_filters.url_allowed(url):
                    continue
                if not source_filters.text_allowed(text):
                    continue

            if out_q.full():
                log.warning("[PRODUCER] [%s] doc queue FULL; waiting...", source_name)
            out_q.put((source_name, url, title, text, last_modified, etag))
            produced += 1
            if produced % out_q.maxsize == 0:
                log.info("[PRODUCER] [%s] produced=%d q=%d", source_name, produced, out_q.qsize())
    except Exception:
        log.exception("[PRODUCER] [%s] crashed", source_name)
    finally:
        out_q.put((source_name, STOP, STOP, STOP, STOP, STOP))
        log.info("[PRODUCER] [%s] finished produced=%d", source_name, produced)

def embed_worker(embed_fn: Callable[[str], tuple[bytes, int]], embed_q: queue.Queue, res_q: queue.Queue, cfg: dict, worker_id: int):
    max_chars = int(cfg.get("pipeline", {}).get("max_embed_chars", 1800))
    min_chars = int(cfg.get("pipeline", {}).get("min_embed_chars", 800))
    batch_size = int(cfg.get("threading", {}).get("embed_batch_size", 4))

    while True:
        first_job = embed_q.get()
        jobs = [first_job]
        try:
            if first_job is STOP:
                return
            while len(jobs) < batch_size:
                try:
                    nxt = embed_q.get_nowait()
                    if nxt is STOP:
                        embed_q.put(STOP)
                        break
                    jobs.append(nxt)
                except queue.Empty:
                    break

            prepared_jobs = []
            batch_texts = []
            for job in jobs:
                assert isinstance(job, EmbedJob)
                txt = job.text
                safe_txt = txt[:max_chars] if len(txt) > max_chars else txt
                safe_txt = defang_tables(safe_txt)
                prepared_jobs.append((job, safe_txt))

            results = embed_batch_resilient(embed_fn, prepared_jobs, min_chars, worker_id)

            for res in results:
                res_q.put(res)

        finally:
            for _ in jobs:
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
                src, url, title, text, last_modified, etag = doc_q.get(timeout=15)
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
                    do_embed=False, last_modified=last_modified,
                    etag=etag
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