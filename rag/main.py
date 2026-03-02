import yaml, re, logging, queue, threading, argparse
from core.db import connect
from core.embed import embed
from core.paths import resolve_db_path
from core.faiss import build_faiss_from_sqlite

from utils.filters import Filters
from utils.audit import audit_integrity, audit_faiss_against_sqlite
from utils.logging_setup import setup_logging
from utils.thread import producer, ingest_consumer

from adapters.kqm import load_kqm_tcl_docs
from adapters.wiki import load_fandom_docs
from adapters.html import crawl_site

# TEST_DOCS = [
#     ("test", "local://xiangling", "Xiangling", "Xiangling is a Pyro polearm character. Guoba breathes fire.")
# ]

def parse_bool(x: str) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--DB_CRAWL", default="True")
    ap.add_argument("--DB_AUDIT", default="True")
    ap.add_argument("--FAISS_MIGRATE", default="False")
    ap.add_argument("--FAISS_AUDIT", default="False")
    ap.add_argument("--FAISS_OVERWRITE", default="False")
    args = ap.parse_args()

    do_crawl = parse_bool(args.DB_CRAWL)
    do_db_audit = parse_bool(args.DB_AUDIT)
    do_faiss_migrate = parse_bool(args.FAISS_migrate)
    do_faiss_audit = parse_bool(args.FAISS_AUDIT)
    faiss_overwrite = parse_bool(args.FAISS_overwrite)

    with open("rag/config.yaml") as f:
        cfg = yaml.safe_load(f)

    setup_logging(
        cfg.get("logging", {}).get("file"),
        cfg.get("logging", {}).get("level", "INFO")
    )
    log = logging.getLogger(__name__)
    log.info("[INFO] Logging initialized")
    db_path = resolve_db_path(cfg)
    log.info("[INFO] Database initialized at %s", db_path)

    def embed_fn(text):
        return embed(cfg["ollama"]["base_url"], cfg["ollama"]["embedding_model"], text)
    
    filters = Filters(cfg["filters"]["deny_url_regex"], cfg["filters"]["deny_text_regex"])
    deny_url_re = re.compile(cfg["filters"]["deny_url_regex"], re.I) if cfg["filters"].get("deny_url_regex") else None
    threading_cfg = cfg.get("threading", {})
    embed_queue_size = int(threading_cfg.get("embed_queue_size", 200))
    embed_workers = int(threading_cfg.get("embed_workers", 2))
    document_queue_size = int(threading_cfg.get("document_queue_size", 200))
    log.info(
        "[INFO] Setting up multi-threading: embed_queue=%d document_queue=%d workers=%d",
        embed_queue_size, document_queue_size, embed_workers
    )
    if do_crawl:
        q = queue.Queue(maxsize=document_queue_size)

        producers = []
        tier_map = {}
        weight_map = {}

        for s in cfg["sources"]:
            if not s.get("enabled", True):
                continue

            docs_iter = None
            name = s["name"]
            kind = s["kind"]
            tier_map[name] = s.get("tier", "primary")
            weight_map[name] = s.get("weight", 1.0)

            if kind == "github":
                docs_iter = load_kqm_tcl_docs(s)

            elif kind == "fandom_api":
                raw_max = s.get("max_pages", 200)
                max_pages = int(raw_max) if raw_max is not None else None
                rate = float(s.get("rate_limit_s", 1.0))
                docs_iter = load_fandom_docs(s, rate_limit_s=rate, max_pages=max_pages)
            
            elif kind == "honey_html":
                seeds = s.get("seeds", [])
                if not seeds:
                    log.warning(f"[WARN] No seed for {name}, skipping")
                    continue

                base_url = s["base_url"]
                rate = float(s.get("rate_limit_s", 1.0))
                raw_max = s.get("max_pages", 200)
                max_pages = int(raw_max) if raw_max is not None else None

                docs_iter = crawl_site(base_url, seeds, deny_url_re, rate_limit_s=rate, max_pages=max_pages)
            else:
                log.warning(f"[WARN] Not implemented kind={kind}, skipping")

            if docs_iter is None:
                continue
            t = threading.Thread(target=producer, args=(name, docs_iter, q))
            producers.append(t)

        t_ingest = threading.Thread(target=ingest_consumer, args=(len(producers), q, str(db_path), embed_fn, cfg, filters, tier_map, weight_map, embed_workers, embed_queue_size))
        t_ingest.start()
        for t in producers:
            t.start()

        for t in producers:
            t.join()
        q.join()
        t_ingest.join()

    if do_faiss_migrate:
        build_faiss_from_sqlite(cfg, overwrite=faiss_overwrite)

    if do_db_audit:
        conn = connect(str(db_path))
        log.info("[INFO] Audit starting")

        try:
            if do_db_audit:
                report = audit_integrity(
                    conn,
                    sample_chunks=1000,
                    sample_docs=1500,
                    max_orphan_failures=2500,
                    max_missing_embedding_failures=2500,
                )

                if report.failures:
                    log.error("[AUDIT] Audit failed with %d problems", len(report.failures))

                    by_reason = {}
                    for f in report.failures:
                        by_reason[f.reason] = by_reason.get(f.reason, 0) + 1

                    for reason, n in sorted(by_reason.items(), key=lambda kv: kv[1], reverse=True):
                        log.error("  %s: %d", reason, n)

                    for f in report.failures[:10]:
                        log.error("[AUDIT] Example failure: %s", f)

                    raise RuntimeError(f"[AUDIT] Audit failed: {len(report.failures)} problems")
                log.info("[AUDIT] SQLite integrity OK")

            if do_faiss_audit:
                frep = audit_faiss_against_sqlite(cfg, sample_self_test=200)
                if frep.failures:
                    log.error("[FAISS_AUDIT] failed with %d problems", len(frep.failures))

                    by_reason = {}
                    for msg in frep.failures:
                        reason = msg.split(":", 1)[0].strip()
                        by_reason[reason] = by_reason.get(reason, 0) + 1

                    for reason, n in sorted(by_reason.items(), key=lambda kv: kv[1], reverse=True):
                        log.error("  %s: %d", reason, n)

                    for msg in frep.failures[:20]:
                        log.error("[FAISS_AUDIT] Example failure: %s", msg)

                    raise RuntimeError(f"[FAISS_AUDIT] failed: {len(frep.failures)} problems")
                log.info(
                    "[FAISS_AUDIT] OK index_total=%d ids_total=%d sqlite_active=%d dims=%d",
                    frep.index_total, frep.ids_total, frep.sqlite_active_embeds, frep.dims
                )

            log.info("[AUDIT] All requested stages completed successfully")

        except Exception:
            log.exception("[AUDIT] terminated due to audit/migrate failures")
    
    # source_meta = {s["name"]: (s.get("tier","primary"), float(s.get("weight", 1.0))) for s in cfg.get("sources", [])}
    # for source, url, title, text in TEST_DOCS:
    #     tier, weight = source_meta.get(source, ("primary", 1.0))
    #     process_document(conn, embed_fn, cfg, source, url, title, text, tier=tier, weight=weight)

if __name__ == "__main__":
    main()