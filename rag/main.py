import yaml, re, logging, queue, threading

from tqdm import tqdm

from core.db import connect
from core.embed import embed
from core.paths import resolve_db_path

from utils.filters import Filters
from utils.audit import audit_integrity
from utils.logging_setup import setup_logging
from utils.thread import producer, ingest_consumer

from adapters.kqm import load_kqm_tcl_docs
from adapters.wiki import load_fandom_docs
from adapters.html import crawl_site

# TEST_DOCS = [
#     ("test", "local://xiangling", "Xiangling", "Xiangling is a Pyro polearm character. Guoba breathes fire.")
# ]


def main():
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

    log.info("[INFO] Setting up multi-threading")
    q = queue.Queue(maxsize=200)

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
                tqdm.write(f"[WARN] No seed for {name}, skipping")
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
        t = threading.Thread(target=producer, args=(name, docs_iter, q), daemon=True)
        producers.append(t)
    

    t_ingest = threading.Thread(target=ingest_consumer, args=(len(producers), q, str(db_path), embed_fn, cfg, filters, tier_map, weight_map), daemon=True)
    t_ingest.start()
    for t in producers:
        t.start()

    for t in producers:
        t.join()
    q.join()
    t_ingest.join()

    conn = connect(str(db_path))
    log.info("[INFO] Audit starting")

    try:
        report = audit_integrity(conn, sample_chunks=1000, sample_docs=1500, max_orphan_failures = 2500, max_missing_embedding_failures = 2500)
        if report.failures:
            log.error(f"Audit failed with {len(report.failures)} problems")
            by_reason = {}
            for f in report.failures:
                by_reason[f.reason] = by_reason.get(f.reason, 0) + 1
            for reason, n in sorted(by_reason.items(), key=lambda kv: kv[1], reverse=True):
                log.error(f"  {reason}: {n}")

            for f in report.failures[:10]:
                log.error(f"Example failure: {f}")

            raise RuntimeError(f"Audit failed: {len(report.failures)} problems")

        log.info("All documents have been processed!")
    except Exception:
        log.exception("Pipeline terminated due to audit failures")
    
    # source_meta = {s["name"]: (s.get("tier","primary"), float(s.get("weight", 1.0))) for s in cfg.get("sources", [])}
    # for source, url, title, text in TEST_DOCS:
    #     tier, weight = source_meta.get(source, ("primary", 1.0))
    #     process_document(conn, embed_fn, cfg, source, url, title, text, tier=tier, weight=weight)

    # tqdm.write("Pipeline ok")

if __name__ == "__main__":
    main()