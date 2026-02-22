import yaml, re, logging

from tqdm import tqdm

from core.db import connect
from core.embed import embed
from core.pipeline import process_document

from utils.filters import Filters
from utils.audit import audit_integrity
from utils.logging_setup import setup_logging

from adapters.kqm import load_kqm_tcl_docs
from adapters.wiki import load_fandom_docs
from adapters.html import crawl_site

# TEST_DOCS = [
#     ("test", "local://xiangling", "Xiangling", "Xiangling is a Pyro polearm character. Guoba breathes fire.")
# ]

def safe_process(conn, embed_fn, cfg, name, url, title, text, tier, weight, log):
    try:
        process_document(conn, embed_fn, cfg, name, url, title, text, tier=tier, weight=weight)
        return True
    except Exception:
        log.exception("process_document failed source=%s title=%s url=%s", name, title, url)
        return False

def main():
    with open("rag/config.yaml") as f:
        cfg = yaml.safe_load(f)

    setup_logging(
        cfg.get("logging", {}).get("file"),
        cfg.get("logging", {}).get("level", "INFO")
    )
    log = logging.getLogger(__name__)
    log.info("Logging initialized")
    conn = connect(cfg["db_path"])

    def embed_fn(text):
        return embed(cfg["ollama"]["base_url"], cfg["ollama"]["embedding_model"], text)
    
    filters = Filters(cfg["filters"]["deny_url_regex"], cfg["filters"]["deny_text_regex"])
    deny_url_re = re.compile(cfg["filters"]["deny_url_regex"], re.I) if cfg["filters"].get("deny_url_regex") else None

    for s in cfg["sources"]:
        if not s.get("enabled", True):
            continue
        name = s["name"]
        kind = s["kind"]
        tier = s.get("tier", "primary")
        weight = float(s.get("weight", 1.0))

        log.info(f"Source {name} ({kind} tier={tier} weight={weight})")
        if kind == "github":
            docs_iter = load_kqm_tcl_docs(s)
            for url, title, text in tqdm(docs_iter, desc=f"{name}", unit="doc"):
                if not filters.url_allowed(url):
                    continue
                if not filters.text_allowed(text):
                    continue
                safe_process(conn, embed_fn, cfg, name, url, title, text, tier, weight, log)
                log.info(f"Processed documents successfully, name={name} with {title}, {url}, and {tier}")

        elif kind == "fandom_api":
            raw_max = s.get("max_pages", 200)
            max_pages = int(raw_max) if raw_max is not None else None
            rate = float(s.get("rate_limit_s", 1.0))
            docs_iter = load_fandom_docs(s, rate_limit_s=rate, max_pages=max_pages)
            for url, title, text in tqdm(docs_iter, desc=f"{name}", total=max_pages, unit="page"):
                if not filters.url_allowed(url):
                    continue
                if not filters.text_allowed(text):
                    continue
                safe_process(conn, embed_fn, cfg, name, url, title, text, tier, weight, log)
                log.info(f"Processed documents successfully, name={name} with {title}, {url}, and {tier}")
        
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
            for url, title, text in tqdm(docs_iter, desc=f"{name}", total=max_pages, unit="page"):
                if not filters.url_allowed(url):
                    continue
                if not filters.text_allowed(text):
                    continue
                safe_process(conn, embed_fn, cfg, name, url, title, text, tier, weight, log)
                log.info(f"Processed documents successfully, name={name} with {title}, {url}, and {tier}")
        else:
            log.warning(f"[WARN] Not implemented kind={kind}, skipping")

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