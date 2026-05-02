from __future__ import annotations
import argparse, json, shutil, sqlite3, subprocess, logging
from pathlib import Path

import numpy as np
import yaml, sys, time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))
REPO_ROOT = Path(__file__).resolve().parents[1]
RAG_DIR   = REPO_ROOT / "rag"

log = logging.getLogger(__name__)

from core.paths import resolve_db_path, resolve_faiss_dir
from utils.logging_setup import setup_logging

def load_cfg(path: str | None = None) -> dict:
    p = Path(path).resolve() if path else RAG_DIR / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def rw_connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=60.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=100000;")
    return conn

def get_kernel_status(kernel_slug: str) -> str | None:
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_slug],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout.strip().lower()
        log.info("[PULL] Kernel status raw: %s", output.strip())
        if "complete"            in output: return "complete"
        if "running"             in output: return "running"
        if "queued"              in output: return "queued"
        if "error"               in output: return "error"
        if "cancelacknowledged"  in output: return "cancelled"
        return output
    except Exception as e:
        log.warning("[PULL] Status check failed: %s", e)
        return None
    
def wait_for_kernel(
    kernel_slug: str,
    poll_interval_s: int = 300,
    timeout_s: int = 18000,
) -> bool:
    elapsed = 0
    log.info("[PULL] Waiting for kernel %s (poll=%ds timeout=%ds)",
             kernel_slug, poll_interval_s, timeout_s)

    while elapsed < timeout_s:
        status = get_kernel_status(kernel_slug)

        if status == "complete":
            log.info("[PULL] Kernel finished successfully")
            return True
        if status in ("error", "cancelled"):
            log.error("[PULL] Kernel ended with status: %s", status)
            return False
        if status in ("running", "queued"):
            log.info("[PULL] Still %s — next check in %ds (elapsed=%ds)",
                     status, poll_interval_s, elapsed)
        else:
            log.warning("[PULL] Unknown status '%s' — will retry", status)

        time.sleep(poll_interval_s)
        elapsed += poll_interval_s

    log.error("[PULL] Timed out after %ds waiting for kernel", timeout_s)
    return False

def pull_kernel_output(kernel_slug: str, out_dir: Path, force: bool = True) -> None:
    ensure_dir(out_dir)
    cmd = ["kaggle", "kernels", "output", kernel_slug, "-p", str(out_dir)]
    if force:
        cmd.append("-o")
    run(cmd)

def import_embeddings(db_path: Path, chunks_ids_path: Path, vectors_path: Path) -> int:
    if not chunks_ids_path.exists():
        raise FileNotFoundError(chunks_ids_path)
    if not vectors_path.exists():
        raise FileNotFoundError(vectors_path)
    
    chunk_ids = np.load(str(chunks_ids_path))
    vectors = np.load(str(vectors_path))

    if len(chunk_ids) != len(vectors):
        raise RuntimeError(f"chunk_ids/vectors length mismatch: {len(chunk_ids)} != {len(vectors)}")

    conn = rw_connect(str(db_path))
    cur = conn.cursor()
    inserted = 0
    cur.execute("BEGIN")
    try:
        rows = [(int(cid), int(vec.shape[0]), vec.astype(np.float32).tobytes()) for cid, vec in zip(chunk_ids, vectors)]
        cur.executemany("INSERT OR REPLACE INTO embeddings(chunk_id, dims, vector) VALUES (?, ?, ?)", rows)
        inserted = cur.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return inserted

def replace_faiss_bundle(cfg: dict, output_dir: Path) -> Path:
    faiss_root = resolve_faiss_dir(cfg)
    current_dir = faiss_root / "current"
    tmp_dir = faiss_root / "tmp_from_kaggle"

    required = ("index.faiss", "ids.npy", "meta.json")
    for name in required:
        if not (output_dir / name).exists():
            raise FileNotFoundError(f"Missing FAISS artifact: {output_dir / name}")
    
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    for name in required:
        shutil.copy2(output_dir / name, tmp_dir / name)
    old_dir = current_dir.with_name(current_dir.name + ".old")
    if old_dir.exists():
        shutil.rmtree(old_dir)
    if current_dir.exists():
        current_dir.rename(old_dir)

    tmp_dir.rename(current_dir)
    return current_dir

def show_meta(output_dir: Path) -> None:
    meta_path = output_dir / "meta.json"
    if not meta_path.exists():
        log.warning("[WARN] meta.json not found")
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        log.info("[INFO] Kaggle output meta:")
        print(json.dumps(meta, indent=2))
    except Exception as e:
        log.warning("[WARN] failed reading meta.json: %s", {e})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default=str(RAG_DIR / "config.yaml"))
    ap.add_argument("--kernel-slug", required=True, help="owner/kernel-slug")
    ap.add_argument("--work-dir", default=str(RAG_DIR / "kaggle_results" / "output"))
    ap.add_argument("--replace-faiss", action="store_true")
    ap.add_argument("--wait", action="store_true", help="Poll kernel until complete before pulling")
    ap.add_argument("--poll-interval", type=int, default=300, help="Seconds between status checks (default: 300 = 5min)")
    ap.add_argument("--kernel-timeout", type=int, default=18000, help="Max seconds to wait (default: 18000 = 5hrs)")
    ap.add_argument("--skip-import-embeddings", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    setup_logging(
        cfg.get("logging", {}).get("file"),
        cfg.get("logging", {}).get("level", "INFO")
    )

    kernel_slug = args.kernel_slug or cfg.get("kaggle", {}).get("kernel_slug")
    if not kernel_slug:
        log.error("[PULL] kernel_slug not set — pass --kernel-slug or set kaggle.kernel_slug in config.yaml")
        sys.exit(1)

    db_path = resolve_db_path(cfg)
    output_dir = ensure_dir(Path(args.work_dir))

    if args.wait:
        ok = wait_for_kernel(
            kernel_slug,
            poll_interval_s=args.poll_interval,
            timeout_s=args.kernel_timeout,
        )
        if not ok:
            log.error("[PULL] Kernel did not complete — aborting")
            sys.exit(1)
    else:
        status = get_kernel_status(kernel_slug)
        if status != "complete":
            log.error("[PULL] Kernel status is '%s', not complete. Use --wait or --force", status)
            sys.exit(1)

    pull_kernel_output(kernel_slug, output_dir, force=True)
    log.info("[KAGGLE_PULL] downloaded Kaggle notebook output into %s", output_dir)

    show_meta(output_dir)

    if not args.skip_import_embeddings:
        inserted = import_embeddings(
            db_path,
            output_dir / "chunk_ids.npy",
            output_dir / "vectors.npy",
        )
        log.info("[KAGGLE_PULL] imported %d embeddings into %s", inserted, db_path)
    else:
        log.info("[INFO] skipped SQLite embedding import")

    if args.replace_faiss:
        current_dir = replace_faiss_bundle(cfg, output_dir)
        log.info("[KAGGLE_PULL] replaced FAISS bundle at %s", current_dir)
    else:
        log.info("[INFO] skipped FAISS replacement")

if __name__ == "__main__":
    main()