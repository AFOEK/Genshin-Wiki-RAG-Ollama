from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))

REPO_ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = REPO_ROOT / "rag"

log = logging.getLogger(__name__)

from core.paths import resolve_db_path, resolve_faiss_dir
from utils.logging_setup import setup_logging


FAISS_REQUIRED = ("index.faiss", "ids.npy", "meta.json")
EMBEDDING_REQUIRED = ("chunk_ids.npy", "vectors.npy")

OPTIONAL_KAGGLE_FILES = (
    "chunks.jsonl",
    "config.yaml",
    "embedding_meta.json",
    "upload_manifest.json",
    "dataset-metadata.json",
)


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
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip().lower()
        log.info("[PULL] Kernel status raw: %s", output.strip())

        if "complete" in output:
            return "complete"
        if "running" in output:
            return "running"
        if "queued" in output:
            return "queued"
        if "error" in output:
            return "error"
        if "cancelacknowledged" in output:
            return "cancelled"

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
    log.info(
        "[PULL] Waiting for kernel %s (poll=%ds timeout=%ds)",
        kernel_slug,
        poll_interval_s,
        timeout_s,
    )

    while elapsed < timeout_s:
        status = get_kernel_status(kernel_slug)

        if status == "complete":
            log.info("[PULL] Kernel finished successfully")
            return True

        if status in ("error", "cancelled"):
            log.error("[PULL] Kernel ended with status: %s", status)
            return False

        if status in ("running", "queued"):
            log.info(
                "[PULL] Still %s — next check in %ds (elapsed=%ds)",
                status,
                poll_interval_s,
                elapsed,
            )
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

def find_artifact_dir(root: Path, required: tuple[str, ...]) -> Path:
    """
    Kaggle may download files directly into work-dir, or preserve a subfolder such
    as dataset_upload_with_faiss/. This function finds the directory containing
    the required artifact bundle.
    """
    root = root.resolve()

    if all((root / name).exists() for name in required):
        return root

    candidates: list[Path] = []
    for p in root.rglob(required[0]):
        parent = p.parent
        if all((parent / name).exists() for name in required):
            candidates.append(parent)

    if not candidates:
        found = "\n".join(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
        raise FileNotFoundError(
            f"Could not find required files {required} under {root}.\n"
            f"Files found:\n{found}"
        )

    candidates.sort(key=lambda p: len(p.relative_to(root).parts))
    return candidates[0]


def show_meta(output_dir: Path) -> None:
    meta_path = output_dir / "meta.json"

    if not meta_path.exists():
        log.warning("[WARN] meta.json not found")
        return

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        log.info("[INFO] Kaggle FAISS meta:")
        print(json.dumps(meta, indent=2))
    except Exception as e:
        log.warning("[WARN] failed reading meta.json: %s", e)


def show_available_files(output_dir: Path) -> None:
    log.info("[PULL] Artifact directory: %s", output_dir)

    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            size_gb = p.stat().st_size / (1024 ** 3)
            role = "optional"

            if p.name in FAISS_REQUIRED:
                role = "needed-for-faiss"
            elif p.name in EMBEDDING_REQUIRED:
                role = "needed-for-sqlite-embedding-import"

            log.info("[PULL] %-24s %8.3f GB  %s", p.name, size_gb, role)


def import_embeddings(
    db_path: Path,
    chunk_ids_path: Path,
    vectors_path: Path,
    batch_size: int = 2000,
) -> int:
    if not chunk_ids_path.exists():
        raise FileNotFoundError(chunk_ids_path)
    if not vectors_path.exists():
        raise FileNotFoundError(vectors_path)

    chunk_ids = np.load(str(chunk_ids_path), mmap_mode="r")
    vectors = np.load(str(vectors_path), mmap_mode="r")

    if len(chunk_ids) != len(vectors):
        raise RuntimeError(
            f"chunk_ids/vectors length mismatch: {len(chunk_ids)} != {len(vectors)}"
        )

    if vectors.ndim != 2:
        raise RuntimeError(f"vectors.npy must be 2D, got shape={vectors.shape}")

    dims = int(vectors.shape[1])
    total = int(vectors.shape[0])

    log.info(
        "[EMBED_IMPORT] Importing %d vectors, dims=%d, batch_size=%d",
        total,
        dims,
        batch_size,
    )

    conn = rw_connect(str(db_path))
    cur = conn.cursor()

    inserted_total = 0
    t0 = time.time()

    try:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            ids_block = chunk_ids[start:end]
            vec_block = np.asarray(vectors[start:end], dtype=np.float32)

            rows = [
                (int(cid), dims, vec_block[i].tobytes())
                for i, cid in enumerate(ids_block)
            ]

            cur.execute("BEGIN")
            cur.executemany(
                """
                INSERT OR REPLACE INTO embeddings(chunk_id, dims, vector)
                VALUES (?, ?, ?)
                """,
                rows,
            )
            conn.commit()

            inserted_total += len(rows)

            if inserted_total % max(batch_size * 10, 1) == 0 or inserted_total == total:
                elapsed = time.time() - t0
                rate = inserted_total / max(elapsed, 1e-9)
                log.info(
                    "[EMBED_IMPORT] inserted=%d/%d rate=%.0f vec/s",
                    inserted_total,
                    total,
                    rate,
                )

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return inserted_total


def replace_faiss_bundle(cfg: dict, artifact_dir: Path) -> Path:
    faiss_root = resolve_faiss_dir(cfg)
    current_dir = faiss_root / "current"
    tmp_dir = faiss_root / "tmp_from_kaggle"

    for name in FAISS_REQUIRED:
        if not (artifact_dir / name).exists():
            raise FileNotFoundError(f"Missing FAISS artifact: {artifact_dir / name}")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=False)

    for name in FAISS_REQUIRED:
        shutil.copy2(artifact_dir / name, tmp_dir / name)

    old_dir = current_dir.with_name(current_dir.name + ".old")

    if old_dir.exists():
        shutil.rmtree(old_dir)

    if current_dir.exists():
        current_dir.rename(old_dir)

    tmp_dir.rename(current_dir)
    return current_dir


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", default=str(RAG_DIR / "config.yaml"))
    ap.add_argument(
        "--kernel-slug",
        default=None,
        help="owner/kernel-slug. If omitted, uses kaggle.kernel_slug from config.yaml.",
    )
    ap.add_argument(
        "--work-dir",
        default=str(RAG_DIR / "kaggle_results" / "output"),
        help="Directory where Kaggle output is downloaded.",
    )

    ap.add_argument("--wait", action="store_true", help="Poll kernel until complete before pulling")
    ap.add_argument("--poll-interval", type=int, default=300)
    ap.add_argument("--kernel-timeout", type=int, default=18000)

    ap.add_argument(
        "--replace-faiss",
        action="store_true",
        help="Replace local faiss/current using index.faiss, ids.npy, meta.json.",
    )
    ap.add_argument(
        "--import-embeddings",
        action="store_true",
        help="Import vectors.npy + chunk_ids.npy into SQLite embeddings table.",
    )
    ap.add_argument(
        "--embedding-batch-size",
        type=int,
        default=2000,
        help="SQLite import batch size for --import-embeddings.",
    )

    ap.add_argument(
        "--clean-work-dir",
        action="store_true",
        help="Delete work-dir before pulling to avoid stale Kaggle output files.",
    )
    ap.add_argument(
        "--no-pull",
        action="store_true",
        help="Do not call kaggle kernels output; use existing files in work-dir.",
    )

    args = ap.parse_args()

    cfg = load_cfg(args.config)
    setup_logging(
        cfg.get("logging", {}).get("file"),
        cfg.get("logging", {}).get("level", "INFO"),
    )

    kernel_slug = args.kernel_slug or cfg.get("kaggle", {}).get("kernel_slug")
    if not kernel_slug and not args.no_pull:
        log.error("[PULL] kernel_slug not set — pass --kernel-slug or set kaggle.kernel_slug in config.yaml")
        sys.exit(1)

    db_path = resolve_db_path(cfg)
    output_root = Path(args.work_dir)

    if args.clean_work_dir and output_root.exists():
        log.info("[PULL] Removing old work-dir: %s", output_root)
        shutil.rmtree(output_root)

    output_root = ensure_dir(output_root)

    if not args.no_pull:
        assert kernel_slug is not None

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
                log.error("[PULL] Kernel status is '%s', not complete. Use --wait or --no-pull", status)
                sys.exit(1)

        pull_kernel_output(kernel_slug, output_root, force=True)
        log.info("[KAGGLE_PULL] Downloaded Kaggle notebook output into %s", output_root)
    else:
        log.info("[PULL] --no-pull set; using existing work-dir: %s", output_root)

    artifact_dir = find_artifact_dir(output_root, FAISS_REQUIRED)

    show_available_files(artifact_dir)
    show_meta(artifact_dir)

    if args.import_embeddings:
        embedding_dir = find_artifact_dir(output_root, EMBEDDING_REQUIRED)
        inserted = import_embeddings(
            db_path,
            embedding_dir / "chunk_ids.npy",
            embedding_dir / "vectors.npy",
            batch_size=args.embedding_batch_size,
        )
        log.info("[KAGGLE_PULL] Imported %d embeddings into %s", inserted, db_path)
    else:
        log.info("[INFO] skipped SQLite embedding import")

    if args.replace_faiss:
        current_dir = replace_faiss_bundle(cfg, artifact_dir)
        log.info("[KAGGLE_PULL] Replaced FAISS bundle at %s", current_dir)
    else:
        log.info("[INFO] skipped FAISS replacement")


if __name__ == "__main__":
    main()
