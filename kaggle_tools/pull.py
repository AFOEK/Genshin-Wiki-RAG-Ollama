from __future__ import annotations
import argparse, json, shutil, sqlite3, subprocess, logging
from pathlib import Path

import numpy as np
import yaml, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))

from core.paths import resolve_db_path, resolve_faiss_dir

log = logging.getLogger(__name__)

def load_cfg(path: str = "../rag/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
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
        for cid, vec in zip(chunk_ids, vectors):
            vec = np.asarray(vec, dtype=np.float32)
            cur.execute(
                "INSERT OR REPLACE INTO embeddings(chunk_id, dims, vector) VALUES (?, ?, ?)",
                (int(cid), int(vec.shape[0]), vec.tobytes()),
            )
            inserted += 1
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
        src = output_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing FAISS artifacts: {src}")
    
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
        log.warning(f"[WARN] failed reading meta.json: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../rag/config.yaml")
    ap.add_argument("--kernel-slug", required=True, help="owner/kernel-slug")
    ap.add_argument("--work-dir", default="../rag/kaggle_results/output")
    ap.add_argument("--skip-import-embeddings", action="store_true")
    ap.add_argument("--replace-faiss", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    db_path = resolve_db_path(cfg)
    output_dir = ensure_dir(Path(args.work_dir))

    pull_kernel_output(args.kernel_slug, output_dir, force=True)
    log.info(f"[OK] downloaded Kaggle notebook output into {output_dir}")

    show_meta(output_dir)

    if not args.skip_import_embeddings:
        inserted = import_embeddings(
            db_path,
            output_dir / "chunk_ids.npy",
            output_dir / "vectors.npy",
        )
        log.info(f"[OK] imported {inserted} embeddings into {db_path}")
    else:
        log.info("[INFO] skipped SQLite embedding import")

    if args.replace_faiss:
        current_dir = replace_faiss_bundle(cfg, output_dir)
        log.info(f"[OK] replaced FAISS bundle at {current_dir}")
    else:
        log.info("[INFO] skipped FAISS replacement")

if __name__ == "__main__":
    main()