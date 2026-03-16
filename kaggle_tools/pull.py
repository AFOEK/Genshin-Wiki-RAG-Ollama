from __future__ import annotations
import argparse, json, shutil, sqlite3, subprocess
from pathlib import Path

import numpy as np
import yaml

from rag.core.paths import resolve_db_path, resolve_faiss_dir

def load_cfg(path: str = "rag/config.yaml") -> dict:
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