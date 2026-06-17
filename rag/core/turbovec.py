from __future__ import annotations

from pathlib import Path
from turbovec import IdMapIndex
import json
import logging
import sqlite3
import time

import numpy as np

from core.paths import resolve_db_path, resolve_turbovec_dir

log = logging.getLogger(__name__)

def normalize_model_name(x) -> str:
    if x is None:
        return ""
    
    if isinstance(x, (list, tuple, set)):
        x = next(iter(x), "")
    elif isinstance(x, dict):
        x = x.get("name") or x.get("model") or x.get("embedding_model") or ""

    s = str(x).strip().lower().replace("\\", "/")

    if s.endswith(":latest"):
        s = s[:-7]

    if s.startswith("/") or s.count("/") > 1:
        s = s.split("/")[-1]

    return s

def normalize_vector_batch(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D vector batch, got shape={vectors.shape}")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.ascontiguousarray(vectors / np.maximum(norms, 1.0e-12), dtype=np.float32)

def embedding_model_from_cfg(cfg: dict, *, backend: str | None = None, source: str = "runtime") -> str:
    source = str(source or "runtime").strip().lower()

    runtime = cfg.get("runtime", {}) or {}
    provider = (backend or runtime.get("embedding_provider", "ollama")).strip().lower()
    if provider == "llama.cpp":
        provider = "llamacpp"

    if source == "kaggle":
        return str(cfg.get("kaggle", {}).get("embedding_model", ""))

    if source == "ollama":
        return str(cfg.get("ollama", {}).get("embedding_model", ""))

    if source in ("llamacpp", "llama.cpp"):
        return str(cfg.get("llamacpp", {}).get("embedding_model", ""))

    if provider == "llamacpp":
        return str(cfg.get("llamacpp", {}).get("embedding_model", ""))

    return str(cfg.get("ollama", {}).get("embedding_model", ""))

def build_turbovec_from_sqlite(cfg: dict, *, overwrite: bool = False, backend: str | None = None) -> dict:
    db_path = resolve_db_path(cfg)
    tv_cfg = cfg.get("turbovec", {}) or {}
    
    out_dir = resolve_turbovec_dir(cfg)
    current = out_dir / "current"
    current.mkdir(parents=True, exist_ok=True)

    index_path = current / "index.tvim"
    meta_path = current / "meta.json"
    ids_path = current / "ids.npy"

    if index_path.exists() and not overwrite:
        raise FileExistsError(f"TurboVec index already exists: {index_path}")
    
    bit_width = int(tv_cfg.get("bit_width", 4))
    batch_size = int(tv_cfg.get("batch_size", 4096))
    calibration_size = max(batch_size, int(tv_cfg.get("calibration_size", 50000)))

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        cur.execute("""
            SELECT COUNT(*)
            FROM embeddings e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            JOIN docs d ON d.doc_id = c.doc_id
            WHERE c.is_active = 1
            AND COALESCE(d.status, 1) = 1
        """)
        total = int(cur.fetchone()[0] or 0)

        if total == 0:
            conn.close()
            raise RuntimeError("No active embeddings found for TurboVec build")
        
        cur.execute("""
            SELECT e.dims
            FROM embeddings e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            JOIN docs d ON d.doc_id = c.doc_id
            WHERE c.is_active = 1
            AND COALESCE(d.status, 1) = 1
            LIMIT 1
        """)
        dims = int(cur.fetchone()["dims"])

        log.info("[TURBOVEC] Build starting total=%d, dims=%d, bit_width=%d, batch_size=%d", total, dims, bit_width, batch_size)

        index = IdMapIndex(dim=dims, bit_width=bit_width)

        cur.execute("""
            SELECT e.chunk_id, e.dims, e.vector
            FROM embeddings e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            JOIN docs d ON d.doc_id = c.doc_id
            WHERE c.is_active = 1
            AND COALESCE(d.status, 1) = 1
            ORDER BY ((e.chunk_id * 1103515245 + 12345) & 2147483647)
        """)

        ids_batch: list[int] = []
        vecs_batch: list[np.ndarray] = []

        added = 0
        added_id_batches: list[np.ndarray] = []

        def flush_batch() -> None:
            nonlocal added, ids_batch, vecs_batch
            if not ids_batch:
                return

            vectors = normalize_vector_batch(np.stack(vecs_batch, axis=0))
            ids = np.ascontiguousarray(ids_batch, dtype=np.uint64)

            index.add_with_ids(vectors, ids)
            added_id_batches.append(ids.copy())
            added += int(ids.size)

            if added % (batch_size * 10) == 0 or added == total:
                log.info("[TURBOVEC] added=%d/%d", added, total)

            ids_batch = []
            vecs_batch = []

        for row in cur:
            cid = int(row["chunk_id"])
            d = int(row["dims"])
            if d != dims:
                raise RuntimeError(f"mixed embedding dims: expected={dims}, got={d}, chunk_id={cid}")

            v = np.frombuffer(row["vector"], dtype=np.float32)
            if v.size != dims:
                raise RuntimeError(f"bad vector size chunk_id={cid}: expected={dims}, got={v.size}")

            ids_batch.append(cid)
            vecs_batch.append(v)

            target_batch_size = calibration_size if added == 0 else batch_size
            if len(ids_batch) >= target_batch_size:
                flush_batch()

        flush_batch()
    finally:
        conn.close()
        
    index.write(str(index_path))
    indexed_ids = np.concatenate(added_id_batches) if added_id_batches else np.empty(0, dtype=np.uint64)

    if indexed_ids.size != added:
        raise RuntimeError(f"TurboVec indexed ID count mismatch: added={added}, ids={indexed_ids.size}")

    np.save(ids_path, indexed_ids)

    model_source = str(tv_cfg.get("model_source", "runtime"))
    embedding_model = embedding_model_from_cfg(cfg, backend=backend, source=model_source)

    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "index_type": "turbovec_idmap",
        "path": str(index_path),
        "ids_path": str(ids_path),
        "dims": int(dims),
        "count": int(added),
        "bit_width": int(bit_width),
        "calibration_size": int(calibration_size),
        "embedding_model": embedding_model,
        "model_source": model_source,
        "normalized": True,
        "metric": "cosine",
        "db_path": str(db_path),
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info(
        "[TURBOVEC] build done added=%d path=%s model=%s",
        added,
        index_path,
        embedding_model,
    )

    return meta    