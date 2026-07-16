from __future__ import annotations

from functools import lru_cache
from sentence_transformers import SparseEncoder
from pathlib import Path
from scipy import sparse

import numpy as np
import logging
import torch
import json

from utils.io import write_json_atomic
from .db import connect
from .paths import resolve_db_path, resolve_splade_dir, resolve_storage_root

log = logging.getLogger(__name__)

@lru_cache(maxsize=2)
def load_splade_model(model_name: str, *, device: str, max_length: int, max_active_dims: int | None, cache_folder: str | None = None) -> SparseEncoder:
    model = SparseEncoder(model_name, device=device, cache_folder=cache_folder, max_active_dims=max_active_dims)
    model.max_seq_length = max_length
    model.eval()
    return model

def encode_documents_to_csc(model: SparseEncoder, texts: list[str], *, batch_size: int, max_active_dims: int | None) -> sparse.csc_matrix:
    embeddings = model.encode_document(texts, batch_size=batch_size, show_progress_bar=False, convert_to_tensor=True, convert_to_sparse_tensor=True, save_to_cpu=True, max_active_dims=max_active_dims,)
    embeddings = embeddings.coalesce()
    coordinates = embeddings.indices().cpu().numpy()
    values = (embeddings.values().float().cpu().numpy().astype(np.float32, copy=False))
    matrix = sparse.coo_matrix((values, (coordinates[0], coordinates[1],),), shape=tuple(embeddings.shape), dtype=np.float32,)
    return matrix.tocsc()

def encode_query_sparse(model: SparseEncoder, query: str, *, max_active_dims: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    embedding = model.encode_query(query, show_progress_bar=False, convert_to_tensor=True, convert_to_sparse_tensor=True, save_to_cpu=True, max_active_dims=max_active_dims,)
    embedding = embedding.coalesce()
    if embedding.ndim != 1:
        raise ValueError(
            "Expected one-dimensional SPLADE query vector, "
            f"got shape={tuple(embedding.shape)}"
        )

    indices = (embedding.indices()[0].cpu().numpy().astype(np.int32, copy=False))
    values = (embedding.values().float().cpu().numpy().astype(np.float32, copy=False))
    return indices, values, int(embedding.shape[0])

def save_csc_shard(shard_dir: Path, matrix: sparse.csc_matrix, chunk_ids: np.ndarray) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    matrix = matrix.astype(np.float32, copy=False)
    matrix.sort_indices()
    if matrix.shape[0] != chunk_ids.size:
        raise ValueError(
            "SPLADE shard row count does not match chunk ID count: "
            f"rows={matrix.shape[0]} ids={chunk_ids.size}")

    np.save(shard_dir / "data.npy", matrix.data)
    np.save(shard_dir / "indices.npy", matrix.indices)
    np.save(shard_dir / "indptr.npy", matrix.indptr)
    np.save(shard_dir / "chunk_ids.npy", chunk_ids.astype(np.int64, copy=False))

    metadata = {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
    }
    (shard_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("[SPLADE] CSC matrix saved successfully at: %s", str(shard_dir))

def load_csc_shard(shard_dir: Path) -> tuple[sparse.csc_matrix, np.ndarray, dict]:
    metadata = json.loads((shard_dir / "meta.json").read_text(encoding="utf-8"))

    data = np.load(shard_dir / "data.npy", mmap_mode="r", allow_pickle=False)
    indices = np.load(shard_dir / "indices.npy", mmap_mode="r", allow_pickle=False)
    indptr = np.load(shard_dir / "indptr.npy", mmap_mode="r", allow_pickle=False)
    chunk_ids = np.load(shard_dir / "chunk_ids.npy", mmap_mode="r", allow_pickle=False)
    matrix = sparse.csc_matrix((data, indices, indptr), shape=(int(metadata["rows"]), int(metadata["columns"])), copy=False)

    return matrix, chunk_ids, metadata

def search_csc_shard(matrix: sparse.csc_matrix, chunk_ids: np.ndarray, query_indices: np.ndarray, query_values: np.ndarray, *, k: int) -> list[tuple[int, float]]:
    if k <= 0 or query_indices.size == 0:
        return []
    valid = ((query_indices >= 0) & (query_indices < matrix.shape[1]))

    query_indices = query_indices[valid]
    query_values = query_values[valid]

    if query_indices.size == 0:
        return []
    
    selected_columns = matrix[:, query_indices]
    scores = np.asarray(selected_columns @ query_values, dtype=np.float32).reshape(-1)
    positive_rows = np.flatnonzero(scores > 0.0)
    if positive_rows.size == 0:
        return []

    effective_k = min(k, positive_rows.size)

    if positive_rows.size > effective_k:
        selected = positive_rows[np.argpartition(scores[positive_rows], -effective_k)[-effective_k:]]
    else:
        selected = positive_rows

    selected = selected[np.argsort(scores[selected])[::-1]]

    return [(int(chunk_ids[row_index]), float(scores[row_index])) for row_index in selected]

def build_splade_from_sqlite(cfg: dict, *, overwrite: bool = False, limit: int | None = None) -> dict:
    splade_cfg = cfg.get("splade", {}) or {}

    if not splade_cfg.get("enabled", False):
        raise RuntimeError("SPLADE is disabled in config")

    db_path = resolve_db_path(cfg)
    splade_dir = resolve_splade_dir(cfg)
    current_dir = splade_dir / "current"
    manifest_path = (current_dir / "manifest.json")
    if overwrite and current_dir.exists():
        import shutil
        shutil.rmtree(current_dir)

    current_dir.mkdir(
        parents=True, exist_ok=True)
    model_name = str(splade_cfg["model"])
    device = str(splade_cfg.get("device", "cpu")) or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = int(splade_cfg.get("batch_size", 4))
    shard_size = int(splade_cfg.get("shard_size", 50_000))
    max_length = int(splade_cfg.get("max_length", 256))
    raw_active_dims = splade_cfg.get("max_active_dims", 128)
    max_active_dims = (int(raw_active_dims) if raw_active_dims is not None else None)

    model = load_splade_model(model_name, device=device, max_length=max_length, max_active_dims=max_active_dims, cache_folder=_resolve_cache_folder(cfg))
    vocabulary_size = int(model.get_embedding_dimension())
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "format": (
                "scipy_csc_memmap_v1"
            ),
            "model": model_name,
            "vocabulary_size": (
                vocabulary_size
            ),
            "max_length": max_length,
            "max_active_dims": (
                max_active_dims
            ),
            "shard_size": shard_size,
            "last_chunk_id": 0,
            "chunk_count": 0,
            "shard_count": 0,
            "completed": False,
        }

        write_json_atomic(manifest_path, manifest)
    
    expected = {
        "model": model_name,
        "vocabulary_size": (vocabulary_size),
        "max_length": max_length,
        "max_active_dims": (max_active_dims)}

    for key, expected_value in expected.items():
        actual_value = manifest.get(key)
        if actual_value != expected_value:
            raise RuntimeError(
                "SPLADE manifest mismatch: "
                f"{key}={actual_value!r}, "
                f"expected={expected_value!r}. "
                "Rebuild with overwrite=True.")

def _fetch_active_chunks(conn, *, after_chunk_id: int, limit: int):
    return conn.execute(
        """
        SELECT
            c.chunk_id,
            d.title,
            c.text
        FROM chunks c
        JOIN docs d
          ON d.doc_id = c.doc_id
        WHERE c.is_active = 1
          AND c.chunk_id > ?
        ORDER BY c.chunk_id
        LIMIT ?
        """,
        (int(after_chunk_id), int(limit))).fetchall()

def _resolve_cache_folder(cfg: dict) -> str | None:
    splade_cfg = cfg.get("splade", {}) or {}
    raw_value = splade_cfg.get("cache_folder")
    if not raw_value:
        return None
    cache_path = Path(str(raw_value)).expanduser()

    if not cache_path.is_absolute():
        cache_path = (resolve_storage_root(cfg) / cache_path)

    cache_path.mkdir(parents=True, exist_ok=True)
    return str(cache_path.resolve())

