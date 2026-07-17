from __future__ import annotations

from functools import lru_cache
from sentence_transformers import SparseEncoder
from pathlib import Path
from scipy import sparse

import numpy as np
import logging
import torch
import json
import shutil

from utils.io import write_json_atomic
from .db import read_only_connect
from .paths import resolve_db_path, resolve_splade_dir, resolve_storage_root, resolve_cache_folder

log = logging.getLogger(__name__)

@lru_cache(maxsize=2)
def load_splade_model(model_name: str, *, device: str, max_length: int, max_active_dims: int | None, cache_folder: str | None = None, precision: str = "fp32") -> SparseEncoder:
    model = SparseEncoder(model_name, device=device, cache_folder=cache_folder, max_active_dims=max_active_dims)
    model.max_seq_length = max_length
    if device.startswith("cuda"):
        if precision == "fp16":
            model.half()
        elif precision == "bf16":
            model.bfloat16()
    model.eval()
    return model

def get_splade_output_dimension(model: SparseEncoder) -> int:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        return int(len(tokenizer))

    transformer = getattr(model, "transformers_model", None)
    config = getattr(transformer, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)

    probe = model.encode_document(["dimension probe"], batch_size=1, show_progress_bar=False, convert_to_tensor=True, convert_to_sparse_tensor=True, save_to_cpu=True)
    return int(probe.shape[-1])

def encode_documents_to_csc(model: SparseEncoder, texts: list[str], *, batch_size: int, max_active_dims: int | None) -> sparse.csc_matrix:
    with torch.inference_mode():
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
        shutil.rmtree(current_dir)

    current_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(splade_cfg["model"])
    requested_device = str(splade_cfg.get("device", "auto")).strip().lower()
    if requested_device == "auto":
        device = ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = requested_device
    batch_size = int(splade_cfg.get("batch_size", 4))
    shard_size = int(splade_cfg.get("shard_size", 50_000))
    max_length = int(splade_cfg.get("max_length", 256))
    raw_active_dims = splade_cfg.get("max_active_dims", 128)
    precision = str(splade_cfg.get("precision", "fp32")).strip().lower()
    max_active_dims = (int(raw_active_dims) if raw_active_dims is not None else None)
    encode_block_size = max(batch_size, int(splade_cfg.get("encode_block_size", 512)))

    model = load_splade_model(model_name, device=device, max_length=max_length, max_active_dims=max_active_dims, cache_folder=resolve_cache_folder(cfg), precision=precision)
    vocabulary_size = get_splade_output_dimension(model)
    parameter = next(model.parameters())
    parameter = next(model.parameters())
    log.info("[SPLADE] vocabulary=%d hidden=%s device=%s dtype=%s batch=%d block=%d", vocabulary_size, getattr(model.transformers_model.config, "hidden_size", None), parameter.device, parameter.dtype, batch_size, encode_block_size)
    
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
            raise RuntimeError("SPLADE manifest mismatch: " f"{key}={actual_value!r}, " f"expected={expected_value!r}. " "Rebuild with overwrite=True.")

        conn = read_only_connect(str(db_path))
        built_this_run = 0
        exhausted = False

    try:
        while limit is None or built_this_run < limit:
            target_rows = (shard_size if limit is None else min(shard_size, limit - built_this_run))
            if target_rows <= 0:
                break

            shard_matrices = []
            shard_chunk_ids: list[int] = []
            last_chunk_id = int(manifest["last_chunk_id"])

            while len(shard_chunk_ids) < target_rows:
                fetch_size = min(encode_block_size, target_rows - len(shard_chunk_ids))
                rows = _fetch_active_chunks(conn, after_chunk_id=last_chunk_id, limit=fetch_size)

                if not rows:
                    exhausted = True
                    break

                texts = [(f"{row['title'] or ''}\n{row['text'] or ''}").strip() or "[empty chunk]" for row in rows]
                batch_matrix = encode_documents_to_csc(model, texts, batch_size=batch_size, max_active_dims=max_active_dims)
                if batch_matrix.shape[1] != vocabulary_size:
                    raise RuntimeError(
                        "SPLADE vocabulary mismatch: "
                        f"matrix={batch_matrix.shape[1]} "
                        f"model={vocabulary_size}"
                    )

                batch_ids = [int(row["chunk_id"]) for row in rows]
                shard_matrices.append(batch_matrix)
                shard_chunk_ids.extend(batch_ids)
                last_chunk_id = batch_ids[-1]
                log.info("[SPLADE] encoded run=%d " "current_shard=%d/%d " "last_chunk_id=%d", built_this_run + len(shard_chunk_ids), len(shard_chunk_ids), target_rows, last_chunk_id,)

            if not shard_chunk_ids:
                break

            shard_matrix = sparse.vstack(shard_matrices, format="csc", dtype=np.float32,)
            chunk_ids = np.asarray(shard_chunk_ids, dtype=np.int64,)
            shard_number = int(manifest["shard_count"])
            shard_name = (f"shard_{shard_number:05d}")
            temporary_dir = (current_dir / f".{shard_name}.tmp")
            final_dir = (current_dir / shard_name)

            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)

            if final_dir.exists():
                shutil.rmtree(final_dir)

            save_csc_shard(temporary_dir, shard_matrix, chunk_ids,)
            temporary_dir.replace(final_dir)
            manifest["last_chunk_id"] = (last_chunk_id)
            manifest["chunk_count"] = (int(manifest["chunk_count"]) + len(shard_chunk_ids))
            manifest["shard_count"] = (shard_number + 1)
            manifest["completed"] = exhausted
            
            write_json_atomic(manifest_path, manifest)
            built_this_run += len(shard_chunk_ids)
            average_active_dims = (shard_matrix.nnz / max(shard_matrix.shape[0], 1))
            log.info("[SPLADE] saved shard=%d rows=%d nnz=%d avg_active_dims=%.2f", shard_number, shard_matrix.shape[0], shard_matrix.nnz, average_active_dims,)

            del shard_matrix
            del shard_matrices

            if exhausted:
                break
    finally:
        conn.close()

    log.info("[SPLADE] build finished built_this_run=%d total=%d shards=%d completed=%s", built_this_run, manifest["chunk_count"], manifest["shard_count"], manifest["completed"])
    return dict(manifest)

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

