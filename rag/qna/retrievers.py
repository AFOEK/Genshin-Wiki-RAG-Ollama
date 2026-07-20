from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from turbovec import IdMapIndex
import faiss
import logging
import threading
import numpy as np

from .utils import normalize_vec_from_blob, make_fts5_query, normalize_model_name, check_faiss_model_match
from core.splade import encode_query_sparse, load_csc_shard, load_splade_model, search_csc_shard

log = logging.getLogger(__name__)

faiss_retriever_cache: dict[str, FaissRetriever] = {}

splade_retriever_cache = {}
splade_model_locks = {}
splade_model_locks_guard = threading.Lock()

class FaissRetriever:
    def __new__(cls, faiss_dir: Path, *, expected_model: str | None = None, mismatch_policy:str = "error"):
        key = str(faiss_dir)
        if key not in faiss_retriever_cache:
            instance = super().__new__(cls)
            instance._initialized = False
            faiss_retriever_cache[key] = instance
        return faiss_retriever_cache[key]

    def __init__(self, faiss_dir: Path, *, expected_model: str | None = None, mismatch_policy: str = "error"):
        if self._initialized:
            if expected_model:
                check_faiss_model_match(actual_model=self.model, expected_model=expected_model, policy=mismatch_policy)
            return
        current = faiss_dir / "current"
        self.index_path = current / "index.faiss"
        self.ids_path = current / "ids.npy"
        self.meta_path = current / "meta.json"

        if not (self.index_path.exists() and self.ids_path.exists() and self.meta_path.exists()):
            raise FileNotFoundError(f"FAISS bundle missing under {current}")

        self.index = faiss.read_index(str(self.index_path))
        self.ids = np.load(str(self.ids_path))
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.dims = int(self.meta["dims"])
        self.model = self.meta.get("embedding_model", "unknown")
        if expected_model:
            check_faiss_model_match(actual_model=self.model, expected_model=expected_model, policy=mismatch_policy)
        log.info("[FAISS] loaded index dims=%d model=%s ntotal=%d",
             self.dims, self.model, self.index.ntotal)
        self._initialized = True

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, self.index.ntotal)
        if k == 0:
            return []
        dists, indices = self.index.search(query_vec, k)
        out = []
        for i, score in zip(indices[0], dists[0]):
            if i < 0:
                continue
            out.append((int(self.ids[i]), float(score)))
        return out

class SqliteEmbeddingRetriever:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        cur = conn.cursor()
        cur.execute("""
            SELECT e.dims
            FROM embeddings e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            WHERE c.is_active=1
            LIMIT 1
        """)
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("No active embeddings found in SQLite")
        self.dims = int(row["dims"])

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT e.chunk_id, e.dims, e.vector
            FROM embeddings e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            WHERE c.is_active=1
        """)

        scores: list[tuple[float, int]] = []
        q = query_vec[0]

        for row in cur:
            chunk_id = int(row["chunk_id"])
            dims = int(row["dims"])
            blob = row["vector"]
            v = normalize_vec_from_blob(blob, dims)
            score = float(np.dot(q, v))
            scores.append((score, chunk_id))

        scores.sort(key=lambda x: x[0], reverse=True)
        k = min(k, len(scores))
        return [(cid, score) for score, cid in scores[:k]]
    
class BM25Retriever:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.dims = None

    def _search_fts(self, fts_query: str, top_k: int, *, weights: tuple[float, float, float, float, float]):
        if not fts_query:
            return []
        
        chunk_id_w, doc_id_w, source_w, title_w, text_w = weights
        cur = self.conn.cursor()
        cur.execute("""
            SELECT
                chunk_id,
                -bm25(chunks_fts, ?, ?, ?, ?, ?) as score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score DESC
            LIMIT ?
        """, (chunk_id_w, doc_id_w, source_w, title_w, text_w, fts_query, top_k))
        return [(int(row[0]), float(row[1])) for row in cur.fetchall()]
    
    def search(self, query: str, top_k: int, *, weights: tuple[float, float, float, float, float] | None = None):
        if weights is None:
            weights = (0.0, 0.0, 0.0, 4.0, 1.0)
        return self.search_fts(make_fts5_query(query), top_k, weights=weights)
    
    def search_fts(self, fts_query: str, top_k: int, *, weights: tuple[float, float, float, float, float] | None = None):
        return self._search_fts(fts_query, top_k, weights=weights)
    
class TurboVecRetriever:
    def __init__(self, turbovec_dir: Path, *, expected_model: str | None = None, mismatch_policy: str = "error"):
        current = turbovec_dir / "current"
        self.index_path = current / "index.tvim"
        self.meta_path = current / "meta.json"

        if not (self.index_path.exists() and self.meta_path.exists()):
            raise FileNotFoundError(f"TurboVec bundle missing under {current}")

        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.dims = int(self.meta["dims"])
        self.model = self.meta.get("embedding_model", "unknown")
        self.count = int(self.meta.get("count", 0))

        if expected_model:
            actual = normalize_model_name(self.model)
            expected = normalize_model_name(expected_model)

            if actual and expected and actual != expected:
                msg = (f"[TURBOVEC] embedding model mismatch: meta.json={self.model!r} config_expected={expected_model!r}")

                policy = str(mismatch_policy or "error").strip().lower()
                if policy == "error":
                    raise RuntimeError(msg)
                if policy == "warn":
                    log.warning(msg)
                elif policy == "ignore":
                    log.warning("[TURBOVEC] ignoring model mismatch: %s", msg)
                else:
                    raise RuntimeError(f"Unknown TurboVec mismatch policy: {policy}")

        self.index = IdMapIndex.load(str(self.index_path))

        log.info("[TURBOVEC] loaded index dims=%d model=%s count=%d path=%s", self.dims, self.model, self.count, self.index_path)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if k <= 0 or self.count <= 0:
            return []

        q = np.asarray(query_vec, dtype=np.float32)

        if q.ndim == 1:
            q = q.reshape(1, -1)

        if q.ndim != 2 or q.shape[0] != 1:
            raise ValueError(f"TurboVecRetriever expects one query with shape (1, dims), got {q.shape}")

        if q.shape[1] != self.dims:
            raise ValueError(f"TurboVec query dimension mismatch: expected {self.dims}, got {q.shape[1]}")

        norm = np.linalg.norm(q, axis=1, keepdims=True)
        q = np.ascontiguousarray(q / np.maximum(norm, 1.0e-12), dtype=np.float32)
        k = min(int(k), self.count)

        try:
            scores, ids = self.index.search(q, k=k)
        except (TypeError, ValueError):
            scores, ids = self.index.search(q[0], k=k)

        scores = np.asarray(scores)
        ids = np.asarray(ids)

        if scores.ndim == 2:
            scores = scores[0]

        if ids.ndim == 2:
            ids = ids[0]

        return [(int(cid), float(score)) for cid, score in zip(ids, scores) if int(cid) >= 0]
    
class SpladeRetriever:
    def __new__(cls, index_dir: Path, *, model_name: str, device: str, max_length: int, max_active_dims: int | None, cache_folder: str | None = None,):
        key = (str(index_dir.resolve()), model_name, device, max_length, max_active_dims, cache_folder)

        if key not in splade_retriever_cache:
            instance = super().__new__(cls)
            instance._initialized = False
            splade_retriever_cache[key] = instance

        return splade_retriever_cache[key]

    def __init__(self, index_dir: Path, *, model_name: str, device: str, max_length: int, max_active_dims: int | None, cache_folder: str | None = None,):
        if self._initialized:
            return

        current = index_dir / "current"
        manifest_path = current / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"SPLADE manifest missing: {manifest_path}")

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        expected = {
            "model": model_name,
            "max_length": max_length,
            "max_active_dims": max_active_dims,
        }

        for key, expected_value in expected.items():
            actual_value = self.manifest.get(key)

            if actual_value != expected_value:
                raise RuntimeError(f"[SPLADE] configuration mismatch: {key}={actual_value!r}, expected={expected_value!r}")

        model_key = (model_name, device, max_length, max_active_dims, cache_folder,)
        self.model = load_splade_model(model_name, device=device, max_length=max_length, max_active_dims=max_active_dims, cache_folder=cache_folder,)
        with splade_model_locks_guard:
            self.query_lock = splade_model_locks.setdefault(model_key, threading.Lock())

        self.max_active_dims = max_active_dims
        self.vocabulary_size = int(self.manifest["vocabulary_size"])

        shard_directories = sorted(path for path in current.glob("shard_*") if path.is_dir())

        if not shard_directories:
            raise RuntimeError(f"[SPLADE] No SPLADE shards found under {current}")

        self.shards = [load_csc_shard(path) for path in shard_directories]

        log.info("[SPLADE] loaded shards=%d chunks=%d model=%s", len(self.shards), int(self.manifest["chunk_count"]), model_name,)

        self._initialized = True

    def search(self, query: str, k: int,) -> list[tuple[int, float]]:
        if k <= 0:
            return []

        with self.query_lock:
            (query_indices, query_values, dimensions,) = encode_query_sparse(self.model, query, max_active_dims=self.max_active_dims,)

        if dimensions != self.vocabulary_size:
            raise RuntimeError(f"[SPLADE] query dimension mismatch: query={dimensions} index={self.vocabulary_size}")

        candidates: list[tuple[int, float]] = []
        for matrix, chunk_ids, _ in self.shards:
            candidates.extend(search_csc_shard(matrix, chunk_ids, query_indices, query_values, k=k))
        candidates.sort(key=lambda item: item[1], reverse=True)
        results = candidates[:k]
        log.info("[SPLADE] query_dims=%d candidates=%d returned=%d", query_indices.size, len(candidates), len(results))
        return results