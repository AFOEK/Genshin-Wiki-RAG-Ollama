from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from turbovec import IdMapIndex
import faiss
import logging
import numpy as np

from .utils import normalize_vec_from_blob, make_fts5_query, normalize_model_name, check_faiss_model_match

log = logging.getLogger(__name__)

faiss_retriever_cache: dict[str, FaissRetriever] = {}

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

    def search(self, query: str, top_k: int):
        fts_query = make_fts5_query(query)
        if not fts_query:
            return []
        
        cur = self.conn.cursor()
        cur.execute("""
            SELECT
                chunk_id,
                -bm25(chunks_fts) as score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts)
            LIMIT ?
        """, (fts_query, top_k))
        return [(int(row[0]), float(row[1])) for row in cur.fetchall()]
    
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
                msg = (
                    "[TURBOVEC] embedding model mismatch: "
                    f"meta.json={self.model!r} config_expected={expected_model!r}"
                )

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
        if k <= 0:
            return []
        
        q = query_vec.astype(np.float32, copy=False)

        try:
            log.info("[TURBOVEC] 2D index search")
            scores, ids = self.index.search(q, k=k)
        except Exception:
            log.info("[TURBOVEC] 1D index search")
            scores, ids = self.index.search(q.reshape(-1), k=k)

        scores = np.asarray(scores).reshape(-1)
        ids = np.asarray(ids).reshape(-1)

        out = []
        for cid, score in zip(ids, scores):
            out.append((int(cid), float(score)))

        return out