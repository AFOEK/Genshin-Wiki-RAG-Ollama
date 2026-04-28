from __future__ import annotations
import json
import sqlite3
from pathlib import Path
import faiss
import logging
import numpy as np

from .utils import normalize_vec_from_blob

log = logging.getLogger(__name__)

faiss_retriever_cache: dict[str, FaissRetriever] = {}

class FaissRetriever:
    def __new__(cls, faiss_dir: Path):
        key = str(faiss_dir)
        if key not in faiss_retriever_cache:
            instance = super().__new__(cls)
            instance._initialized = False
            faiss_retriever_cache[key] = instance
        return faiss_retriever_cache[key]

    def __init__(self, faiss_dir: Path):
        if self._initialized:
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
        cur = self.conn.cursor()
        cur.execute("""
            SELECT
                chunk_id,
                -bm25(chunks_fts) as score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts)
            LIMIT ?
        """, (query, top_k))
        return [(int(row[0]), float(row[1])) for row in cur.fetchall()]