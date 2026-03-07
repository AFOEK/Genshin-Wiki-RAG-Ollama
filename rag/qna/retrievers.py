from __future__ import annotations
import json
import sqlite3
from pathlib import Path
import faiss
import numpy as np

from .utils import normalize_vec_from_blob

class FaissRetriever:
    def __init__(self, faiss_dir: Path):
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

    def search(self, query_vec: np.ndarray, k: int) -> list[int]:
        _dists, indices = self.index.search(query_vec, k)
        out = []
        for i in indices[0]:
            if i < 0:
                continue
            out.append(int(self.ids[i]))
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

    def search(self, query_vec: np.ndarray, k: int) -> list[int]:
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
        return [cid for _, cid in scores[:k]]