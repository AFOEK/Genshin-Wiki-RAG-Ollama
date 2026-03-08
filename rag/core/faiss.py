from __future__ import annotations

import json, time, logging
import faiss

from pathlib import Path
import numpy as np

from core.paths import resolve_db_path, resolve_faiss_dir
from core.db import read_only_connect

log = logging.getLogger(__name__)

def atomic_promote(build_dir: Path, current_dir: Path):
    old = current_dir.with_name(current_dir.name + ".old")
    if old.exists():
        pass

    if current_dir.exists():
        current_dir.rename(old)
    build_dir.rename(current_dir)

def build_faiss_from_sqlite(
        cfg: dict, *, batch: int = 5000,
        add_batch: int = 2000, log_every: int = 20000,
        threads: int = 4, overwrite: bool = False
) -> Path:
    db_path = resolve_db_path(cfg)
    faiss_root = resolve_faiss_dir(cfg)
    current_dir = faiss_root / "current"
    
    if current_dir.exists() and not overwrite:
        raise RuntimeError(f"[FAISS] {current_dir} exists; pass overwrite=True to rebuild")
    
    build_dir = faiss_root / f"tmp_build_{int(time.time())}"
    build_dir.mkdir(parents=True, exist_ok=True)

    index_path = build_dir / "index.faiss"
    ids_path = build_dir / "ids.npy"
    meta_path = build_dir / "meta.json"

    log.info("[FAISS] Building from db=%s into %s", db_path, build_dir)

    conn = read_only_connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        SELECT e.dims AS dims
        FROM embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        WHERE c.is_active=1 LIMIT 1
    """)
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("[FAISS] no active embeddings found (nothing to build)")
    
    d = int(row["dims"])
    log.info("[FAISS] dims=%d", d)
    
    faiss_cfg = cfg.get("faiss", {}) or {}
    index_mode = str(faiss_cfg.get("index_mode", "flat_ip")).strip().lower()
    metric_name = str(faiss_cfg.get("metric", "cosine")).strip().lower()

    if metric_name == "cosine":
        metric_type = faiss.METRIC_INNER_PRODUCT
        normalize = True
    elif metric_name == "ip":
        metric_type = faiss.METRIC_INNER_PRODUCT
        normalize = False
    elif metric_name == "l2":
        metric_type = faiss.METRIC_L2
        normalize = False
    else:
        raise RuntimeError(f"[FAISS] unsupported metric={metric_name}")
    
    if index_mode == "flat_ip":
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            index = faiss.IndexFlatIP(d)
        else:
            index = faiss.IndexFlatL2(d)
    elif index_mode == "flat_l2":
        index = faiss.IndexFlatL2(d)
    elif index_mode == "hnsw":
        hnsw_m = int(faiss_cfg.get("hnsw_m", 32))
        index = faiss.IndexHNSWFlat(d, hnsw_m, metric_type)
        index.hnsw.efConstruction = int(faiss_cfg.get("hnsw_ef_construction", 200))
        index.hnsw.efSearch = int(faiss_cfg.get("hnsw_ef_search", 64))
    elif index_mode == "ivf_flat":
        nlist = int(faiss_cfg.get("nlist", 256))
        if metric_type ==  faiss.METRIC_INNER_PRODUCT:
            quantizer = faiss.IndexFlatIP(d)
        else:
            quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, metric_type)
    else:
        raise RuntimeError(f"[FAISS] unsupported index_mode={index_mode}")

    try:
        faiss.omp_set_num_threads(min(threads, faiss.omp_get_max_threads()))
    except Exception:
        pass

    ids: list[int] = []
    total = 0
    t0 = time.time()
    offset = 0

    train_vecs: list[np.ndarray] = []
    trained = not isinstance(index, faiss.IndexIVF)

    while True:
        cur.execute("""
            SELECT e.chunk_id AS chunk_id, e.dims AS dims, e.vector AS vector
            FROM embeddings e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            WHERE c.is_active=1
            ORDER BY e.chunk_id
            LIMIT ? OFFSET ?
        """, (batch, offset))
        rows = cur.fetchall()
        if not rows:
            break

        batch_ids: list[int] = []
        vecs: list[np.ndarray] = []

        for r in rows:
            cid = int(r["chunk_id"])
            dims = int(r["dims"])
            blob = r["vector"]

            if dims != d:
                raise RuntimeError(f"[FAISS] dims mismatch chunk_id={cid}: {dims} != {d}")
            
            v = np.frombuffer(blob, dtype=np.float32)
            if v.size != d:
                raise RuntimeError(f"[FAISS] vector size mismatch chunk_id={cid}: {v.size} != {d}")

            batch_ids.append(cid)
            vecs.append(v)
        
        X = np.stack(vecs, axis=0).astype(np.float32, copy=False)
        if normalize:
            faiss.normalize_L2(X)

        if isinstance(index, faiss.IndexIVF) and not trained:
            train_vecs.append(x)
            total_train = sum(arr.shape[0] for arr in train_vecs)
            nlist = int(faiss_cfg.get("nlist", 256))
            if total_train >= max(10000, nlist * 40):
                Xtrain = np.vstack(train_vecs)
                log.info("[FAISS] training IVF index on %d vector", Xtrain.shape[0])
                index.train(Xtrain)
                trained = True
                train_vecs.clear()
        
        if isinstance(index, faiss.IndexIVF) and not trained:
            offset += len(rows)
            continue

        n = X.shape[0]
        start = 0
        while start < n:
            end = min(start + add_batch, n)
            index.add(X[start:end])
            ids.extend(batch_ids[start:end])
            total += (end - start)
            start = end

        offset += len(rows)

        if total and total % log_every == 0:
            dt = time.time() - t0
            rate = total / max(dt, 1e-9)
            log.info("[FAISS] indexed=%d rate=%.0f vec/s", total, rate)
    

    if isinstance(index, faiss.IndexIVF):
        if not trained:
            raise RuntimeError(f"[FAISS] IVF index never got enough training vectors")
        index.nprobe = int(faiss_cfg.get("nprobe", 16))
    
    faiss.write_index(index, str(index_path))
    np.save(str(ids_path), np.asarray(ids, dtype=np.int64))

    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dims": d,
        "count": int(index.ntotal),
        "metric": "cosine",
        "faiss_index": "IndexFlatIP",
        "normalized": True,
        "db_path": str(db_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if index.ntotal > 0:
        pick = min(123, index.ntotal - 1)
        q = np.zeros((1, d), dtype=np.float32)
        index.reconstruct(pick, q[0])
        D, I = index.search(q, 5)
        log.info("[FAISS] self-test pick=%d top=%d score=%.4f", pick, int(I[0,0]), float(D[0,0]))

    atomic_promote(build_dir, current_dir)
    log.info("[FAISS] promoted -> %s", current_dir)
    return current_dir