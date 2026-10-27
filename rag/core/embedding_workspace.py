from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path

from filelock import FileLock

from core.embed import embed
from core.faiss import build_faiss_from_sqlite
from core.paths import resolve_storage_root
from core.turbovec import build_turbovec_from_sqlite
from core.pipeline import defang_tables

log = logging.getLogger(__name__)


def _absolute(root: Path, raw: str) -> Path:
    value = Path(raw).expanduser()
    return (value if value.is_absolute() else root / value).resolve()


def _file_stat(path: Path) -> dict | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _identity(cfg: dict, root: Path, source: Path) -> dict:
    qcfg = cfg["quantum_peft"]
    model_dir = _absolute(root, str(qcfg["model_dir"]))
    config_path = model_dir / "quantum_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    weights = {name: _file_stat(model_dir / name) for name in ("adapter.safetensors", "model.safetensors")}
    if not any(value is not None for value in weights.values()):
        raise FileNotFoundError(f"No Safetensors checkpoint in {model_dir}")

    return {
        "model_id": str(qcfg["model_id"]),
        "model_dir": str(model_dir),
        "model_config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "weights": weights,
        "source": str(source),
        "source_db": _file_stat(source),
        "source_wal": _file_stat(Path(str(source) + "-wal")),
        "max_embed_chars": int(
            cfg.get("pipeline", {}).get("max_embed_chars", 1800)
        ),
        "max_length": qcfg.get("max_length"),
        "faiss": cfg.get("faiss", {}),
        "turbovec": {
            key: cfg.get("turbovec", {}).get(key)
            for key in (
                "bit_width",
                "batch_size",
                "calibration_size",
            )
        },
    }


def _copy_snapshot(original: Path, target: Path):
    if target.exists():
        return

    temp = target.with_suffix(".copying.sqlite")
    temp.unlink(missing_ok=True)
    with sqlite3.connect(f"{original.as_uri()}?mode=ro", uri=True) as src:
        with sqlite3.connect(str(temp)) as dst:
            src.backup(dst, pages=1000)

    temp.replace(target)

def _embed_missing(cfg: dict, target: Path, signature: str, batch_size: int) -> int:
    with sqlite3.connect(str(target), timeout=120) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        row = conn.execute(
            "SELECT value FROM embedding_meta "
            "WHERE key = 'hq_signature'"
        ).fetchone()

        if row is None or row["value"] != signature:
            # Clear old embeddings ONLY in the workspace copy.
            conn.execute("DELETE FROM embeddings")
            conn.execute("""
                INSERT INTO embedding_meta(key,value)
                VALUES('hq_signature',?)
                ON CONFLICT(key)
                DO UPDATE SET value=excluded.value
            """, (signature,))
            conn.commit()

        completed = 0
        max_chars = int(cfg.get("pipeline", {}).get("max_embed_chars", 1800))
        while True:
            rows = conn.execute("""
                SELECT c.chunk_id, c.text
                FROM chunks c
                JOIN docs d ON d.doc_id = c.doc_id
                LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
                WHERE c.is_active = 1
                  AND COALESCE(d.status, 1) = 1
                  AND e.chunk_id IS NULL
                ORDER BY c.chunk_id
                LIMIT ?
            """, (batch_size,)).fetchall()

            if not rows:
                break

            texts = [defang_tables(str(r["text"] or "")[:max_chars]) for r in rows]
            packed = embed(cfg, texts, mode="passage")
            if len(packed) != len(rows):
                raise RuntimeError("HQ embedding batch returned wrong vector count")

            values = []
            for row, (blob, dims) in zip(rows, packed):
                if (not isinstance(blob, bytes) or dims <= 0 or len(blob) != dims * 4):
                    raise RuntimeError(f"Invalid HQ embedding at chunk {row['chunk_id']}")

                values.append((
                    int(row["chunk_id"]),
                    int(dims),
                    blob,
                ))

            conn.executemany("""
                INSERT INTO embeddings(chunk_id,dims,vector)
                VALUES (?,?,?)
            """, values)

            # Make indexing resumable after interruption.
            conn.commit()
            completed += len(rows)
            if completed % (batch_size * 100) == 0:
                log.info("[Custom_Embedding] Embedded %d chunks this session", completed)

        missing = conn.execute("""
            SELECT COUNT(*)
            FROM chunks c
            JOIN docs d ON d.doc_id=c.doc_id
            LEFT JOIN embeddings e ON e.chunk_id=c.chunk_id
            WHERE c.is_active=1
              AND COALESCE(d.status,1)=1
              AND e.chunk_id IS NULL
        """).fetchone()[0]

        if missing:
            raise RuntimeError(
                f"Incomplete HQ embeddings: {missing} missing"
            )

        total = conn.execute("""
            SELECT COUNT(*)
            FROM chunks c
            JOIN docs d ON d.doc_id=c.doc_id
            WHERE c.is_active=1
              AND COALESCE(d.status,1)=1
        """).fetchone()[0]

        return int(total)


def activate_hq_workspace(cfg: dict) -> dict:
    if cfg.get("_hq_workspace_active"):
        return cfg

    provider = str(cfg.get("runtime", {}).get("embedding_provider", "ollama")).lower()
    if provider != "quantum_peft":
        return cfg

    qcfg = cfg.get("quantum_peft", {})
    if not qcfg.get("auto_prepare", False):
        return cfg

    if not qcfg.get("model_id"):
        raise ValueError("quantum_peft.model_id must be set")

    batch_size = int(qcfg.get("batch_size", 8))
    if batch_size < 1:
        raise ValueError("quantum_peft.batch_size must be positive")

    root = resolve_storage_root(cfg)
    original = _absolute(root, str(cfg.get("db_path", "data/genshin_rag.db")),)

    if not original.is_file():
        raise FileNotFoundError(original)

    identity = _identity(cfg, root, original)
    encoded = json.dumps(identity, sort_keys=True, default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
    workspace = (root / "data" / "embedding_workspaces" / digest)
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "genshin_rag.db"
    faiss_index = (workspace / "faiss" / "current" / "index.faiss")
    tv_index = ( workspace / "turbovec" / "current" / "index.tvim")
    ready = workspace / "ready.json"
    # Override paths in memory, not in config.yaml.
    cfg["db_path"] = str(target)
    cfg["faiss_path"] = str(workspace / "faiss")
    cfg.setdefault("turbovec", {})["path"] = str(workspace / "turbovec")
    cfg.setdefault("retrieval_cache", {})["path"] = str(workspace / "retrieval_cache.sqlite")
    cfg["_hq_workspace_active"] = True

    lock = FileLock(str(workspace / "prepare.lock"), timeout=-1,)

    with lock:
        if (ready.is_file() and target.is_file() and faiss_index.is_file() and tv_index.is_file()):
            try:
                saved = json.loads(ready.read_text(encoding="utf-8"))
                if saved.get("identity") == digest:
                    log.info("[Custom_Embedding] Reusing workspace: %s", workspace,)
                    return cfg
            except (OSError, ValueError):
                pass

        _copy_snapshot(original, target)
        count = _embed_missing(cfg, target, digest, batch_size)
        if count == 0:
            raise RuntimeError("Cannot create dense indexes from an empty corpus")

        log.info("[Custom_Embedding] Building FAISS index")
        build_faiss_from_sqlite(cfg, overwrite=True)
        log.info("[Custom_Embedding] Building TurboVec index")
        build_turbovec_from_sqlite(cfg, overwrite=True, backend="quantum_peft",)
        if not (faiss_index.is_file() and tv_index.is_file()):
            raise RuntimeError("Dense index files missing after build")

        temp = ready.with_suffix(".tmp")
        temp.write_text(
            json.dumps({
                "identity": digest,
                "chunks": count,
            }), encoding="utf-8",)
        temp.replace(ready)
        log.info("[Custom_Embedding] Workspace ready: %d chunks", count,)

    return cfg
