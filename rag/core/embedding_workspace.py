from __future__ import annotations

import hashlib
import json
import logging
import sqlite3

from typing import cast
from pathlib import Path
from filelock import FileLock
from copy import deepcopy
from contextlib import closing

from core.embed import embed, embedding_model_name, normalize_backend_name
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
    provider = normalize_backend_name(
        cfg.get("runtime", {}).get(
            "embedding_provider", "ollama"
        )
    )

    identity = {
        "provider": provider,
        "model_id": embedding_model_name(cfg),
        "source": str(source),
        "source_db": _file_stat(source),
        "source_wal": _file_stat(Path(str(source) + "-wal")),
        "max_embed_chars": int(cfg.get("pipeline", {}).get("max_embed_chars", 1800)),
        "embedding_prompts": cfg.get("embedding_prompts", {}),
        "faiss": cfg.get("faiss", {}),
        "build_turbovec": bool(cfg.get("embedding_workspace", {}).get("build_turbovec", True)),
        "turbovec": {
            key: cfg.get("turbovec", {}).get(key)
            for key in (
                "bit_width",
                "batch_size",
                "calibration_size",
            )
        },
    }

    if not identity["model_id"]:
        raise ValueError("Embedding model is not configured")

    if provider == "quantum_peft":
        qcfg = cfg["quantum_peft"]
        model_dir = _absolute(root, str(qcfg["model_dir"]))
        config_path = model_dir / "quantum_config.json"

        if not config_path.is_file():
            raise FileNotFoundError(config_path)

        weights = {}
        for name in ("adapter.safetensors", "model.safetensors",):
            path = model_dir / name
            if path.is_file():
                with path.open("rb") as f:
                    digest = hashlib.file_digest(f, "sha256").hexdigest()

                weights[name] = {
                    "sha256": digest,
                    "size": path.stat().st_size,
                }

        if not any(weights.values()):
            raise FileNotFoundError(f"No Safetensors found in {model_dir}")

        identity.update({
            "model_dir": str(model_dir),
            "model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "weights": weights,
            "weights_kind": qcfg.get("weights_kind", "auto"),
            "query_max_length": int(qcfg.get("query_max_length", 80)),
            "document_max_length": int(qcfg.get("document_max_length", 320)),
            "include_document_title": bool(qcfg.get("include_document_title", True)),
            "batch_size": int(qcfg.get("batch_size", 16)),
            "preprocessing_version": 2,
        })

    elif provider == "ollama":
        identity["base_url"] = cfg["ollama"]["base_url"]

    elif provider == "llamacpp":
        identity["embedding_url"] = cfg["llamacpp"]["embedding_url"]

    return identity


def _copy_snapshot(original: Path, target: Path):
    if target.exists():
        return

    temp = target.with_suffix(".copying.sqlite")
    temp.unlink(missing_ok=True)

    with closing(sqlite3.connect(
        f"{original.as_uri()}?mode=ro",
        uri=True,
    )) as src:
        with closing(sqlite3.connect(str(temp))) as dst:
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
                SELECT c.chunk_id, c.text, d.title
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

            provider = normalize_backend_name(cfg.get("runtime", {}).get("embedding_provider", "ollama"))
            if provider == "quantum_peft":
                texts = [str(r["text"] or "") for r in rows]
            else:
                texts = [defang_tables(str(r["text"] or "")[:max_chars]) for r in rows]
            titles = [str(r["title"] or "") for r in rows]
            packed = embed(cfg, texts, mode="passage", title=titles)
            if not isinstance(packed, list):
                raise TypeError("Expected batch embedding results")

            if len(packed) != len(rows):
                raise RuntimeError("Embedding batch returned wrong vector count")

            values = []
            for row, (blob, dims) in zip(rows, packed, strict=True):
                if (not isinstance(blob, bytes) or not isinstance(dims, int) or dims <= 0 or len(blob) != dims * 4):
                    raise RuntimeError(f"Invalid HQ embedding at chunk {row['chunk_id']}")

                values.append((int(row["chunk_id"]), dims, blob,))

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

def activate_hq_workspace(cfg: dict, *, embed_only: bool = False,) -> dict:
    if cfg.get("_hq_workspace_active"):
        return cfg

    cfg = deepcopy(cfg)
    provider = normalize_backend_name(
        cfg.get("runtime", {}).get(
            "embedding_provider", "ollama"
        )
    )

    cfg.setdefault("runtime", {})["embedding_provider"] = provider
    qcfg = cfg.get("quantum_peft", {}) or {}
    wcfg = cfg.get("embedding_workspace", {}) or {}
    build_turbovec = bool(wcfg.get("build_turbovec", True))
    enabled = bool(wcfg.get("enabled", False))
    if provider == "quantum_peft":
        enabled = enabled or bool(qcfg.get("auto_prepare", False))

    if not embed_only and not enabled:
        return cfg

    # Select the appropriate embedding batch size.
    if provider == "quantum_peft":
        batch_size = int(qcfg.get("batch_size", 8))
    else:
        batch_size = int(cfg.get("threading", {}).get("embed_batch_size", 32))

    if batch_size < 1:
        raise ValueError("Embedding batch size must be positive")

    root = resolve_storage_root(cfg)
    original = _absolute(root, str(cfg.get("db_path", "data/genshin_rag.db")),)
    if not original.is_file():
        raise FileNotFoundError(original)

    identity = _identity(cfg, root, original)
    digest = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:20]

    workspace = (root / "data" / "embedding_workspaces" / digest)
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "genshin_rag.db"
    ready = workspace / "ready.json"
    faiss_index = (workspace / "faiss" / "current" / "index.faiss")
    tv_index = (workspace / "turbovec" / "current" / "index.tvim")
    cfg["db_path"] = str(target)
    cfg["faiss_path"] = str(workspace / "faiss")
    cfg.setdefault("turbovec", {})["path"] = str(workspace / "turbovec")
    cfg["turbovec"]["model_source"] = "runtime"
    cfg["runtime"]["faiss_expected_model"] = "runtime"
    cfg.setdefault("retrieval_cache", {})["path"] = str(workspace / "retrieval_cache.sqlite")
    log.info("[EMBED] Provider: %s", provider)
    log.info("[EMBED] Model: %s", identity["model_id"])
    log.info("[EMBED] Workspace: %s", workspace)

    lock = FileLock(str(workspace / "prepare.lock"),  timeout=-1,)
    with lock:
        # Reuse the original corpus without crawling.
        _copy_snapshot(original, target)
        # Generate or resume embeddings.
        count = _embed_missing(cfg, target, digest, batch_size,)
        if count == 0:
            raise RuntimeError("No active chunks to embed")

        log.info("[EMBED] Active embedded chunks: %d", count,)
        if embed_only:
            log.info("[EMBED_ONLY] Skipping dense index construction")
            return cfg

        # Only import index builders when needed.
        from core.faiss import build_faiss_from_sqlite
        from core.turbovec import build_turbovec_from_sqlite
        indexes_ready = False
        if (ready.is_file() and faiss_index.is_file() and (not build_turbovec or tv_index.is_file())):
            try:
                saved = json.loads(ready.read_text(encoding="utf-8"))
                indexes_ready = (saved.get("identity") == digest and saved.get("chunks") == count)

            except (OSError, ValueError):
                indexes_ready = False

        if not indexes_ready:
            ready.unlink(missing_ok=True)
            log.info("[EMBED] Building FAISS")
            build_faiss_from_sqlite(cfg, overwrite=True,)
            if build_turbovec:
                log.info("[EMBED] Building TurboVec")
                build_turbovec_from_sqlite(cfg, overwrite=True, backend=provider,)
                
            if not faiss_index.is_file():
                raise RuntimeError("FAISS index construction incomplete")

            if build_turbovec and not tv_index.is_file():
                raise RuntimeError("TurboVec index construction incomplete")

            temp = ready.with_suffix(".tmp")
            temp.write_text(
                json.dumps({
                    "identity": digest,
                    "model": identity["model_id"],
                    "provider": provider,
                    "chunks": count,
                }, indent=2),
                encoding="utf-8",
            )
            temp.replace(ready)

    cfg["_hq_workspace_active"] = True
    return cfg