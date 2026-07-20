from __future__ import annotations
from pathlib import Path
import os
import logging

log = logging.getLogger(__name__)

def expand_path(p: Path) -> Path:
    return Path(os.path.expandvars(str(p))).expanduser()

def is_usable_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink()
        return True
    except Exception:
        return False
    
def resolve_storage_root(cfg: dict) -> Path:
    storage = cfg.get("storage", {}) or {}
    primary = expand_path(Path(storage.get("primary_root", "")))
    secondary = expand_path(Path(storage.get("secondary_root", "")))
    if primary and is_usable_dir(primary):
        log.info("[SPLADE] Storage root: PRIMARY %s", primary)
        return primary
    
    if secondary and is_usable_dir(secondary):
        log.info("[SPLADE] Storage root: SECONDARY %s", secondary)
        return secondary
    
    raise RuntimeError(f"No usable storage root. primary={primary} secondary={secondary}")
    
def resolve_db_path(cfg: dict) -> Path:
    root = resolve_storage_root(cfg)
    db_rel = cfg.get("db_path", "data/genshin_rag.db")
    db_path = (root / db_rel).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"[DB] DB path resolved at {db_path}")
    return db_path

def resolve_faiss_dir(cfg: dict) -> Path:
    root = resolve_storage_root(cfg)
    faiss_rel =cfg.get("faiss_path", "data/faiss")
    faiss_dir = (root / faiss_rel).resolve()
    faiss_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[FAISS] FAISS directory resolved at {faiss_dir}")
    return faiss_dir

def resolve_turbovec_dir(cfg: dict) -> Path:
    root = resolve_storage_root(cfg)

    tv_cfg = cfg.get("turbovec", {}) or {}
    tv_rel = tv_cfg.get("path", "data/turbovec")

    tv_dir = expand_path(Path(str(tv_rel)))

    if not tv_dir.is_absolute():
        tv_dir = (root / tv_dir).resolve()

    if tv_dir.suffix in {".db", ".sqlite", ".sqlite3"}:
        raise RuntimeError(f"Invalid TurboVec path: {tv_dir}. TurboVec path must be a directory, e.g. data/turbovec")

    tv_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[TurboVec] TurboVec directory resolved at {tv_dir}")
    return tv_dir

def resolve_faiss_paths(cfg: dict):
    d = resolve_faiss_dir(cfg)
    return (
        d,
        d / "index.faiss",
        d / "ids.npy",
        d / "meta.json",
    )

def resolve_splade_dir(cfg: dict) -> Path:
    root = resolve_storage_root(cfg)
    splade_cfg = cfg.get("splade", {}) or {}
    raw_path = expand_path(Path(str(splade_cfg.get("path", "data/splade"))))
    splade_dir = (raw_path if raw_path.is_absolute() else (root / raw_path).resolve())
    splade_dir.mkdir(parents=True, exist_ok=True)
    log.info("[SPLADE] directory resolved at %s", splade_dir)
    return splade_dir

def resolve_cache_folder(cfg: dict) -> str | None:
    splade_cfg = cfg.get("splade", {}) or {}
    raw_value = splade_cfg.get("cache_folder")
    if not raw_value:
        return None

    path = Path(str(raw_value)).expanduser()
    if not path.is_absolute():
        path = resolve_storage_root(cfg) / path

    path.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())