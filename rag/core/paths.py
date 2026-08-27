from __future__ import annotations
from pathlib import Path
import os
import logging

log = logging.getLogger(__name__)

def expand_path(value: str | Path) -> Path:
    raw = str(value).strip()
    home = str(Path.home())
    raw = raw.replace("${HOME}", home).replace("$HOME", home)
    raw = os.path.expandvars(raw)
    return Path(raw).expanduser()

def is_usable_dir(p: Path, *, create: bool=False) -> bool:
    try:
        if create:
            p.mkdir(parents=True, exist_ok=True)
        elif not p.is_dir():
            return False

        test=p/".write_test"
        test.write_text("ok",encoding="utf-8")
        test.unlink()
        return True
    except Exception:
        return False

def resolve_storage_root(cfg: dict) -> Path:
    storage=cfg.get("storage",{}) or {}
    primary_raw=storage.get("primary_root")
    secondary_raw=storage.get("secondary_root")
    primary_mount_raw=storage.get("primary_mount")

    if primary_raw:
        primary=expand_path(Path(primary_raw))
        mount_ok=True
        if primary_mount_raw:
            primary_mount=expand_path(Path(primary_mount_raw))
            mount_ok=primary_mount.is_mount()

        if mount_ok and is_usable_dir(primary,create=False):
            log.info("[PATH] Storage root: PRIMARY %s", primary)
            return primary

    if secondary_raw:
        secondary=expand_path(Path(secondary_raw))
        if is_usable_dir(secondary,create=True):
            log.info("[PATH] Storage root: SECONDARY %s", secondary)
            return secondary

    raise RuntimeError(f"[PATH] No usable storage root. Primary={primary_raw} secondary={secondary_raw}")
    
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