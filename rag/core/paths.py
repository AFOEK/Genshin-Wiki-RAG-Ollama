from __future__ import annotations
from pathlib import Path
import os
import sqlite3
import logging

log = logging.getLogger(__name__)

def expand_path(p: Path) -> Path:
    return Path(os.path.expandvars(p)).expanduser()

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
        log.info("Storage root: PRIMARY %s", primary)
        return primary
    
    if secondary and is_usable_dir(secondary):
        log.info("Storage root: SECONDARY %s", secondary)
        return secondary
    
    raise RuntimeError(f"No usable storage root. primary={primary} secondary={secondary}")
    
def resolve_db_path(cfg: dict) -> Path:
    root = resolve_storage_root(cfg)
    db_rel = cfg["db_path"]
    return (root / db_rel).resolve()