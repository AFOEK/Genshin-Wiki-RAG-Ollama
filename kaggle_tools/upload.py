from __future__ import annotations

import argparse, json, shutil, subprocess
from pathlib import Path

def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)

def ensure_file(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(p)
    return p

def write_dataset_metadata(folder: Path, dataset_slug: str, title: str, is_private: bool) -> None:
    meta = {
        "title": title,
        "id": dataset_slug,
        "licenses": [{"name": "CC0-1.0"}]
    }
    if is_private:
        meta["isPrivate"] = True
    
    (folder / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8"
    )