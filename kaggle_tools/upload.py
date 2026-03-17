from __future__ import annotations

import argparse, json, shutil, subprocess, logging
from pathlib import Path

log = logging.getLogger(__name__)

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

def create_or_update_dataset(folder: Path, create_if_missing:bool = True) -> None:
    try:
        run([
            "kaggle", "datasets", "version",
            "-p", str(folder),
            "-m", "Update exported Genshin RAG chunks",
            "-r", "zip",
        ])
    except subprocess.CalledProcessError:
        if not create_if_missing:
            raise
        run([
            "kaggle", "datasets", "create",
            "-p", str(folder),
            "-r", "zip",
        ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-slug",
        required=True,
        help="Kaggle dataset slug in the form owner/dataset-name",
    )
    ap.add_argument(
        "--dataset-title",
        default="Genshin RAG Chunks",
        help="Human-readable Kaggle dataset title",
    )
    ap.add_argument(
        "--chunks-file",
        default="rag/chunks_kaggle/chunks.jsonl",
        help="Path to exported chunks JSONL",
    )
    ap.add_argument(
        "--work-dir",
        default="rag/chunks_kaggle/dataset_upload",
        help="Temporary Kaggle dataset staging folder",
    )
    ap.add_argument(
        "--public",
        action="store_true",
        help="Make dataset public; default is private",
    )
    args = ap.parse_args()

    chunks_file = ensure_file(Path(args.chunks_file))
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    staged_chunks = work_dir / "chunks.jsonl"
    shutil.copy2(chunks_file, staged_chunks)

    write_dataset_metadata(
        work_dir,
        dataset_slug=args.dataset_slug,
        title=args.dataset_title,
        is_private=not args.public,
    )

    create_or_update_dataset(work_dir, create_if_missing=True)

    log.info(f"[INFO] Uploaded {staged_chunks} to Kaggle dataset {args.dataset_slug}")

if __name__ == "__main__":
    main()