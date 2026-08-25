from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import logging
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    sys.path.insert(0, str(REPO_ROOT / "rag"))
    from utils.logging_setup import setup_logging  # type: ignore
except Exception:  # pragma: no cover - fallback path
    def setup_logging(log_path: str | None = None, level: str = "INFO") -> None:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

log = logging.getLogger("combine_hf_dataset")

DEFAULT_HF_REPO = "totmalone/Genshin-Impact-SFT"

DATA_SUFFIXES = {".jsonl", ".json", ".parquet"}


CATEGORIES: list[tuple[str, re.Pattern[str]]] = [
    ("genshin_retrieval_pairs.jsonl", re.compile(r"retrieval[_-]?pairs", re.I)),
    ("genshin_double_negative_pairs.jsonl", re.compile(r"double[_-]?negative", re.I)),
    ("genshin_sft_negative_answerability.jsonl",
     re.compile(r"sft[_-]?negative|negative[_-]?answerability", re.I)),
    ("genshin_rejected.jsonl", re.compile(r"rejected", re.I)),
    ("genshin_rag_sft_candidates.jsonl",
     re.compile(r"(rag[_-]?)?sft[_-]?candidates|sft[_-]?out|lora[_-]?out", re.I)),
]

IGNORE_NAMES = {
    ".gitattributes", "readme.md", "dataset_infos.json",
    "dataset_dict.json", "state.json", "dataset_zip_hash.txt",
}


def classify(filename: str) -> str | None:
    """Return the canonical output name for a source file (by basename), or None
    if it is not a dataset shard we recognise."""
    stem = Path(filename).name
    if stem.lower() in IGNORE_NAMES:
        return None
    for output_name, pattern in CATEGORIES:
        if pattern.search(stem):
            return output_name
    return None


def _payload_suffix(name: str) -> str:
    """Suffix of the payload, seeing through a trailing .gz (e.g. .jsonl.gz -> .jsonl)."""
    core = name[:-3] if name.endswith(".gz") else name
    return Path(core).suffix.lower()


def record_key(record: dict) -> str:
    """Stable de-dup key: the record's `id` if it has a non-empty one, else a
    SHA-1 of its canonical JSON. This is what lets id-less `rejected` rows survive."""
    rid = record.get("id")
    if isinstance(rid, str) and rid.strip():
        return f"id::{rid.strip()}"
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha1::" + hashlib.sha1(canonical.encode("utf-8")).hexdigest()


# --- record readers -------------------------------------------------------

def _parse_json_text(text: str, label: str, strict: bool) -> Iterator[dict]:
    """Parse an in-memory blob that is either a JSON array or object-per-line."""
    stripped = text.lstrip()
    if stripped.startswith("["):
        for row in json.loads(text):
            if isinstance(row, dict):
                yield row
        return
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"{label}:{lineno} is not valid JSON ({exc.msg})"
            if strict:
                raise ValueError(msg) from exc
            log.warning("skipping malformed line: %s", msg)
            continue
        if isinstance(row, dict):
            yield row
        elif strict:
            raise ValueError(f"{label}:{lineno} is JSON but not an object")


def _iter_parquet_bytes(data: bytes) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("A Parquet source was found but pyarrow is not installed. "
                           "Run: pip install pyarrow") from exc
    parquet_file = pq.ParquetFile(io.BytesIO(data))
    for batch in parquet_file.iter_batches():
        for row in batch.to_pylist():
            yield row


def iter_loose_records(path: Path, strict: bool) -> Iterator[dict]:
    """Stream records from a loose .jsonl / .json / .parquet file (optionally .gz)."""
    suffix = _payload_suffix(path.name)
    raw_bytes: bytes | None = None
    if path.suffix == ".gz":
        raw_bytes = gzip.decompress(path.read_bytes())

    if suffix == ".parquet":
        yield from _iter_parquet_bytes(raw_bytes if raw_bytes is not None else path.read_bytes())
        return

    if raw_bytes is not None:
        yield from _parse_json_text(raw_bytes.decode("utf-8", "replace"), path.name, strict)
        return

    # Uncompressed .jsonl/.json on disk -> stream line by line (memory-flat).
    with path.open("r", encoding="utf-8") as handle:
        head = handle.read(64).lstrip()
        handle.seek(0)
        if head.startswith("["):
            for row in json.load(handle):
                if isinstance(row, dict):
                    yield row
            return
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"{path.name}:{lineno} is not valid JSON ({exc.msg})"
                if strict:
                    raise ValueError(msg) from exc
                log.warning("skipping malformed line: %s", msg)
                continue
            if isinstance(row, dict):
                yield row
            elif strict:
                raise ValueError(f"{path.name}:{lineno} is JSON but not an object")


def iter_zip_member_records(zf: zipfile.ZipFile, member: str, strict: bool) -> Iterator[dict]:
    raw = zf.read(member)
    name = member
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
        name = name[:-3]
    if Path(name).suffix.lower() == ".parquet":
        yield from _iter_parquet_bytes(raw)
        return
    yield from _parse_json_text(raw.decode("utf-8", "replace"), member, strict)


# --- source discovery -----------------------------------------------------

def _is_cache_path(path: Path, root: Path) -> bool:
    return any(part == ".cache" for part in path.relative_to(root).parts)


def gather_loose_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_cache_path(path, root):
            continue
        if _payload_suffix(path.name) in DATA_SUFFIXES:
            files.append(path)
    return files


def gather_zip_files(root: Path) -> list[Path]:
    return [
        p for p in sorted(root.rglob("*.zip"))
        if p.is_file() and not _is_cache_path(p, root)
    ]


def download_hf_repo(repo: str, revision: str, download_dir: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --hf-repo mode. "
            "Run: pip install huggingface_hub  (or use --local-dir)."
        ) from exc
    log.info("downloading dataset repo %s (revision=%s) -> %s", repo, revision, download_dir)
    local_path = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        local_dir=str(download_dir),
        token=token,
        allow_patterns=["*.zip", "*.jsonl", "*.json", "*.parquet", "*.gz"],
    )
    return Path(local_path)


# --- combine --------------------------------------------------------------

def combine(source_root: Path, out_dir: Path, dedup: bool, strict: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    loose_files = gather_loose_files(source_root)
    zip_files = gather_zip_files(source_root)
    if not loose_files and not zip_files:
        raise RuntimeError(f"No .zip/.jsonl/.json/.parquet files found under {source_root}")

    handles: dict[str, io.TextIOBase] = {}
    seen: dict[str, set[str]] = {}
    stats: dict[str, dict] = {}
    for name, _ in CATEGORIES:
        handles[name] = (out_dir / name).open("w", encoding="utf-8")
        seen[name] = set()
        stats[name] = {"sources": [], "records_read": 0,
                       "records_written": 0, "duplicates_dropped": 0,
                       "output_path": str(out_dir / name)}
    unmatched: list[str] = []

    def route(category: str | None, label: str, records: Iterator[dict]) -> None:
        if category is None:
            unmatched.append(label)
            # still drain so any warnings fire, but there's nothing to write
            for _ in records:
                pass
            return
        st = stats[category]
        st["sources"].append(label)
        s = seen[category]
        handle = handles[category]
        for record in records:
            st["records_read"] += 1
            if dedup:
                key = record_key(record)
                if key in s:
                    st["duplicates_dropped"] += 1
                    continue
                s.add(key)
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            st["records_written"] += 1

    try:
        for path in loose_files:
            label = str(path.relative_to(source_root))
            log.info("  loose: %s", label)
            route(classify(path.name), label, iter_loose_records(path, strict))

        for zpath in zip_files:
            zlabel = str(zpath.relative_to(source_root))
            log.info("  zip:   %s", zlabel)
            with zipfile.ZipFile(zpath) as zf:
                for member in zf.namelist():
                    if member.endswith("/"):
                        continue
                    if _payload_suffix(member) not in DATA_SUFFIXES:
                        continue
                    member_label = f"{zlabel}::{member}"
                    route(classify(member), member_label,
                          iter_zip_member_records(zf, member, strict))
    finally:
        for handle in handles.values():
            handle.close()

    report = {name: stats[name] for name, _ in CATEGORIES}
    report["_unmatched_files"] = unmatched
    for name, _ in CATEGORIES:
        st = stats[name]
        log.info("  [%s] sources=%d read=%d written=%d dupes=%d",
                 name, len(st["sources"]), st["records_read"],
                 st["records_written"], st["duplicates_dropped"])
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine the totmalone/Genshin-Impact-SFT dataset into 5 consolidated JSONL files.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--hf-repo", default=DEFAULT_HF_REPO,
                        help="Hugging Face dataset repo id (default: %(default)s).")
    source.add_argument("--local-dir", default=None,
                        help="Combine files already on disk instead of downloading from the Hub.")
    parser.add_argument("--revision", default="main", help="Hub revision/branch (default: main).")
    parser.add_argument("--download-dir", default=None,
                        help="Where to cache the Hub snapshot (default: fine_tune/data/hf_cache/<repo>).")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "fine_tune" / "data" / "combined"),
                        help="Directory for the five output files (default: %(default)s).")
    parser.add_argument("--token", default=None,
                        help="Hugging Face token for a private repo (else uses the HF_TOKEN env / cached login).")
    parser.add_argument("--no-dedup", action="store_true", help="Concatenate without de-duplicating.")
    parser.add_argument("--strict", action="store_true",
                        help="Abort on the first malformed record instead of warning and skipping.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_logging(level=args.log_level)

    if args.local_dir:
        source_root = Path(args.local_dir).expanduser().resolve()
        if not source_root.is_dir():
            raise SystemExit(f"--local-dir does not exist: {source_root}")
    else:
        download_dir = (
            Path(args.download_dir).expanduser().resolve()
            if args.download_dir
            else REPO_ROOT / "fine_tune" / "data" / "hf_cache" / args.hf_repo.replace("/", "__")
        )
        download_dir.mkdir(parents=True, exist_ok=True)
        source_root = download_hf_repo(args.hf_repo, args.revision, download_dir, args.token)

    out_dir = Path(args.out_dir).expanduser().resolve()
    report = combine(source_root=source_root, out_dir=out_dir,
                     dedup=not args.no_dedup, strict=args.strict)

    report_path = out_dir / "combine_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    log.info("=" * 60)
    log.info("Combine complete. Outputs in %s", out_dir)
    for name, _ in CATEGORIES:
        st = report[name]
        log.info("  %-45s %8d records (%d dupes dropped)",
                 name, st["records_written"], st["duplicates_dropped"])
    if report["_unmatched_files"]:
        log.warning("%d source file(s) did not match any category; see combine_report.json",
                    len(report["_unmatched_files"]))
    log.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
