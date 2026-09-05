from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
import yaml

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_SOURCES = [
    "genshin_wiki",
    "kqm_tcl",
    "kqm_news",
    "honey",
    "genshin_gg",
    "game8",
]

def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

def add_sample(samples: dict[str, list], category: str, value: dict, limit: int = 20) -> None:
    bucket = samples[category]
    if len(bucket) < limit:
        bucket.append(value)

def assistant_answer(record: dict) -> str:
    for message in reversed(record.get("message", []) or []):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""

def parse_sources(value) -> list[str]:
    if value is None:
        return list(DEFAULT_DATASET_SOURCES)
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return list(DEFAULT_DATASET_SOURCES)

def open_db(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()

    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=FILE")
    con.execute("PRAGMA locking_mode=EXCLUSIVE")

    con.execute("""
        CREATE TABLE sft (
            id TEXT PRIMARY KEY,
            source TEXT,
            question TEXT,
            normalized_question TEXT,
            record_type TEXT,
            positive_chunk_id TEXT
        )
    """)

    con.execute("""
        CREATE TABLE questions (
            normalized_question TEXT PRIMARY KEY,
            record_id TEXT
        )
    """)

    con.execute("""
        CREATE TABLE retrieval (
            origin_id TEXT PRIMARY KEY,
            retrieval_id TEXT,
            positive_source TEXT
        )
    """)

    con.execute("""
        CREATE TABLE seen (
            kind TEXT,
            id TEXT,
            PRIMARY KEY(kind, id)
        )
    """)

    return con

def register_seen(con: sqlite3.Connection, kind: str, record_id: str) -> bool:
    cur = con.execute(
        "INSERT OR IGNORE INTO seen(kind, id) VALUES (?, ?)",
        (kind, record_id),
    )
    return cur.rowcount == 1

def progress(label: str, count: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 0.001)
    rate = count / elapsed
    print(f"\r{label}: {count:,} records ({rate:,.0f}/s)", end="", flush=True)

def scan_sft(path: Path, con: sqlite3.Connection, samples: dict[str, list], unavailable_sources: set[str]) -> dict:
    stats = Counter()
    sources = Counter()
    types = Counter()
    unavailable = Counter()
    started = time.monotonic()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            stats["lines"] += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                stats["invalid_json"] += 1
                add_sample(samples, "sft_invalid_json", {
                    "line": line_number,
                    "error": str(exc),
                })
                continue

            rid = str(row.get("id", "")).strip()
            metadata = row.get("metadata", {}) or {}
            source = str(metadata.get("source", "")).strip()
            question = str(metadata.get("question", "")).strip()
            reference = str(metadata.get("reference_answer", "")).strip()
            answer = assistant_answer(row)
            record_type = str(metadata.get("type", "")).strip()
            positive_chunk_id = str(metadata.get("positive_chunk_id", "")).strip()

            sources[source or "<missing>"] += 1
            types[record_type or "<missing>"] += 1

            if source in unavailable_sources:
                unavailable[source] += 1

            if not rid:
                stats["missing_id"] += 1
                add_sample(samples, "sft_missing_id", {
                    "line": line_number,
                    "source": source,
                })
                continue

            cur = con.execute(
                """
                INSERT OR IGNORE INTO sft
                (id, source, question, normalized_question, record_type, positive_chunk_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    source,
                    question,
                    normalize_question(question),
                    record_type,
                    positive_chunk_id,
                ),
            )

            if cur.rowcount == 0:
                stats["duplicate_id"] += 1
                add_sample(samples, "sft_duplicate_id", {
                    "id": rid,
                    "source": source,
                })
                continue

            if not question:
                stats["missing_question"] += 1
                add_sample(samples, "sft_missing_question", {
                    "id": rid,
                    "source": source,
                })

            if not reference:
                stats["missing_reference_answer"] += 1
                add_sample(samples, "sft_missing_reference_answer", {
                    "id": rid,
                    "source": source,
                    "question": question,
                })

            if not answer:
                stats["missing_assistant_answer"] += 1
                add_sample(samples, "sft_missing_assistant_answer", {
                    "id": rid,
                    "source": source,
                    "question": question,
                })

            if not source:
                stats["missing_source"] += 1

            qkey = normalize_question(question)

            if qkey:
                qcur = con.execute(
                    "INSERT OR IGNORE INTO questions(normalized_question, record_id) VALUES (?, ?)",
                    (qkey, rid),
                )

                if qcur.rowcount == 0:
                    stats["duplicate_question"] += 1
                    existing = con.execute(
                        "SELECT record_id FROM questions WHERE normalized_question=?",
                        (qkey,),
                    ).fetchone()

                    add_sample(samples, "sft_duplicate_question", {
                        "id": rid,
                        "other_id": existing[0] if existing else None,
                        "question": question,
                    })

            stats["valid_records"] += 1

            if stats["lines"] % 50000 == 0:
                con.commit()
                progress("SFT", stats["lines"], started)

    con.commit()
    print()

    return {
        **dict(stats),
        "source_distribution": dict(sources.most_common()),
        "type_distribution": dict(types.most_common()),
        "unavailable_source_distribution": dict(unavailable.most_common()),
    }

def scan_retrieval(path: Path, con: sqlite3.Connection, samples: dict[str, list], unavailable_sources: set[str]) -> dict:
    stats = Counter()
    positive_sources = Counter()
    hard_sources = Counter()
    easy_sources = Counter()
    unavailable_positive = Counter()
    started = time.monotonic()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            stats["lines"] += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                stats["invalid_json"] += 1
                add_sample(samples, "retrieval_invalid_json", {
                    "line": line_number,
                    "error": str(exc),
                })
                continue

            rid = str(row.get("id", "")).strip()
            origin = str(row.get("origin_record_id", "")).strip()
            query = str(row.get("query", "")).strip()
            positive = row.get("positive", {}) or {}
            hard = row.get("hard_negatives", []) or []
            easy = row.get("easy_negatives", []) or []
            positive_source = str(positive.get("source", "")).strip()

            positive_sources[positive_source or "<missing>"] += 1

            if positive_source in unavailable_sources:
                unavailable_positive[positive_source] += 1

            if rid and not register_seen(con, "retrieval_id", rid):
                stats["duplicate_id"] += 1
                add_sample(samples, "retrieval_duplicate_id", {
                    "id": rid,
                    "origin_record_id": origin,
                })

            if not origin:
                stats["missing_origin_record_id"] += 1
                add_sample(samples, "retrieval_missing_origin", {
                    "id": rid,
                    "line": line_number,
                })
                continue

            sft_row = con.execute(
                "SELECT source, question FROM sft WHERE id=?",
                (origin,),
            ).fetchone()

            if sft_row is None:
                stats["orphan_retrieval_pair"] += 1
                add_sample(samples, "orphan_retrieval_pair", {
                    "id": rid,
                    "origin_record_id": origin,
                    "positive_source": positive_source,
                })
            else:
                sft_source, sft_question = sft_row

                if positive_source and sft_source and positive_source != sft_source:
                    stats["positive_source_mismatch"] += 1
                    add_sample(samples, "positive_source_mismatch", {
                        "origin_record_id": origin,
                        "sft_source": sft_source,
                        "retrieval_positive_source": positive_source,
                    })

                if query and sft_question and normalize_question(query) != normalize_question(sft_question):
                    stats["query_mismatch"] += 1
                    add_sample(samples, "retrieval_query_mismatch", {
                        "origin_record_id": origin,
                        "sft_question": sft_question,
                        "retrieval_query": query,
                    })

            cur = con.execute(
                """
                INSERT OR IGNORE INTO retrieval(origin_id, retrieval_id, positive_source)
                VALUES (?, ?, ?)
                """,
                (origin, rid, positive_source),
            )

            if cur.rowcount == 0:
                stats["duplicate_origin_record_id"] += 1
                add_sample(samples, "retrieval_duplicate_origin", {
                    "origin_record_id": origin,
                    "id": rid,
                })

            if not positive:
                stats["missing_positive"] += 1

            if not hard and not easy:
                stats["no_negatives"] += 1

            stats["hard_negatives"] += len(hard)
            stats["easy_negatives"] += len(easy)

            for negative in hard:
                hard_sources[str((negative or {}).get("source", "")).strip() or "<missing>"] += 1

            for negative in easy:
                easy_sources[str((negative or {}).get("source", "")).strip() or "<missing>"] += 1

            stats["valid_records"] += 1

            if stats["lines"] % 50000 == 0:
                con.commit()
                progress("Retrieval", stats["lines"], started)

    con.commit()
    print()

    pair_count = max(stats["valid_records"], 1)

    return {
        **dict(stats),
        "average_hard_negatives": stats["hard_negatives"] / pair_count,
        "average_easy_negatives": stats["easy_negatives"] / pair_count,
        "positive_source_distribution": dict(positive_sources.most_common()),
        "hard_negative_source_distribution": dict(hard_sources.most_common()),
        "easy_negative_source_distribution": dict(easy_sources.most_common()),
        "unavailable_positive_source_distribution": dict(unavailable_positive.most_common()),
    }

def scan_double_negatives(path: Path, con: sqlite3.Connection, samples: dict[str, list]) -> dict:
    stats = Counter()
    types = Counter()
    started = time.monotonic()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            stats["lines"] += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                stats["invalid_json"] += 1
                add_sample(samples, "double_negative_invalid_json", {
                    "line": line_number,
                    "error": str(exc),
                })
                continue

            rid = str(row.get("id", "")).strip()
            origin = str(row.get("origin_record_id", "")).strip()
            negative_1 = row.get("negative_1", {}) or {}
            negative_2 = row.get("negative_2", {}) or {}

            if rid and not register_seen(con, "double_negative_id", rid):
                stats["duplicate_id"] += 1

            if not negative_1:
                stats["missing_negative_1"] += 1

            if not negative_2:
                stats["missing_negative_2"] += 1

            if negative_1:
                types[str(negative_1.get("type", "<missing>"))] += 1

            if negative_2:
                types[str(negative_2.get("type", "<missing>"))] += 1

            if not origin or con.execute(
                "SELECT 1 FROM sft WHERE id=?",
                (origin,),
            ).fetchone() is None:
                stats["orphan_origin"] += 1
                add_sample(samples, "double_negative_orphan", {
                    "id": rid,
                    "origin_record_id": origin,
                })

            stats["valid_records"] += 1

            if stats["lines"] % 50000 == 0:
                progress("Double negative", stats["lines"], started)

    print()

    return {
        **dict(stats),
        "negative_type_distribution": dict(types.most_common()),
    }

def scan_negative_sft(path: Path, con: sqlite3.Connection, samples: dict[str, list]) -> dict:
    stats = Counter()
    types = Counter()
    sources = Counter()
    started = time.monotonic()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            stats["lines"] += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                stats["invalid_json"] += 1
                add_sample(samples, "negative_sft_invalid_json", {
                    "line": line_number,
                    "error": str(exc),
                })
                continue

            rid = str(row.get("id", "")).strip()
            metadata = row.get("metadata", {}) or {}
            origin = str(metadata.get("origin_record_id", "")).strip()
            negative_type = str(metadata.get("negative_type", "")).strip()
            source = str(metadata.get("source", "")).strip()
            confidence = float(metadata.get("negative_validation_confidence", 0.0) or 0.0)
            validated = bool(metadata.get("negative_answerability_validated", False))

            types[negative_type or "<missing>"] += 1
            sources[source or "<missing>"] += 1

            if rid and not register_seen(con, "negative_sft_id", rid):
                stats["duplicate_id"] += 1

            if not origin or con.execute(
                "SELECT 1 FROM sft WHERE id=?",
                (origin,),
            ).fetchone() is None:
                stats["orphan_origin"] += 1
                add_sample(samples, "negative_sft_orphan", {
                    "id": rid,
                    "origin_record_id": origin,
                })

            if not validated:
                stats["not_validated"] += 1
                add_sample(samples, "negative_sft_not_validated", {
                    "id": rid,
                    "origin_record_id": origin,
                    "confidence": confidence,
                })

            if confidence < 0.85:
                stats["confidence_below_085"] += 1

            stats["valid_records"] += 1

            if stats["lines"] % 50000 == 0:
                progress("Negative SFT", stats["lines"], started)

    print()

    return {
        **dict(stats),
        "negative_type_distribution": dict(types.most_common()),
        "source_distribution": dict(sources.most_common()),
    }

def scan_rejected(path: Path, samples: dict[str, list]) -> dict:
    stats = Counter()
    reasons = Counter()
    sources = Counter()
    started = time.monotonic()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            stats["lines"] += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                stats["invalid_json"] += 1
                add_sample(samples, "rejected_invalid_json", {
                    "line": line_number,
                    "error": str(exc),
                })
                continue

            reason = str(row.get("reason", "<missing>")).strip() or "<missing>"
            source = str(row.get("source", "<missing>")).strip() or "<missing>"

            reasons[reason] += 1
            sources[source] += 1
            stats["valid_records"] += 1

            if stats["lines"] % 50000 == 0:
                progress("Rejected", stats["lines"], started)

    print()

    return {
        **dict(stats),
        "reason_distribution": dict(reasons.most_common()),
        "source_distribution": dict(sources.most_common()),
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "rag" / "config.yaml"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--unavailable-source", action="append", default=[])
    args = parser.parse_args()

    config_path = Path(args.config).resolve()

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    ds_cfg = cfg.get("dataset_creation", {}) or {}

    configured_data_dir = Path(str(ds_cfg.get("out_dir", "rag/data/training")))
    data_dir = Path(args.data_dir).resolve() if args.data_dir else (
        configured_data_dir if configured_data_dir.is_absolute()
        else (ROOT / configured_data_dir).resolve()
    )

    out_dir = Path(args.out_dir).resolve() if args.out_dir else data_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_path = data_dir / str(ds_cfg.get("sft_out", "genshin_rag_sft_candidates.jsonl"))
    retrieval_path = data_dir / str(ds_cfg.get("retrieval_pairs_out", "genshin_retrieval_pairs.jsonl"))
    double_path = data_dir / str(ds_cfg.get("double_negative_out", "genshin_double_negative_pairs.jsonl"))
    negative_sft_path = data_dir / str(ds_cfg.get("sft_negative_out", "genshin_sft_negative_answerability.jsonl"))
    rejected_path = data_dir / str(ds_cfg.get("rejected_out", "genshin_rejected.jsonl"))

    required = [sft_path, retrieval_path, double_path, negative_sft_path, rejected_path]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    configured_enabled = [
        str(source.get("name"))
        for source in cfg.get("sources", [])
        if bool(source.get("enabled", False))
    ]

    configured_disabled = [
        str(source.get("name"))
        for source in cfg.get("sources", [])
        if not bool(source.get("enabled", False))
    ]

    dataset_sources = parse_sources(ds_cfg.get("sources"))
    unavailable_sources = set(args.unavailable_source)

    db_path = out_dir / "_dataset_audit.sqlite3"
    con = open_db(db_path)
    samples: dict[str, list] = defaultdict(list)

    print(f"Data directory: {data_dir}")
    print(f"Unavailable now: {sorted(unavailable_sources) or 'none'}")
    print()

    print("Scanning SFT...")
    sft_stats = scan_sft(sft_path, con, samples, unavailable_sources)

    print("Scanning retrieval pairs...")
    retrieval_stats = scan_retrieval(retrieval_path, con, samples, unavailable_sources)

    print("Scanning double-negative pairs...")
    double_stats = scan_double_negatives(double_path, con, samples)

    print("Scanning negative-answerability SFT...")
    negative_sft_stats = scan_negative_sft(negative_sft_path, con, samples)

    print("Scanning rejected records...")
    rejected_stats = scan_rejected(rejected_path, samples)

    missing_retrieval = con.execute("""
        SELECT COUNT(*)
        FROM sft s
        LEFT JOIN retrieval r ON r.origin_id = s.id
        WHERE r.origin_id IS NULL
    """).fetchone()[0]

    missing_by_source = dict(
        con.execute("""
            SELECT COALESCE(NULLIF(s.source, ''), '<missing>'), COUNT(*)
            FROM sft s
            LEFT JOIN retrieval r ON r.origin_id = s.id
            WHERE r.origin_id IS NULL
            GROUP BY s.source
            ORDER BY COUNT(*) DESC
        """).fetchall()
    )

    sft_total = int(sft_stats.get("valid_records", 0))
    retrieval_total = int(retrieval_stats.get("valid_records", 0))
    matched = max(sft_total - int(missing_retrieval), 0)
    coverage = (matched / sft_total * 100.0) if sft_total else 0.0

    source_distribution = sft_stats.get("source_distribution", {})
    unavailable_sft_count = sum(
        int(source_distribution.get(source, 0))
        for source in unavailable_sources
    )
    unavailable_pct = (
        unavailable_sft_count / sft_total * 100.0
        if sft_total else 0.0
    )

    dataset_sources_present = sorted(
        source
        for source in source_distribution
        if source != "<missing>"
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "data_dir": str(data_dir),
        "configuration": {
            "dataset_sources": dataset_sources,
            "configured_enabled_sources": configured_enabled,
            "configured_disabled_sources": configured_disabled,
            "unavailable_sources_now": sorted(unavailable_sources),
            "dataset_sources_present": dataset_sources_present,
        },
        "files": {
            "sft": {
                "path": str(sft_path),
                **sft_stats,
            },
            "retrieval_pairs": {
                "path": str(retrieval_path),
                **retrieval_stats,
            },
            "double_negative_pairs": {
                "path": str(double_path),
                **double_stats,
            },
            "negative_answerability_sft": {
                "path": str(negative_sft_path),
                **negative_sft_stats,
            },
            "rejected": {
                "path": str(rejected_path),
                **rejected_stats,
            },
        },
        "cross_file": {
            "sft_records": sft_total,
            "retrieval_pairs": retrieval_total,
            "sft_with_retrieval": matched,
            "sft_without_retrieval": missing_retrieval,
            "retrieval_coverage_percent": round(coverage, 4),
            "sft_without_retrieval_by_source": missing_by_source,
        },
        "source_transition": {
            "unavailable_source_sft_records": unavailable_sft_count,
            "unavailable_source_sft_percent": round(unavailable_pct, 4),
            "unavailable_sources": sorted(unavailable_sources),
        },
        "interpretation": {
            "sft_without_retrieval": "Warning only. dataset_creation.py may emit an SFT record without a retrieval pair when no validated hard/easy negatives were available.",
            "orphan_retrieval_pair": "Structural error. A retrieval pair references an SFT origin_record_id that does not exist.",
            "positive_source_mismatch": "Structural/suspicious error. SFT metadata source differs from retrieval positive source.",
            "duplicate_sft_id": "Structural error.",
            "duplicate_question": "Suspicious; may be legitimate only if intentionally duplicated.",
            "unavailable_source": "Historical provenance warning, not automatically a bad record. The fact may still be corroborated by other sources.",
        },
        "samples": dict(samples),
    }

    report_path = out_dir / "dataset_audit_report.json"
    samples_path = out_dir / "dataset_audit_samples.json"

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    with samples_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(samples), handle, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print("DATASET AUDIT SUMMARY")
    print("=" * 70)
    print(f"SFT records:                {sft_total:,}")
    print(f"Retrieval pairs:            {retrieval_total:,}")
    print(f"SFT with retrieval:         {matched:,}")
    print(f"SFT without retrieval:      {missing_retrieval:,}")
    print(f"Retrieval coverage:         {coverage:.2f}%")
    print(f"Duplicate SFT IDs:          {sft_stats.get('duplicate_id', 0):,}")
    print(f"Duplicate questions:        {sft_stats.get('duplicate_question', 0):,}")
    print(f"Malformed SFT JSON:         {sft_stats.get('invalid_json', 0):,}")
    print(f"Orphan retrieval pairs:     {retrieval_stats.get('orphan_retrieval_pair', 0):,}")
    print(f"Source mismatches:          {retrieval_stats.get('positive_source_mismatch', 0):,}")
    print(f"Query mismatches:           {retrieval_stats.get('query_mismatch', 0):,}")
    print(f"Unavailable-source SFT:     {unavailable_sft_count:,} ({unavailable_pct:.2f}%)")
    print()
    print("SFT source distribution:")
    for source, count in source_distribution.items():
        pct = count / sft_total * 100.0 if sft_total else 0.0
        marker = " [UNAVAILABLE NOW]" if source in unavailable_sources else ""
        print(f"  {source:20s} {count:10,d}  {pct:6.2f}%{marker}")

    print()
    print("Missing retrieval by source:")
    for source, count in missing_by_source.items():
        print(f"  {source:20s} {count:10,d}")

    print()
    print(f"Report:  {report_path}")
    print(f"Samples: {samples_path}")

    con.close()

    try:
        db_path.unlink()
    except OSError:
        pass

if __name__ == "__main__":
    main()