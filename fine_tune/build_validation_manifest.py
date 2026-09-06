from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

def norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def assistant_answer(row: dict) -> str:
    for message in reversed(row.get("messages", []) or []):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="fine_tune/data/audit/validation_manifest.jsonl")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    sft_path = data_dir / "genshin_rag_sft_candidates.jsonl"
    retrieval_path = data_dir / "genshin_retrieval_pairs.jsonl"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect("/tmp/genshin_validation_manifest.sqlite3")
    db.execute("DROP TABLE IF EXISTS retrieval")
    db.execute("DROP TABLE IF EXISTS questions")
    db.execute("CREATE TABLE retrieval(origin_id TEXT PRIMARY KEY)")
    db.execute("CREATE TABLE questions(q TEXT, id TEXT, answer TEXT, PRIMARY KEY(q,id))")

    print("Indexing retrieval pairs...")
    with retrieval_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            origin = str(row.get("origin_record_id", "")).strip()
            if origin:
                db.execute("INSERT OR IGNORE INTO retrieval VALUES (?)", (origin,))
            if i % 50000 == 0:
                db.commit()
                print(f"\rRetrieval: {i:,}", end="", flush=True)
    db.commit()
    print()

    print("Indexing questions...")
    with sft_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata", {}) or {}
            rid = str(row.get("id", "")).strip()
            question = str(metadata.get("question", "")).strip()
            answer = str(metadata.get("reference_answer", "")).strip()
            if rid and question:
                db.execute("INSERT OR IGNORE INTO questions VALUES (?,?,?)", (norm(question), rid, norm(answer)))
            if i % 50000 == 0:
                db.commit()
                print(f"\rSFT index: {i:,}", end="", flush=True)
    db.commit()
    print()

    counts = Counter()

    print("Building manifest...")
    with sft_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for i, line in enumerate(src, 1):
            if not line.strip():
                continue

            row = json.loads(line)
            metadata = row.get("metadata", {}) or {}
            rid = str(row.get("id", "")).strip()
            source = str(metadata.get("source", "")).strip()
            question = str(metadata.get("question", "")).strip()
            reference = str(metadata.get("reference_answer", "")).strip()
            assistant = assistant_answer(row)

            has_retrieval = db.execute("SELECT 1 FROM retrieval WHERE origin_id=?", (rid,)).fetchone() is not None

            duplicates = db.execute(
                "SELECT id,answer FROM questions WHERE q=? AND id<>?",
                (norm(question), rid),
            ).fetchall()

            duplicate_answers = {answer for _other_id, answer in duplicates if answer}
            duplicate_conflict = bool(duplicate_answers and any(answer != norm(reference) for answer in duplicate_answers))

            flags = []

            if source == "honey":
                flags.append("source_unavailable")
            if not has_retrieval:
                flags.append("missing_retrieval_pair")
            if duplicate_conflict:
                flags.append("duplicate_question_answer_conflict")
            elif duplicates:
                flags.append("duplicate_question")
            if not question:
                flags.append("missing_question")
            if not reference:
                flags.append("missing_reference_answer")
            if not assistant:
                flags.append("missing_assistant_answer")

            if any(flag in flags for flag in ["missing_question", "missing_reference_answer", "missing_assistant_answer"]):
                risk = "critical"
            elif "duplicate_question_answer_conflict" in flags:
                risk = "high"
            elif "source_unavailable" in flags:
                risk = "high"
            elif "missing_retrieval_pair" in flags:
                risk = "medium"
            elif "duplicate_question" in flags:
                risk = "medium"
            else:
                risk = "low"

            counts[risk] += 1

            result = {
                "record_id": rid,
                "source": source,
                "question": question,
                "reference_answer": reference,
                "assistant_answer": assistant,
                "has_retrieval_pair": has_retrieval,
                "duplicate_count": len(duplicates),
                "flags": flags,
                "risk": risk,
            }

            dst.write(json.dumps(result, ensure_ascii=False) + "\n")

            if i % 50000 == 0:
                print(f"\rManifest: {i:,}", end="", flush=True)

    print()
    print("Risk distribution:")
    for key in ["critical", "high", "medium", "low"]:
        print(f"  {key:10s} {counts[key]:,}")

    print(f"\nManifest: {out_path}")
    db.close()

if __name__ == "__main__":
    main()