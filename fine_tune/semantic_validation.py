from __future__ import annotations

import argparse
import json
import sys
import sqlite3
import random
import yaml

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fine_tune"))
sys.path.insert(0, str(ROOT / "fine_tune" / "internet_validation"))

from internet_validation.oracle import ollama_structured
from validation_common import assistant_answer, positive_context

SCHEMA = {
    "type": "object",
    "properties": {
        "reference_supported": {"type": "boolean"},
        "assistant_supported": {"type": "boolean"},
        "assistant_has_unsupported_extras": {"type": "boolean"},
        "positive_context_answerable": {"type": "boolean"},
        "negative_leakage": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "verdict": {"type": "string", "enum": ["pass", "review", "fail"]},
        "reason": {"type": "string"},
    },
    "required": [
        "reference_supported",
        "assistant_supported",
        "assistant_has_unsupported_extras",
        "positive_context_answerable",
        "negative_leakage",
        "confidence",
        "verdict",
        "reason",
    ],
}

SYSTEM = (
    "You are a strict dataset quality validator. "
    "Judge only from the supplied dataset evidence. "
    "Do not use outside knowledge. "
    "The positive context must support the correct answer. "
    "The assistant answer must not introduce unsupported claims. "
    "Negative contexts must not contain enough information to answer the question."
)

def load_retrieval(path: Path) -> dict[str, dict]:
    result = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            origin = str(row.get("origin_record_id", "")).strip()
            if origin:
                result[origin] = row
    return result

def derive_actions(result: dict) -> list[str]:
    if not bool(result.get("positive_context_answerable", False)):
        return ["regenerate_positive_context"]

    actions = []

    if not bool(result.get("reference_supported", False)):
        actions.append("repair_reference")

    if not bool(result.get("assistant_supported", False)) or bool(result.get("assistant_has_unsupported_extras", False)):
        actions.append("regenerate_assistant_answer")

    if bool(result.get("negative_leakage", False)):
        actions.append("regenerate_negatives")

    verdict = str(result.get("verdict", "")).lower()
    if verdict == "review":
        actions.append("human_review")
    elif verdict == "fail" and not actions:
        actions.append("human_review")

    return actions or ["keep"]

def build_retrieval_index(path: Path, db_path: Path) -> sqlite3.Connection:
    if db_path.exists() and db_path.stat().st_mtime >= path.stat().st_mtime:
        con = sqlite3.connect(db_path)
        table = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='retrieval'").fetchone()
        if table:
            print(f"Reusing retrieval index: {db_path}")
            return con
        con.close()

    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("CREATE TABLE retrieval(origin_id TEXT PRIMARY KEY, payload TEXT)")

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue

            row = json.loads(line)
            origin = str(row.get("origin_record_id", "")).strip()

            if origin:
                con.execute(
                    "INSERT OR REPLACE INTO retrieval VALUES (?,?)",
                    (origin, json.dumps(row, ensure_ascii=False)),
                )

            if i % 10000 == 0:
                con.commit()
                print(f"\rRetrieval index: {i:,}", end="", flush=True)

    con.commit()
    print()
    return con

def validate_record(cfg: dict, row: dict, retrieval: dict | None) -> dict:
    metadata = row.get("metadata", {}) or {}

    question = str(metadata.get("question", "")).strip()
    reference = str(metadata.get("reference_answer", "")).strip()
    assistant = assistant_answer(row)

    positive_text = positive_context(row)
    negatives = []

    if retrieval:
        for negative in retrieval.get("hard_negatives", []) or []:
            negatives.append({
                "type": "hard",
                "title": str(negative.get("title", "")),
                "text": str(negative.get("text", ""))[:1800],
            })

        for negative in retrieval.get("easy_negatives", []) or []:
            negatives.append({
                "type": "easy",
                "title": str(negative.get("title", "")),
                "text": str(negative.get("text", ""))[:1800],
            })

        if not positive_text:
            positive = retrieval.get("positive", {}) or {}
            positive_text = str(positive.get("text", "")).strip()

    payload = {
        "question": question,
        "reference_answer": reference,
        "assistant_answer": assistant,
        "positive_context": positive_text[:125000],
        "negative_contexts": negatives,
    }

    validation_cfg = cfg["internet_validation"]

    prompt = (
        "Validate this training record.\n\n"
        "Rules:\n"
        "1. The positive context must contain enough information to answer the question.\n"
        "2. The reference answer must be supported by the positive context.\n"
        "3. The assistant answer must be supported by the positive context.\n"
        "4. The assistant must not add unsupported factual details.\n"
        "5. Negative contexts must not contain enough information to answer the question.\n"
        "6. If evidence is ambiguous, return review rather than pass.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    return ollama_structured(
        ollama_url=validation_cfg["ollama_url"],
        model=validation_cfg["ollama_model"],
        system=SYSTEM,
        prompt=prompt,
        schema=SCHEMA,
        timeout_s=float(validation_cfg.get("ollama_timeout_s", 360)),
        num_ctx=int(validation_cfg.get("ollama_num_ctx", 8192)),
        num_predict=768,
        num_thread=int(validation_cfg.get("ollama_num_thread", 32)),
    )

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--risk", choices=["critical", "high", "medium", "low", "all"], default="high")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "audit" / "validation_manifest.jsonl"
    sft_path = data_dir / "genshin_rag_sft_candidates.jsonl"
    retrieval_path = data_dir / "genshin_retrieval_pairs.jsonl"
    output_path = Path(args.out) if args.out else data_dir / "audit" / "semantic_validation_v2.jsonl"

    with (ROOT / "rag" / "config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    completed = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed.add(str(json.loads(line).get("record_id", "")))

    eligible = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            manifest_row = json.loads(line)
            if args.risk != "all" and manifest_row.get("risk") != args.risk:
                continue

            if manifest_row["record_id"] in completed:
                continue

            eligible.append(manifest_row)

    if args.shuffle:
        random.Random(args.seed).shuffle(eligible)

    if args.limit:
        eligible = eligible[:args.limit]

    wanted = {row["record_id"]: row for row in eligible}
    print(f"Selected records: {len(wanted):,}")
    db = build_retrieval_index(retrieval_path, data_dir / "audit" / "_semantic.sqlite3",)
    processed = 0
    with sft_path.open("r", encoding="utf-8") as src, output_path.open("a", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue

            row = json.loads(line)
            rid = str(row.get("id", "")).strip()
            if rid not in wanted or rid in completed:
                continue

            manifest_row = wanted[rid]
            db_row = db.execute("SELECT payload FROM retrieval WHERE origin_id=?", (rid,),).fetchone()
            retrieval = json.loads(db_row[0]) if db_row else None
            try:
                result = validate_record(cfg, row, retrieval)
            except Exception as exc:
                result = {
                    "reference_supported": False,
                    "assistant_supported": False,
                    "assistant_has_unsupported_extras": False,
                    "positive_context_answerable": False,
                    "negative_leakage": False,
                    "confidence": 0.0,
                    "verdict": "review",
                    "reason": f"validator_error: {type(exc).__name__}: {exc}",
                }

            output = {
                "record_id": rid,
                "source": str((row.get("metadata", {}) or {}).get("source", "")),
                "risk": manifest_row.get("risk"),
                "risk_flags": manifest_row.get("flags", []),
                "has_retrieval_pair": retrieval is not None,
                "semantic": result,
                "recommended_actions": derive_actions(result),
            }

            dst.write(json.dumps(output, ensure_ascii=False) + "\n")
            dst.flush()
            processed += 1
            print(f"\rValidated: {processed:,} {rid} → {result['verdict']} {result['confidence']:.3f}", end="", flush=True,)

    print()
    db.close()

if __name__ == "__main__":
    main()