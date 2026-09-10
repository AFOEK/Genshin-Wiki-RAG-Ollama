from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fine_tune"))
sys.path.insert(0, str(ROOT / "fine_tune" / "internet_validation"))

from validation_common import normalize_answer, normalize_question
from internet_validation.oracle import ollama_structured

SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {
            "type": "string",
            "enum": ["equivalent", "compatible", "contradictory", "ambiguous"],
        },
        "preferred_record_id": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["relation", "preferred_record_id", "confidence", "reason"],
}

SYSTEM = (
    "You compare answers written for exactly the same question. "
    "Equivalent means they express the same factual answer. "
    "Compatible means one contains additional information but they do not conflict. "
    "Contradictory means both cannot simultaneously be true. "
    "Ambiguous means the relationship cannot be determined. "
    "Do not use outside knowledge."
)

def group_id(question: str) -> str:
    return hashlib.sha256(normalize_question(question).encode()).hexdigest()[:16]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with (ROOT / "rag" / "config.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    validation_cfg = cfg["internet_validation"]
    groups = defaultdict(list)

    with Path(args.manifest).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("duplicate_count", 0)) > 0:
                groups[normalize_question(row["question"])].append(row)

    candidates = []
    for question_key, rows in groups.items():
        answers = {normalize_answer(row.get("reference_answer", "")) for row in rows}
        if len(answers) > 1:
            candidates.append(rows)

    random.Random(args.seed).shuffle(candidates)

    if args.limit:
        candidates = candidates[:args.limit]

    out_path = Path(args.out)
    completed = set()

    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed.add(json.loads(line)["group_id"])

    with out_path.open("a", encoding="utf-8") as dst:
        for i, rows in enumerate(candidates, 1):
            gid = group_id(rows[0]["question"])

            if gid in completed:
                continue

            payload = {
                "question": rows[0]["question"],
                "answers": [
                    {
                        "record_id": row["record_id"],
                        "source": row["source"],
                        "answer": row["reference_answer"],
                    }
                    for row in rows
                ],
            }

            try:
                result = ollama_structured(
                    ollama_url=validation_cfg["ollama_url"],
                    model=validation_cfg["ollama_model"],
                    system=SYSTEM,
                    prompt=json.dumps(payload, ensure_ascii=False),
                    schema=SCHEMA,
                    timeout_s=float(validation_cfg.get("ollama_timeout_s", 360)),
                    num_ctx=int(validation_cfg.get("ollama_num_ctx", 8192)),
                    num_predict=1024,
                    num_thread=int(validation_cfg.get("ollama_num_thread", 32)),
                )
            except Exception as exc:
                result = {
                    "relation": "ambiguous",
                    "preferred_record_id": "",
                    "confidence": 0.0,
                    "reason": f"validator_error: {type(exc).__name__}: {exc}",
                }

            output = {
                "group_id": gid,
                "question": rows[0]["question"],
                "record_ids": [row["record_id"] for row in rows],
                "result": result,
                "validator_error": result.get("reason", "").startswith("validator_error:"),
            }

            dst.write(json.dumps(output, ensure_ascii=False) + "\n")
            dst.flush()

            print(
                f"\r{i:,}/{len(candidates):,} "
                f"{result['relation']} {result['confidence']:.3f}",
                end="",
                flush=True,
            )

    print()

if __name__ == "__main__":
    main()