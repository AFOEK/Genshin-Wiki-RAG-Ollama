from __future__ import annotations

import json
from pathlib import Path


def load_verdicts(validation_path: Path) -> dict[str, bool]:
    verdicts: dict[str, bool] = {}
    with validation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            verdicts[str(row["record_id"])] = bool(row.get("external_verified", False))
    return verdicts


def apply_verdicts(sft_path: Path, validation_path: Path, out_path: Path) -> tuple[int, int]:
    verdicts = load_verdicts(validation_path)
    updated = 0
    total = 0

    with sft_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            record_id = str(record.get("id", ""))
            if record_id in verdicts:
                record.setdefault("metadata", {})["answer_support_validated"] = verdicts[record_id]
                updated += 1
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    return updated, total


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    updated, total = apply_verdicts(Path(args.sft), Path(args.validation), Path(args.out))
    print(f"Updated {updated}/{total} records with external validation verdicts -> {args.out}")