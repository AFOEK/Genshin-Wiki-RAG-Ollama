from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Iterator


log = logging.getLogger(__name__)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)

    with path.open("r", encoding="utf-8",) as handle:
        for line_number, line in enumerate(handle, start=1,):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}") from exc


def index_retrieval_records(retrieval_path: str | Path) -> dict[str, dict]:
    result: dict[str, dict] = {}

    for record in read_jsonl(retrieval_path):
        origin_id = str(
            record.get("origin_record_id", "")).strip()

        if not origin_id:
            continue

        result[origin_id] = record
    return result


def make_bundle(sft_record: dict, retrieval_record: dict) -> dict:
    metadata = (sft_record.get("metadata", {}) or {})
    messages = (sft_record.get("messages", []) or [])

    if not messages:
        raise ValueError(f"SFT record {sft_record.get('id')} has no messages")

    assistant_answer = ""

    for message in reversed(messages):
        if message.get("role") == "assistant":
            assistant_answer = str(message.get("content", "")).strip()
            break

    hard_negatives = [
        {
            **dict(negative),
            "negative_type": "hard",
        }
        for negative
        in retrieval_record.get(
            "hard_negatives",
            [],
        )
    ]

    easy_negatives = [
        {
            **dict(negative),
            "negative_type": "easy",
        }
        for negative
        in retrieval_record.get(
            "easy_negatives",
            [],
        )
    ]

    return {
        "record_id": str(
            sft_record["id"]
        ),
        "retrieval_record_id": str(
            retrieval_record.get(
                "id",
                "",
            )
        ),
        "question": str(
            metadata.get(
                "question",
                "",
            )
        ).strip(),
        "reference_answer": str(
            metadata.get(
                "reference_answer",
                "",
            )
        ).strip(),
        "assistant_answer": (
            assistant_answer
        ),
        "positive": dict(
            retrieval_record.get(
                "positive",
                {},
            )
        ),
        "negatives": [
            *hard_negatives,
            *easy_negatives,
        ],

        "sft_record": sft_record,
        "retrieval_record": retrieval_record,
    }


def iter_dataset_bundles(*, sft_path: str | Path, retrieval_path: str | Path) -> Iterator[dict]:
    retrieval_by_origin = (index_retrieval_records(retrieval_path))

    for sft_record in read_jsonl(sft_path):
        record_id = str(sft_record.get("id", "")).strip()

        if not record_id:
            log.warning("[DATASET_LOADER] Skipping SFT record without ID")
            continue

        retrieval_record = (retrieval_by_origin.get(record_id))

        if retrieval_record is None:
            log.warning("[DATASET_LOADER] No retrieval record for SFT ID=%s", record_id)
            continue

        bundle = make_bundle(sft_record, retrieval_record,)

        if not bundle["question"]:
            log.warning("[DATSET_LOADER] Skipping record without question ID=%s", record_id,)
            continue

        yield bundle