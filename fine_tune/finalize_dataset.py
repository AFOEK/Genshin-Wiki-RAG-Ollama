from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(f, row: dict):
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def replace_assistant(row: dict[str, Any], answer: str) -> dict[str, Any]:
    row = copy.deepcopy(row)
    messages = row.get("messages", []) or []

    for message in reversed(messages):
        if str(message.get("role", "")).strip().lower() == "assistant":
            message["content"] = answer
            return row

    messages.append({"role": "assistant", "content": answer})
    row["messages"] = messages
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="fine_tune/data")
    ap.add_argument("--semantic", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--duplicate", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--auto-repair-assistant",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--allow-unavailable-source",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    sft_path = data_dir / "genshin_rag_sft_candidates.jsonl"
    retrieval_path = data_dir / "genshin_retrieval_pairs.jsonl"

    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else data_dir / "audit" / "validation_manifest.jsonl"
    )

    semantic_path = (
        Path(args.semantic)
        if args.semantic
        else data_dir / "audit" / "semantic_validation_context_v3.jsonl"
    )

    duplicate_path = (
        Path(args.duplicate)
        if args.duplicate
        else data_dir / "audit" / "duplicate_validation.jsonl"
    )

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else data_dir / "clean"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading manifest...")
    manifest = {}
    for row in read_jsonl(manifest_path):
        rid = str(row.get("record_id", "")).strip()
        if rid:
            manifest[rid] = row

    print("Loading semantic v3...")
    semantic = {}
    for row in read_jsonl(semantic_path):
        rid = str(row.get("record_id", "")).strip()
        if rid:
            semantic[rid] = row

    print(f"Semantic records: {len(semantic):,}")

    print("Loading duplicate decisions...")
    duplicate_state = {}

    if duplicate_path.exists():
        for row in read_jsonl(duplicate_path):
            result = row.get("result", {}) or {}
            relation = str(result.get("relation", "")).strip().lower()
            preferred = str(
                result.get("preferred_record_id", "")
            ).strip()

            for rid in row.get("record_ids", []) or []:
                duplicate_state[str(rid)] = {
                    "relation": relation,
                    "preferred_record_id": preferred,
                    "group_id": row.get("group_id"),
                }

    print(f"Duplicate decisions: {len(duplicate_state):,} records")

    paths = {
        "clean": out_dir / "genshin_sft_clean.jsonl",
        "repair": out_dir / "genshin_sft_repair.jsonl",
        "review": out_dir / "genshin_sft_review.jsonl",
        "quarantine": out_dir / "genshin_sft_quarantine.jsonl",
        "unvalidated": out_dir / "genshin_sft_unvalidated.jsonl",
        "retrieval_clean": out_dir / "genshin_retrieval_clean.jsonl",
        "retrieval_regenerate": out_dir / "genshin_retrieval_regenerate.jsonl",
        "retrieval_unvalidated": out_dir / "genshin_retrieval_unvalidated.jsonl",
        "retrieval_missing": out_dir / "genshin_retrieval_missing.jsonl",
        "summary": out_dir / "validation_summary.json",
    }

    counts = Counter()
    risk_counts = Counter()
    repair_counts = Counter()
    retrieval_status = {}

    with ExitStack() as stack:
        outputs = {
            name: stack.enter_context(
                path.open("w", encoding="utf-8")
            )
            for name, path in paths.items()
            if name not in {
                "retrieval_clean",
                "retrieval_regenerate",
                "retrieval_unvalidated",
                "retrieval_missing",
                "summary",
            }
        }

        print("Finalizing SFT...")

        for i, row in enumerate(read_jsonl(sft_path), 1):
            rid = str(row.get("id", "")).strip()
            metadata = row.get("metadata", {}) or {}

            man = manifest.get(rid, {})
            semrow = semantic.get(rid)

            risk = str(
                (semrow or {}).get(
                    "risk",
                    man.get("risk", "unknown"),
                )
            )

            flags = list(
                (semrow or {}).get(
                    "risk_flags",
                    man.get("flags", []),
                )
                or []
            )

            risk_counts[risk] += 1

            # -------------------------------------------------
            # Unvalidated records are NOT silently accepted.
            # -------------------------------------------------
            if semrow is None:
                final = copy.deepcopy(row)
                final.setdefault("metadata", {})[
                    "finalization"
                ] = {
                    "status": "unvalidated",
                    "risk": risk,
                    "risk_flags": flags,
                    "sft_usable": False,
                    "reference_usable": None,
                    "retrieval_usable": None,
                }

                write_jsonl(outputs["unvalidated"], final)
                counts["unvalidated"] += 1

                retrieval_status[rid] = {
                    "validated": False,
                    "usable": False,
                }
                continue

            s = semrow.get("semantic", {}) or {}
            actions = set(
                semrow.get("recommended_actions", []) or []
            )

            positive_ok = bool(
                s.get("positive_context_answerable", False)
            )

            reference_ok = bool(
                s.get("reference_supported", False)
            )

            assistant_ok = (
                bool(s.get("assistant_supported", False))
                and not bool(
                    s.get(
                        "assistant_has_unsupported_extras",
                        False,
                    )
                )
            )

            negative_ok = not bool(
                s.get("negative_leakage", False)
            )

            verdict = str(
                s.get("verdict", "")
            ).strip().lower()

            has_retrieval = bool(
                semrow.get(
                    "has_retrieval_pair",
                    man.get("has_retrieval_pair", False),
                )
            )

            source_unavailable = (
                "source_unavailable" in flags
            )

            # Retrieval eligibility is intentionally independent
            # from SFT/reference eligibility.
            retrieval_usable = (
                has_retrieval
                and positive_ok
                and negative_ok
                and verdict != "review"
            )

            duplicate = duplicate_state.get(rid, {})
            dup_relation = duplicate.get("relation", "")
            dup_preferred = duplicate.get(
                "preferred_record_id", ""
            )

            duplicate_block = False

            if dup_relation == "ambiguous":
                duplicate_block = True

            elif dup_relation == "contradictory":
                if not dup_preferred or rid != dup_preferred:
                    duplicate_block = True

            # -------------------------------------------------
            # Human-review / duplicate ambiguity
            # -------------------------------------------------
            needs_review = (
                verdict == "review"
                or "human_review" in actions
                or duplicate_block
            )

            auto_repaired = False
            repair_reason = None
            final = copy.deepcopy(row)

            # -------------------------------------------------
            # Determine SFT eligibility
            # -------------------------------------------------
            if needs_review:
                status = "review"
                sft_usable = False

            elif not positive_ok:
                status = "repair"
                sft_usable = False
                repair_reason = "positive_context"

            elif assistant_ok:
                # Reference may be wrong. That's okay for SFT if
                # context -> assistant is independently supported.
                status = "clean"
                sft_usable = True

            elif (
                args.auto_repair_assistant
                and reference_ok
                and metadata.get("reference_answer")
            ):
                # Deterministic repair:
                # supported reference replaces bad assistant.
                final = replace_assistant(
                    final,
                    str(metadata["reference_answer"]).strip(),
                )
                status = "clean"
                sft_usable = True
                auto_repaired = True
                repair_reason = "assistant_from_reference"
                repair_counts[
                    "assistant_from_reference"
                ] += 1

            else:
                status = "repair"
                sft_usable = False
                repair_reason = "assistant_and_or_reference"

            # Current unavailable source policy (Honey).
            # Do this only after determining whether the row
            # itself needs semantic repair.
            if (
                status == "clean"
                and source_unavailable
                and not args.allow_unavailable_source
            ):
                status = "quarantine"
                sft_usable = False

            final_metadata = final.setdefault(
                "metadata", {}
            )

            final_metadata["finalization"] = {
                "status": status,
                "risk": risk,
                "risk_flags": flags,
                "semantic_verdict": verdict,
                "semantic_confidence": float(
                    s.get("confidence", 0.0) or 0.0
                ),
                "sft_usable": sft_usable,
                "reference_usable": reference_ok,
                "retrieval_usable": retrieval_usable,
                "auto_repaired_assistant": auto_repaired,
                "repair_reason": repair_reason,
                "recommended_actions": sorted(actions),
                "duplicate_relation": dup_relation or None,
                "duplicate_preferred_record_id": (
                    dup_preferred or None
                ),
            }

            write_jsonl(outputs[status], final)
            counts[status] += 1

            retrieval_status[rid] = {
                "validated": True,
                "usable": retrieval_usable,
                "negative_leakage": not negative_ok,
                "has_retrieval": has_retrieval,
                "positive_ok": positive_ok,
                "verdict": verdict,
            }

            if i % 10000 == 0:
                print(
                    f"\rSFT: {i:,}",
                    end="",
                    flush=True,
                )

        print()

    # ---------------------------------------------------------
    # Retrieval outputs
    # ---------------------------------------------------------
    print("Finalizing retrieval pairs...")

    seen_retrieval = set()

    with (
        paths["retrieval_clean"].open(
            "w", encoding="utf-8"
        ) as clean_f,
        paths["retrieval_regenerate"].open(
            "w", encoding="utf-8"
        ) as regen_f,
        paths["retrieval_unvalidated"].open(
            "w", encoding="utf-8"
        ) as unvalidated_f,
    ):
        for i, pair in enumerate(
            read_jsonl(retrieval_path), 1
        ):
            rid = str(
                pair.get("origin_record_id", "")
            ).strip()

            seen_retrieval.add(rid)

            state = retrieval_status.get(rid)

            if state is None or not state["validated"]:
                write_jsonl(unvalidated_f, pair)
                counts["retrieval_unvalidated"] += 1

            elif state["usable"]:
                write_jsonl(clean_f, pair)
                counts["retrieval_clean"] += 1

            else:
                pair = copy.deepcopy(pair)
                pair["finalization"] = state
                write_jsonl(regen_f, pair)
                counts["retrieval_regenerate"] += 1

            if i % 50000 == 0:
                print(
                    f"\rRetrieval: {i:,}",
                    end="",
                    flush=True,
                )

    print()

    # Missing retrieval pairs for validated SFT records
    with paths["retrieval_missing"].open(
        "w", encoding="utf-8"
    ) as f:
        for rid, state in retrieval_status.items():
            if (
                state.get("validated")
                and state.get("has_retrieval") is False
            ):
                write_jsonl(
                    f,
                    {
                        "record_id": rid,
                        "reason": "missing_retrieval_pair",
                    },
                )
                counts["retrieval_missing"] += 1

    summary = {
        "inputs": {
            "sft": str(sft_path),
            "retrieval": str(retrieval_path),
            "manifest": str(manifest_path),
            "semantic": str(semantic_path),
            "duplicate": (
                str(duplicate_path)
                if duplicate_path.exists()
                else None
            ),
        },
        "policy": {
            "semantic_version": "context_v3",
            "auto_repair_assistant": (
                args.auto_repair_assistant
            ),
            "allow_unavailable_source": (
                args.allow_unavailable_source
            ),
            "unvalidated_rows_are_trainable": False,
        },
        "counts": dict(counts),
        "risk_distribution": dict(risk_counts),
        "automatic_repairs": dict(repair_counts),
    }

    with paths["summary"].open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nFinalization complete")
    print("=" * 60)

    for key in [
        "clean",
        "repair",
        "review",
        "quarantine",
        "unvalidated",
    ]:
        print(f"{key:15s} {counts[key]:10,d}")

    print()
    print(
        f"{'retrieval_clean':25s}"
        f"{counts['retrieval_clean']:10,d}"
    )
    print(
        f"{'retrieval_regenerate':25s}"
        f"{counts['retrieval_regenerate']:10,d}"
    )
    print(
        f"{'retrieval_unvalidated':25s}"
        f"{counts['retrieval_unvalidated']:10,d}"
    )
    print(
        f"{'retrieval_missing':25s}"
        f"{counts['retrieval_missing']:10,d}"
    )

    print(
        "\nPEFT input:",
        paths["clean"],
    )
    print(
        "Summary:",
        paths["summary"],
    )


if __name__ == "__main__":
    main()