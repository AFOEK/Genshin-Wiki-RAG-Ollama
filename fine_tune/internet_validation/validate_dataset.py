from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from audit import run_dataset_audit
from dataset_loader import iter_dataset_bundles
from oracle import run_blind_oracle
from policy_loader import load_source_policies
from search import collect_parallel_evidence

log = logging.getLogger(__name__)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_bundle(cfg: dict, *, bundle: dict, policies: list, executor: ThreadPoolExecutor) -> dict:
    validation_cfg = cfg["internet_validation"]

    evidence = collect_parallel_evidence(
        executor=executor,
        question=bundle["question"],
        policies=policies,
        validation_cfg=validation_cfg)

    oracle_result = run_blind_oracle(cfg, question=bundle["question"], evidence=evidence)
    audit_result = run_dataset_audit(cfg, oracle_result=oracle_result, bundle=bundle)

    return {
        "record_id": bundle["record_id"],
        "retrieval_record_id": bundle["retrieval_record_id"],
        "question": bundle["question"],
        "evidence": evidence,
        "oracle": oracle_result,
        "audit": audit_result,
        "external_verified": (
            audit_result["verdict"] == "pass"
            and float(audit_result["confidence"]) >= float(validation_cfg.get("min_audit_confidence", 0.90))
        ),
        "human_verified": False,
        "validation_method": "searxng_ollama_blind_v1",
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config_path = Path("rag/config.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    validation_cfg = cfg["internet_validation"]
    policies = load_source_policies(cfg)
    source_workers = min(int(validation_cfg.get("source_workers", 5)), len(policies))
    sft_path = Path(validation_cfg["sft_dataset_path"])
    retrieval_path = Path(validation_cfg["retrieval_dataset_path"])
    output_path = Path(
        validation_cfg.get(
            "validation_output_path",
            "fine_tune/internet_validation/validation_results.jsonl",
        )
    )
    bundles = iter_dataset_bundles(sft_path=sft_path, retrieval_path=retrieval_path)

    with ThreadPoolExecutor(max_workers=source_workers, thread_name_prefix="validation-source") as executor:
        for index, bundle in enumerate(bundles, start=1):
            try:
                result = validate_bundle(
                    cfg,
                    bundle=bundle,
                    policies=policies,
                    executor=executor,
                )
            except Exception:
                log.exception("Validation failed ID=%s", bundle["record_id"])
                continue

            append_jsonl(output_path, result)
            log.info(
                "Validated #%d ID=%s verdict=%s confidence=%.3f",
                index,
                bundle["record_id"],
                result["audit"]["verdict"],
                float(result["audit"]["confidence"]),
            )


if __name__ == "__main__":
    main()