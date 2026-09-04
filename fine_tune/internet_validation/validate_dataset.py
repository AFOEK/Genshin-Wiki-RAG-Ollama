from __future__ import annotations

import json
import logging
import sys
import yaml
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag"))

from audit import run_dataset_audit
from dataset_loader import iter_dataset_bundles
from oracle import run_blind_oracle
from policy_loader import load_source_policies
from search import collect_parallel_evidence
from utils.logging_setup import setup_logging

log = logging.getLogger(__name__)

TIER_MODIFIER = {
    "primary": 1.00,
    "supplementary": 0.85
}

def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

def require_searxng(base_url: str, *, timeout_s: float = 5.0) -> None:
    base_url = base_url.rstrip("/")
    try:
        response = requests.get(f"{base_url}/config", timeout=timeout_s)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"[SearXNG] is not available at {base_url!r}. Start it before running internet validation.") from exc

    content_type = response.headers.get("content-type", "",).lower()

    if "json" not in content_type:
        raise RuntimeError(f"[SearXNG] /config did not return JSON: content_type={content_type!r}")

def evaluate_validation(*, validation_cfg: dict, bundle: dict, evidence: list[dict], oracle: dict, audit: dict) -> dict:
    evidence_by_id = {str(row["evidence_id"]): row for row in evidence}
    supporting: list[tuple[float, str, str]] = []
    contradiction_score = 0.0

    for judgement in oracle.get("evidence_judgements", []):
        row = evidence_by_id.get(str(judgement.get("evidence_id", "")))
        if row is None or not bool(row.get("fetch_ok", False)):
            continue

        relevance = max(0.0, min(1.0, float(judgement.get("relevance", 0.0))))
        tier_modifier = TIER_MODIFIER.get(str(row.get("tier", "")).lower(), 0.75)
        source_weight = max(0.0, min(1.0, float(row.get("source_weight", 0.5))))
        score = relevance * (0.85 + 0.15 * tier_modifier * source_weight)

        if bool(judgement.get("contradicts_answer", False)):
            contradiction_score = max(contradiction_score, score)

        if bool(judgement.get("supports_answer", False)):
            supporting.append((score, str(row.get("source", "")), str(row.get("tier", ""))))

    strong_primary = any(score >= 0.82 and tier == "primary" for score, _source, tier in supporting)
    independent_sources = {source for score, source, _tier in supporting if score >= 0.72}
    evidence_ok = strong_primary or len(independent_sources) >= 2

    expected_negative_ids = {str(negative.get("chunk_id", f"negative-{index}")) for index, negative in enumerate(bundle.get("negatives", []))}
    negative_results = audit.get("negative_results", [])
    actual_negative_ids = {str(result.get("negative_id", "")) for result in negative_results}
    negative_confidence = float(validation_cfg.get("min_negative_confidence", 0.85))
    negatives_ok = (
        actual_negative_ids == expected_negative_ids
        and all(
            not bool(result.get("answerable_from_negative", True))
            and float(result.get("confidence", 0.0)) >= negative_confidence
            for result in negative_results
        )
    )

    gates = {
        "oracle_answerable": bool(oracle.get("answerable", False)),
        "oracle_confident": float(oracle.get("confidence", 0.0)) >= float(validation_cfg.get("min_oracle_confidence", 0.90)),
        "internet_evidence_supported": evidence_ok,
        "internet_not_contradicted": contradiction_score < 0.70,
        "audit_passed": audit.get("verdict") == "pass",
        "audit_confident": float(audit.get("confidence", 0.0)) >= float(validation_cfg.get("min_audit_confidence", 0.90)),
        "reference_supported": bool(audit.get("reference_answer_supported", False)),
        "assistant_supported": bool(audit.get("assistant_answer_supported", False)),
        "no_unsupported_extras": not bool(audit.get("assistant_has_unsupported_extras", True)),
        "positive_supported": bool(audit.get("positive_context_supports_answer", False)),
        "negatives_valid": negatives_ok,
    }
    gates["passed"] = all(gates.values())
    return gates

def make_no_evidence_result(bundle: dict) -> dict:
    gates = {
        "oracle_answerable": False,
        "oracle_confident": False,
        "internet_evidence_supported": False,
        "internet_not_contradicted": True,
        "audit_passed": False,
        "audit_confident": False,
        "reference_supported": False,
        "assistant_supported": False,
        "no_unsupported_extras": False,
        "positive_supported": False,
        "negatives_valid": False,
        "passed": False,
    }

    return {
        "record_id": bundle["record_id"],
        "retrieval_record_id": bundle["retrieval_record_id"],
        "question": bundle["question"],
        "evidence": [],
        "oracle": {
            "answerable": False,
            "answer": "",
            "confidence": 1.0,
            "evidence_judgements": [],
            "reason": "No independent internet evidence was retrieved.",
        },
        "audit": {
            "reference_answer_supported": False,
            "assistant_answer_supported": False,
            "assistant_has_unsupported_extras": False,
            "positive_context_supports_answer": False,
            "negative_results": [],
            "confidence": 1.0,
            "verdict": "not_found",
            "reason": "Dataset audit skipped because no independent internet evidence was retrieved.",
        },
        "external_verified": False,
        "validation_gates": gates,
        "human_verified": False,
        "validation_method": "searxng_ollama_blind_v1",
    }

def validate_bundle(cfg: dict, *, bundle: dict, policies: list, executor: ThreadPoolExecutor) -> dict:
    validation_cfg = cfg["internet_validation"]
    evidence = collect_parallel_evidence(executor=executor, question=bundle["question"], policies=policies, validation_cfg=validation_cfg)
    if not evidence:
        log.info("[DATASET_VALIDATION] No internet evidence ID=%s; skipping oracle/audit", bundle["record_id"])
        return make_no_evidence_result(bundle)
    oracle_result = run_blind_oracle(cfg, question=bundle["question"], evidence=evidence)
    # if not bool(oracle_result.get("answerable", False)):
    #     return make_unanswerable_result(bundle, evidence, oracle_result)
    audit_result = run_dataset_audit(cfg, oracle_result=oracle_result, bundle=bundle,)
    validation_gates = evaluate_validation(validation_cfg=validation_cfg, bundle=bundle, evidence=evidence, oracle=oracle_result, audit=audit_result,)

    return {
        "record_id": bundle["record_id"],
        "retrieval_record_id": bundle["retrieval_record_id"],
        "question": bundle["question"],
        "evidence": evidence,
        "oracle": oracle_result,
        "audit": audit_result,
        "external_verified": validation_gates["passed"],
        "validation_gates": validation_gates,
        "human_verified": False,
        "validation_method": "searxng_ollama_blind_v1",
    }

def load_completed_ids(path: Path) -> set[str]:
    completed: set[str] = set()

    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_id = str(record.get("record_id", "")).strip()
            if record_id:
                completed.add(record_id)

    return completed

def main() -> None:
    config_path = Path(__file__).resolve().parents[2] / "rag" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    setup_logging(cfg.get("logging", {}).get("file"), cfg.get("logging", {}).get("level", "INFO"))
    validation_cfg = cfg["internet_validation"]
    searxng_url = str(validation_cfg["searxng_url"]).strip()
    require_searxng(searxng_url, timeout_s=float(validation_cfg.get("searxng_health_timeout_s", 5.0,)))
    policies = load_source_policies(cfg)
    source_workers = min(int(validation_cfg.get("source_workers", 5)), len(policies))
    sft_path = Path(validation_cfg["sft_dataset_path"])
    retrieval_path = Path(validation_cfg["retrieval_dataset_path"])
    output_path = Path(validation_cfg.get("validation_output_path", "fine_tune/internet_validation/validation_results.jsonl"))
    completed_ids = load_completed_ids(output_path)
    log.info("[VALIDATION] completed records=%d", len(completed_ids))

    bundles = iter_dataset_bundles(sft_path=sft_path, retrieval_path=retrieval_path)
    limit = max(0, int(validation_cfg.get("limit", 0)))
    with ThreadPoolExecutor(max_workers=source_workers, thread_name_prefix="validation-source") as executor:
        for index, bundle in enumerate(bundles, start=1):
            if limit and index > limit:
                break
            if bundle["record_id"] in completed_ids:
                continue
            try:
                result = validate_bundle(cfg, bundle=bundle, policies=policies, executor=executor)
            except Exception:
                log.exception("Validation failed ID=%s", bundle["record_id"])
                continue

            append_jsonl(output_path, result)
            log.info("[DATASET_VALIDATION] Validated #%d ID=%s verdict=%s confidence=%.3f", index, bundle["record_id"], result["audit"]["verdict"], float(result["audit"]["confidence"]))


if __name__ == "__main__":
    main()