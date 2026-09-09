from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

def run_stage(name: str, args: list[str]) -> None:
    print()
    print("=" * 80)
    print(f"[VALIDATION PIPELINE] {name}")
    print("=" * 80)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)

def add_unavailable(args: list[str], unavailable_sources: list[str]) -> list[str]:
    for source in unavailable_sources:
        args.extend(["--unavailable-source", source])
    return args

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rag/config.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--mode", choices=["audit", "calibrate", "full"], default=None)
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    vcfg = cfg.get("dataset_validation", {}) or {}
    data_dir = Path(args.data_dir or vcfg.get("data_dir", "fine_tune/data"))
    mode = str(args.mode or vcfg.get("mode", "calibrate")).lower()
    unavailable_sources = [str(x) for x in vcfg.get("unavailable_sources", [])]

    audit_dir = data_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    manifest = audit_dir / "validation_manifest.jsonl"
    duplicate_out = audit_dir / "duplicate_validation.jsonl"
    semantic_out = audit_dir / "semantic_validation.jsonl"

    audit_cmd = [
        "fine_tune/audit_dataset.py",
        "--config", str(config_path),
        "--data-dir", str(data_dir),
    ]
    run_stage("Structural audit", add_unavailable(audit_cmd, unavailable_sources))

    manifest_cmd = [
        "fine_tune/build_validation_manifest.py",
        "--data-dir", str(data_dir),
        "--out", str(manifest),
    ]
    run_stage("Build validation manifest", add_unavailable(manifest_cmd, unavailable_sources))

    if mode == "audit":
        print("\n[VALIDATION PIPELINE] Audit-only mode complete.")
        return

    duplicate_cfg = vcfg.get("duplicate", {}) or {}
    if bool(duplicate_cfg.get("enabled", True)):
        duplicate_cmd = [
            "fine_tune/validate_duplicate.py",
            "--manifest", str(manifest),
            "--out", str(duplicate_out),
            "--seed", str(int(duplicate_cfg.get("seed", 42))),
        ]

        if mode == "calibrate":
            limit = int(duplicate_cfg.get("calibrate_limit", 50))
            if limit > 0:
                duplicate_cmd.extend(["--limit", str(limit)])

        run_stage("Duplicate semantic validation", duplicate_cmd)

    semantic_cfg = vcfg.get("semantic", {}) or {}

    if bool(semantic_cfg.get("enabled", True)):
        seeds = semantic_cfg.get("seeds", {}) or {
            "high": 42,
            "medium": 43,
            "low": 44,
        }

        if mode == "calibrate":
            limits = semantic_cfg.get("calibrate_limits", {}) or {
                "high": 200,
                "medium": 200,
                "low": 500,
            }
        else:
            limits = semantic_cfg.get("full_limits", {}) or {
                "high": 0,
                "medium": 0,
                "low": 0,
            }

        for risk in ("high", "medium", "low"):
            semantic_cmd = [
                "fine_tune/semantic_validation.py",
                "--data-dir", str(data_dir),
                "--risk", risk,
                "--shuffle",
                "--seed", str(int(seeds.get(risk, 42))),
                "--out", str(semantic_out),
            ]

            limit = int(limits.get(risk, 0))
            if limit > 0:
                semantic_cmd.extend(["--limit", str(limit)])

            run_stage(f"Semantic validation [{risk}]", semantic_cmd)

    print()
    print("=" * 80)
    print("[VALIDATION PIPELINE] AUTOMATED VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Audit:      {audit_dir / 'dataset_audit_report.json'}")
    print(f"Manifest:   {manifest}")
    print(f"Duplicates: {duplicate_out}")
    print(f"Semantic:   {semantic_out}")

if __name__ == "__main__":
    main()