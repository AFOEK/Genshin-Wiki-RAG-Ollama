from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))

from utils.logging_setup import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)


def load_cfg(path: str | None) -> dict:
    if not path:
        return {}

    p = Path(path)

    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / path

    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def expand_path(x) -> Path:
    return Path(os.path.expandvars(str(x))).expanduser()


def resolve_output_path(path_value: str | Path, cfg: dict) -> Path:
    p = expand_path(path_value)

    if p.is_absolute():
        return p

    storage = cfg.get("storage", {}) or {}
    primary = storage.get("primary_root")

    if primary:
        return (expand_path(primary) / p).resolve()

    return (REPO_ROOT / p).resolve()


def cfg_bool(x, default: bool = False) -> bool:
    if x is None:
        return default
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")


def cfg_float(x, default: float) -> float:
    if x is None:
        return default
    return float(x)


def cfg_int(x, default: int) -> int:
    if x is None:
        return default
    return int(x)


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {e}") from e

    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rag/config.yaml")
    ap.add_argument("--src", default=None)
    ap.add_argument("--train-out", default=None)
    ap.add_argument("--val-out", default=None)
    ap.add_argument("--val-ratio", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--enabled", action=argparse.BooleanOptionalAction, default=None)

    args = ap.parse_args()

    cfg = load_cfg(args.config)
    split_cfg = cfg.get("dataset_split", {}) or {}

    setup_logging(
        cfg.get("logging", {}).get("file"),
        cfg.get("logging", {}).get("level", "INFO")
    )
    enabled = (args.enabled if args.enabled is not None else cfg_bool(split_cfg.get("enabled"), True))

    if not enabled:
        log.info("[SPLIT] dataset_split.enabled=false; skipping")
        return

    src = resolve_output_path(args.src or split_cfg.get("src", "data/training/genshin_lora_candidates.jsonl"), cfg)
    train_out = resolve_output_path(args.train_out or split_cfg.get("train_out", "data/training/genshin_lora_train.jsonl"), cfg)
    val_out = resolve_output_path(args.val_out or split_cfg.get("val_out", "data/training/genshin_lora_val.jsonl"), cfg)
    val_ratio = cfg_float(args.val_ratio, cfg_float(split_cfg.get("val_ratio"), 0.05))
    seed = cfg_int(args.seed, cfg_int(split_cfg.get("seed"), 1337))

    if not src.exists():
        raise FileNotFoundError(f"Input JSONL not found: {src}")

    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    log.info("[SPLIT] src=%s", src)
    log.info("[SPLIT] train_out=%s", train_out)
    log.info("[SPLIT] val_out=%s", val_out)
    log.info("[SPLIT] val_ratio=%s seed=%s", val_ratio, seed)

    rows = read_jsonl(src)

    if not rows:
        raise RuntimeError(f"Input JSONL is empty: {src}")

    random.Random(seed).shuffle(rows)

    n_val = max(1, int(len(rows) * val_ratio))

    if n_val >= len(rows):
        n_val = max(1, len(rows) - 1)

    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    if not train_rows:
        raise RuntimeError("Train split is empty. Increase dataset size or reduce val_ratio.")

    write_jsonl(train_out, train_rows)
    write_jsonl(val_out, val_rows)

    log.info("[SPLIT] done train=%d val=%d", len(train_rows), len(val_rows))


if __name__ == "__main__":
    main()