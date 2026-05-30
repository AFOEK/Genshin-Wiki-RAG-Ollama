from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig, VeraConfig, IA3Config, AdaLoraConfig
from trl import SFTConfig, SFTTrainer

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


def get_target_modules(train_cfg: dict, method_cfg: dict) -> list[str]:
    profile = method_cfg.get("target_profile", "llama_mlp_attention")
    profiles = train_cfg.get("target_modules", {}) or {}
    return list(profiles.get(profile, profile if isinstance(profile, list) else []))

def resolve_path(path_value: str | Path, cfg: dict) -> Path:
    p = expand_path(path_value)

    if p.is_absolute():
        return p

    storage = cfg.get("storage", {}) or {}
    primary = storage.get("primary_root")

    if primary:
        return (expand_path(primary) / p).resolve()

    return (REPO_ROOT / p).resolve()


def as_bool(x, default: bool = False) -> bool:
    if x is None:
        return default
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")


def get_torch_dtype(name: str):
    name = str(name or "").lower()

    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32

    return None


def build_peft_config(train_cfg: dict):
    mode = str(train_cfg.get("mode", "lora")).strip().lower()

    if mode in ("lora", "qlora"):
        c = train_cfg.get("lora", {}) or {}
        return LoraConfig(
            r=int(c.get("r", 16)),
            lora_alpha=int(c.get("alpha", 32)),
            lora_dropout=float(c.get("dropout", 0.05)),
            bias=str(c.get("bias", "none")),
            target_modules=get_target_modules(train_cfg, c),
            task_type="CAUSAL_LM",
            use_dora=False,
        )

    if mode in ("dora", "qdora"):
        c = train_cfg.get("dora", {}) or {}
        return LoraConfig(
            r=int(c.get("r", 16)),
            lora_alpha=int(c.get("alpha", 32)),
            lora_dropout=float(c.get("dropout", 0.05)),
            bias=str(c.get("bias", "none")),
            target_modules=get_target_modules(train_cfg, c),
            task_type="CAUSAL_LM",
            use_dora=True,
        )

    if mode == "vera":
        c = train_cfg.get("vera", {}) or {}
        return VeraConfig(
            r=int(c.get("r", 256)),
            vera_dropout=float(c.get("dropout", 0.05)),
            projection_prng_key=int(c.get("projection_prng_key", 0)),
            save_projection=str(c.get("save_projection", True)).lower() in ("1", "true", "yes", "y", "on"),
            target_modules=get_target_modules(train_cfg, c),
            task_type="CAUSAL_LM",
        )

    if mode == "ia3":
        c = train_cfg.get("ia3", {}) or {}
        return IA3Config(
            target_modules=get_target_modules(train_cfg, c),
            feedforward_modules=list(c.get("feedforward_modules", ["gate_proj", "up_proj", "down_proj"])),
            task_type="CAUSAL_LM",
        )

    if mode == "adalora":
        c = train_cfg.get("adalora", {}) or {}
        return AdaLoraConfig(
            init_r=int(c.get("init_r", 12)),
            target_r=int(c.get("target_r", 8)),
            beta1=float(c.get("beta1", 0.85)),
            beta2=float(c.get("beta2", 0.85)),
            tinit=int(c.get("tinit", 200)),
            tfinal=int(c.get("tfinal", 1000)),
            deltaT=int(c.get("delta_t", 10)),
            lora_alpha=int(c.get("alpha", 32)),
            lora_dropout=float(c.get("dropout", 0.05)),
            target_modules=get_target_modules(train_cfg, c),
            task_type="CAUSAL_LM",
        )

    if mode == "dvora":
        raise NotImplementedError(
            "DVoRA is not implemented in this train_peft.py yet. "
            "DVoRA means DoRA magnitude decomposition with VeRA-style directional update, "
            "so it needs custom adapter code."
        )

    raise RuntimeError(f"Unknown peft_training.mode: {mode}")


def quantization_enabled(train_cfg: dict) -> bool:
    mode = str(train_cfg.get("mode", "lora")).strip().lower()
    qcfg = train_cfg.get("quantization", {}) or {}

    return (
        mode in ("qlora", "qdora")
        or str(qcfg.get("enabled", False)).lower() in ("1", "true", "yes", "y", "on")
    )

def maybe_quantization_config(train_cfg: dict):
    qcfg = train_cfg.get("quantization", {}) or {}
    mode = str(train_cfg.get("mode", "lora")).strip().lower()

    enabled = as_bool(qcfg.get("enabled"), False) or mode in ("qlora", "qdora")

    if not enabled:
        return None
    
    compute_dtype = get_torch_dtype(qcfg.get("bnb_4bit_compute_dtype", "bfloat16"))

    return BitsAndBytesConfig(
        load_in_4bit=as_bool(qcfg.get("load_in_4bit"), True),
        bnb_4bit_quant_type=str(qcfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=compute_dtype or torch.bfloat16,
        bnb_4bit_use_double_quant=as_bool(qcfg.get("bnb_4bit_use_double_quant"), True),
    )


def main() -> None:
    setup_logging(
        cfg.get("logging", {}).get("file"),
        cfg.get("logging", {}).get("level", "INFO")
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rag/config.yaml")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--train-file", default=None)
    ap.add_argument("--val-file", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    train_cfg = cfg.get("peft_training", {}) or {}

    if not as_bool(train_cfg.get("enabled"), True):
        log.info("[PEFT] peft_training.enabled=false; exiting")
        return

    if args.mode:
        train_cfg["mode"] = args.mode

    mode = str(train_cfg.get("mode", "lora")).strip().lower()

    base_model = args.base_model or train_cfg.get("base_model")
    if not base_model:
        raise RuntimeError("peft_training.base_model is required")

    train_file = resolve_path(
        args.train_file or train_cfg.get("train_file", "data/training/genshin_lora_train.jsonl"), cfg)

    val_file = resolve_path(
        args.val_file or train_cfg.get("val_file", "data/training/genshin_lora_val.jsonl"), cfg)

    output_dir = resolve_path(
        args.output_dir or train_cfg.get("output_dir", f"data/training/adapters/genshin_{mode}"), cfg)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")

    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    log.info("[PEFT] mode=%s", mode)
    log.info("[PEFT] base_model=%s", base_model)
    log.info("[PEFT] train_file=%s", train_file)
    log.info("[PEFT] val_file=%s", val_file)
    log.info("[PEFT] output_dir=%s", output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = maybe_quantization_config(train_cfg)

    torch_dtype = None
    tcfg = train_cfg.get("training", {}) or {}

    if as_bool(tcfg.get("bf16"), False):
        torch_dtype = torch.bfloat16
    elif as_bool(tcfg.get("fp16"), False):
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype, quantization_config=quant_config, device_map="auto")

    model.config.use_cache = False

    peft_config = build_peft_config(train_cfg)

    data_files = {"train": str(train_file), "validation": str(val_file)}

    dataset = load_dataset("json", data_files=data_files)

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(tcfg.get("epochs", 2)),
        learning_rate=float(tcfg.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(tcfg.get("batch_size", 1)),
        per_device_eval_batch_size=int(tcfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 8)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.03)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        save_steps=int(tcfg.get("save_steps", 250)),
        eval_steps=int(tcfg.get("eval_steps", 250)),
        save_total_limit=int(tcfg.get("save_total_limit", 2)),
        bf16=as_bool(tcfg.get("bf16"), False),
        fp16=as_bool(tcfg.get("fp16"), False),
        gradient_checkpointing=as_bool(tcfg.get("gradient_checkpointing"), True),
        max_length=int(train_cfg.get("max_seq_length", 2048)),
        packing=as_bool(train_cfg.get("packing"), False),
        eval_strategy="steps",
        save_strategy="steps",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    log.info("[PEFT] training complete: %s", output_dir)


if __name__ == "__main__":
    main()