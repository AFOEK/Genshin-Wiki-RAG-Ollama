from __future__ import annotations

import argparse
import logging
import sys
import torch

from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer

from .custom_peft.DVoRA import apply_dvora_adapters, save_dvora_adapter
from .custom_peft.BoRA import apply_bora_adapters, save_bora_adapter
from .utils_peft import as_bool, resolve_template_path, load_cfg, resolve_existing_or_fallback, maybe_quantization_config, build_peft_config, build_xlora_config, get_peft_model, get_optimizer_cls, create_loraplus_optimizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))

from utils.logging_setup import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rag/config.yaml")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--train-file", default=None)
    ap.add_argument("--val-file", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = load_cfg(REPO_ROOT, args.config)
    train_cfg = cfg.get("peft", {}) or {}

    setup_logging(
        cfg.get("logging", {}).get("file"),
        cfg.get("logging", {}).get("level", "INFO")
    )

    if not as_bool(train_cfg.get("enabled"), True):
        log.info("[PEFT] peft_training.enabled=false; exiting")
        return

    if args.mode:
        train_cfg["mode"] = args.mode

    mode = str(train_cfg.get("mode", "lora")).strip().lower()

    base_model = args.base_model or train_cfg.get("base_model")
    if not base_model:
        raise RuntimeError("peft_training.base_model is required")

    train_file = resolve_existing_or_fallback(args.train_file or train_cfg.get("train_file", "data/training/genshin_{method}_train.jsonl"), train_cfg.get("fallback_train_file"), cfg, method=mode, label="Train")
    val_file = resolve_existing_or_fallback(args.val_file or train_cfg.get("val_file", "data/training/genshin_{method}_val.jsonl"), train_cfg.get("fallback_val_file"), cfg, method=mode, label="Validation")
    output_dir = resolve_template_path(args.output_dir or train_cfg.get("output_dir", "data/training/adapters/genshin_{method}"), cfg, method=mode)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")

    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    log.info("[PEFT] mode=%s, base_model=%s, train_file=%s, val_file=%s, output_dir=%s", mode, base_model, train_file, val_file, output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if mode in {"dvora", "bora"}:
        quant_config = None
    else:
        quant_config = maybe_quantization_config(train_cfg)

    torch_dtype = None
    tcfg = train_cfg.get("training", {}) or {}

    if as_bool(tcfg.get("bf16"), False):
        torch_dtype = torch.bfloat16
    elif as_bool(tcfg.get("fp16"), False):
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype, quantization_config=quant_config, device_map="auto")
    model.config.use_cache = False

    custom_optimizer = None

    if mode == "dvora":
        model = apply_dvora_adapters(model, train_cfg)
        peft_config = None
    elif mode == "bora":
        model = apply_bora_adapters(model, train_cfg)
        peft_config = None
    elif mode == "xlora":
        peft_config = build_xlora_config(train_cfg, cfg, model.config)
    elif mode in ("loraplus", "lora_plus"):
        peft_config = build_peft_config(train_cfg)
        model = get_peft_model(model, peft_config)
        peft_config = None
        lp_cfg = train_cfg.get("loraplus", {}) or {}
        optimizer_cls = get_optimizer_cls(lp_cfg.get("optimizer", "adamw"))
        custom_optimizer = create_loraplus_optimizer(model=model, optimizer_cls=optimizer_cls, lr=float(tcfg.get("learning_rate", 2e-4)), loraplus_lr_ratio=float(lp_cfg.get("lr_ratio", 16)))
    else:
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
        gradient_checkpointing=as_bool(tcfg.get("gradient_checkpointing", True), True),
        max_length=int(train_cfg.get("max_seq_length", 2048)),
        packing=as_bool(train_cfg.get("packing"), False),
        eval_strategy="steps",
        save_strategy="steps",
        report_to="none",
    )

    trainer_kwargs = dict(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    if custom_optimizer is not None:
        trainer_kwargs["optimizers"] = (custom_optimizer, None)

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    if mode == "dvora":
        save_dvora_adapter(model, output_dir, train_cfg)
    elif mode == "bora":
        save_bora_adapter(model, output_dir, train_cfg)
    log.info("[PEFT] training complete: %s", output_dir)

if __name__ == "__main__":
    main()