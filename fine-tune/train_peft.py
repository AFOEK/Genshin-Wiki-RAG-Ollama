from __future__ import annotations

import argparse
import logging
import os
import sys
import hashlib
import math
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig, VeraConfig, IA3Config, AdaLoraConfig
from trl import SFTConfig, SFTTrainer

import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))

from utils.logging_setup import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)

class DVoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, *, r: int, alpha: float, eps: float = 1e-6, seed: int = 1337, init_b: float = 0.0, init_d: float = 1.0):
        super().__init__()
        
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"DVoRALinear only supports nn.Linear, got {type(linear)}")
        
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.eps = float(eps)

        weight = linear.weight.detach().clone()
        self.register_buffer("base_weight", weight)

        if linear.bias is not None:
            self.register_buffer("base_bias", linear.bias.detach().clone())
        else:
            self.base_bias = None

        magnitude = torch.linalg.vector_norm(weight.float(), dim=1)
        self.magnitude = nn.Parameter(magnitude.to(dtype=weight.dtype))

        shape_key = f"{seed}:{self.out_features}:{self.in_features}:{self.r}"
        shape_hash = int(hashlib.sha256(shape_key.encode("utf-8")).hexdigest()[:8], 16)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(shape_hash)

        A = torch.randn(self.r, self.in_features, generator=gen, dtype=torch.float32)
        B = torch.randn(self.out_features, self.r, generator=gen, dtype=torch.float32)

        A = A / math.sqrt(max(1, self.in_features))
        B = B / math.sqrt(max(1, self.r))

        self.register_buffer("vera_A", A.to(dtype=weight.dtype))
        self.register_buffer("vera_B", B.to(dtype=weight.dtype))

        self.vera_d = nn.Parameter(torch.full((self.r,), float(init_d), dtype=weight.dtype))
        self.vera_b = nn.Parameter(torch.full((self.out_features,), float(init_b), dtype=weight.dtype))

    def delta_weight(self) -> torch.Tensor:
        B_scaled = self.vera_B * self.vera_b[:, None]
        A_scaled = self.vera_A * self.vera_d[:, None]

        return self.scaling * (B_scaled @ A_scaled)

    def effective_weight(self) -> torch.Tensor:
        direction = self.base_weight + self.delta_weight()
        direction_norm = torch.linalg.vector_norm(
            direction.float(),
            dim=1,
            keepdim=True,
        ).clamp_min(self.eps).to(dtype=direction.dtype)

        return self.magnitude[:, None] * direction / direction_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.base_bias)

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

def get_parent_module(model: nn.Module, module_name: str):
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    return parent, child_name

def freeze_all_parameters(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False

def format_template_path(value: str | Path, *, method: str) -> str:
    return str(value).format(method=method, method_name=method)

def resolve_template_path(value: str | Path, cfg: dict, *, method: str) -> Path:
    return resolve_path(format_template_path(value, method=method), cfg)

def apply_dvora_adapters(model: nn.Module, train_cfg: dict) -> nn.Module:
    c = train_cfg.get("dvora", {}) or {}

    r = int(c.get("r", 256))
    alpha = float(c.get("alpha", 32))
    eps = float(c.get("eps", 1e-6))
    seed = int(c.get("seed", 1337))
    init_b = float(c.get("init_b", 0.0))
    init_d = float(c.get("init_d", 1.0))

    target_modules = set(get_target_modules(train_cfg, c))

    if not target_modules:
        raise RuntimeError("[DVoRA] No target modules configured")

    freeze_all_parameters(model)

    replaced = 0

    for name, module in list(model.named_modules()):
        leaf_name = name.split(".")[-1]

        if leaf_name not in target_modules:
            continue

        if not isinstance(module, nn.Linear):
            continue

        parent, child_name = get_parent_module(model, name)
        setattr(parent, child_name, DVoRALinear(module, r=r, alpha=alpha, eps=eps, seed=seed, init_b=init_b, init_d=init_d))
        replaced += 1

    if replaced == 0:
        raise RuntimeError(f"[DVoRA] Replaced 0 modules. Check target_modules={sorted(target_modules)}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    log.info("[DVoRA] replaced=%d trainable=%d total=%d trainable_pct=%.6f", replaced, trainable, total, 100.0 * trainable / max(1, total))
    return model

def save_dvora_adapter(model: nn.Module, output_dir: Path, train_cfg: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_state = {}
    for name, tensor in model.state_dict().items():
        if any(k in name for k in ["magnitude", "vera_b", "vera_d"]):
            adapter_state[name] = tensor.detach().cpu()

    payload = {"type": "dvora",  "config": train_cfg.get("dvora", {}), "state_dict": adapter_state}

    torch.save(payload, output_dir / "dvora_adapter.pt")
    log.info("[DVoRA] saved adapter state: %s", output_dir / "dvora_adapter.pt")

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
            use_dora=False)

    if mode in ("dora", "qdora"):
        c = train_cfg.get("dora", {}) or {}
        return LoraConfig(
            r=int(c.get("r", 16)),
            lora_alpha=int(c.get("alpha", 32)),
            lora_dropout=float(c.get("dropout", 0.05)),
            bias=str(c.get("bias", "none")),
            target_modules=get_target_modules(train_cfg, c),
            task_type="CAUSAL_LM",
            use_dora=True)

    if mode == "vera":
        c = train_cfg.get("vera", {}) or {}
        return VeraConfig(
            r=int(c.get("r", 256)),
            vera_dropout=float(c.get("dropout", 0.05)),
            projection_prng_key=int(c.get("projection_prng_key", 0)),
            save_projection=str(c.get("save_projection", True)).lower() in ("1", "true", "yes", "y", "on"),
            target_modules=get_target_modules(train_cfg, c),
            task_type="CAUSAL_LM")

    if mode == "ia3":
        c = train_cfg.get("ia3", {}) or {}
        return IA3Config(
            target_modules=get_target_modules(train_cfg, c),
            feedforward_modules=list(c.get("feedforward_modules", ["gate_proj", "up_proj", "down_proj"])),
            task_type="CAUSAL_LM")

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
            task_type="CAUSAL_LM")

    if mode == "dvora":
        return None

    raise RuntimeError(f"Unknown peft_training.mode: {mode}")

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

def resolve_existing_or_fallback(primary_value: str | Path, fallback_value: str | Path | None, cfg: dict, *, method: str, label: str) -> Path:
    primary = resolve_template_path(primary_value, cfg, method=method)
    if primary.exists():
        return primary

    if fallback_value:
        fallback = resolve_template_path(fallback_value, cfg, method=method)
        if fallback.exists():
            log.warning(
                "[PEFT] %s file not found: %s; using fallback: %s",
                label,
                primary,
                fallback,
            )
            return fallback
    raise FileNotFoundError(f"{label} file not found: {primary}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rag/config.yaml")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--train-file", default=None)
    ap.add_argument("--val-file", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
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

    log.info("[PEFT] mode=%s", mode)
    log.info("[PEFT] base_model=%s", base_model)
    log.info("[PEFT] train_file=%s", train_file)
    log.info("[PEFT] val_file=%s", val_file)
    log.info("[PEFT] output_dir=%s", output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if mode == "dvora":
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

    if mode == "dvora":
        model = apply_dvora_adapters(model, train_cfg)
        peft_config = None
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
    if mode == "dvora":
        save_dvora_adapter(model, output_dir, train_cfg)
    log.info("[PEFT] training complete: %s", output_dir)


if __name__ == "__main__":
    main()