import os
import yaml
import torch
import logging
import torch.nn as nn

from pathlib import Path
from transformers import BitsAndBytesConfig
from peft import LoraConfig, VeraConfig, IA3Config, AdaLoraConfig, LoHaConfig, LoKrConfig, OFTConfig, BOFTConfig, RandLoraConfig, PveraConfig, XLoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
import bitsandbytes as bnb

log = logging.getLogger(__name__)

def load_cfg(root: str| None ,path: str | None) -> dict:
    if not path:
        return {}

    p = Path(path)

    if not p.is_absolute() and not p.exists():
        p = root / path

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

def resolve_template_path(root: str| Path, value: str | Path, cfg: dict, *, method: str) -> Path:
    return resolve_path(root, format_template_path(value, method=method), cfg)

def expand_path(x) -> Path:
    return Path(os.path.expandvars(str(x))).expanduser()

def get_target_modules(train_cfg: dict, method_cfg: dict) -> list[str]:
    profile = method_cfg.get("target_profile", "llama_mlp_attention")
    profiles = train_cfg.get("target_modules", {}) or {}
    return list(profiles.get(profile, profile if isinstance(profile, list) else []))

def resolve_path(root: str|Path, path_value: str | Path, cfg: dict) -> Path:
    p = expand_path(path_value)

    if p.is_absolute():
        return p

    storage = cfg.get("storage", {}) or {}
    primary = storage.get("primary_root")

    if primary:
        return (expand_path(primary) / p).resolve()

    return (root / p).resolve()

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

def get_optimizer_cls(name: str):
    name = str(name or "adamw").strip().lower()

    if name in ("adamw", "torch_adamw"):
        return torch.optim.AdamW

    if name in ("adam8bit", "bnb_adam8bit", "bitsandbytes_adam8bit"):
        if bnb is None:
            raise RuntimeError(
                "bitsandbytes is required for adam8bit. Install with: pip install bitsandbytes"
            )
        return bnb.optim.Adam8bit

    raise RuntimeError(f"Unknown LoRA+ optimizer: {name}")

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
    
    if mode == "pvera":
        c = train_cfg.get("pvera", {}) or {}
        return PveraConfig(
            r=int(c.get("r", 256)),
            pvera_dropout=float(c.get("dropout", 0.05)),
            projection_prng_key=int(c.get("projection_prng_key", 0)),
            save_projection=as_bool(c.get("save_projection"), True),
            sample_at_inference=as_bool(c.get("sample_at_inference"), False),
            target_modules=get_target_modules(train_cfg, c),
            task_type="CAUSAL_LM",
        )
    
    if mode == "loha":
        c = train_cfg.get("loha", {}) or {}
        return LoHaConfig(
            r=int(c.get("r", 16)),
            alpha=int(c.get("alpha", 32)),
            rank_dropout=float(c.get("rank_dropout", 0.0)),
            module_dropout=float(c.get("module_dropout", 0.0)),
            target_modules=get_target_modules(train_cfg, c),
            init_weights=as_bool(c.get("init_weights"), True),
            task_type="CAUSAL_LM",
        )

    if mode == "lokr":
        c = train_cfg.get("lokr", {}) or {}
        return LoKrConfig(
            r=int(c.get("r", 16)),
            alpha=int(c.get("alpha", 32)),
            rank_dropout=float(c.get("rank_dropout", 0.0)),
            module_dropout=float(c.get("module_dropout", 0.0)),
            decompose_both=as_bool(c.get("decompose_both"), False),
            decompose_factor=int(c.get("decompose_factor", -1)),
            rank_dropout_scale=as_bool(c.get("rank_dropout_scale"), False),
            target_modules=get_target_modules(train_cfg, c),
            init_weights=as_bool(c.get("init_weights"), True),
            task_type="CAUSAL_LM",
        )

    if mode == "randlora":
        c = train_cfg.get("randlora", {}) or {}
        return RandLoraConfig(
            r=int(c.get("r", 32)),
            randlora_alpha=int(c.get("alpha", c.get("randlora_alpha", 640))),
            randlora_dropout=float(c.get("dropout", c.get("randlora_dropout", 0.0))),
            projection_prng_key=int(c.get("projection_prng_key", 0)),
            save_projection=as_bool(c.get("save_projection"), True),
            sparse=as_bool(c.get("sparse"), False),
            very_sparse=as_bool(c.get("very_sparse"), False),
            bias=str(c.get("bias", "none")),
            target_modules=get_target_modules(train_cfg, c),
            task_type="CAUSAL_LM",
        )

    if mode == "oft":
        c = train_cfg.get("oft", {}) or {}
        return OFTConfig(
            r=int(c.get("r", 0)),
            oft_block_size=int(c.get("oft_block_size", 32)),
            module_dropout=float(c.get("module_dropout", 0.0)),
            bias=str(c.get("bias", "none")),
            coft=as_bool(c.get("coft"), False),
            eps=float(c.get("eps", 6e-5)),
            block_share=as_bool(c.get("block_share"), False),
            target_modules=get_target_modules(train_cfg, c),
            task_type="CAUSAL_LM",
        )

    if mode == "boft":
        c = train_cfg.get("boft", {}) or {}
        return BOFTConfig(
            boft_block_size=int(c.get("boft_block_size", 4)),
            boft_block_num=int(c.get("boft_block_num", 0)),
            boft_n_butterfly_factor=int(c.get("boft_n_butterfly_factor", 1)),
            boft_dropout=float(c.get("boft_dropout", 0.0)),
            bias=str(c.get("bias", "none")),
            target_modules=get_target_modules(train_cfg, c),
            task_type="CAUSAL_LM",
        )

    if mode in ("loraplus", "lora_plus"):
        c = train_cfg.get("loraplus", {}) or train_cfg.get("lora", {}) or {}
        return LoraConfig(
            r=int(c.get("r", 16)),
            lora_alpha=int(c.get("alpha", 32)),
            lora_dropout=float(c.get("dropout", 0.05)),
            bias=str(c.get("bias", "none")),
            target_modules=get_target_modules(train_cfg, c),
            task_type="CAUSAL_LM",
            use_dora=False,
        )

    if mode in ("dvora", "bora"):
        return None

    raise RuntimeError(f"Unknown peft_training.mode: {mode}")

def build_xlora_config(train_cfg: dict, cfg: dict, model_config) -> XLoraConfig:
    c = train_cfg.get("xlora", {}) or {}

    adapters_raw = c.get("adapters", {}) or {}
    if not adapters_raw:
        raise RuntimeError("[X-LoRA] peft.xlora.adapters is empty")

    adapters = {}
    for name, path in adapters_raw.items():
        adapter_path = resolve_template_path(path, cfg, method="lora")
        if not adapter_path.exists():
            raise FileNotFoundError(f"[X-LoRA] adapter not found: {name} -> {adapter_path}")
        adapters[str(name)] = str(adapter_path)

    hidden_size = getattr(model_config, "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("[X-LoRA] Could not infer model.config.hidden_size")

    top_k_lora = c.get("top_k_lora", None)
    if str(top_k_lora).lower() in ("none", "null", ""):
        top_k_lora = None

    return XLoraConfig(
        task_type="CAUSAL_LM",
        hidden_size=int(hidden_size),
        adapters=adapters,
        enable_softmax=as_bool(c.get("enable_softmax"), True),
        enable_softmax_topk=as_bool(c.get("enable_softmax_topk"), False),
        layerwise_scalings=as_bool(c.get("layerwise_scalings"), False),
        xlora_depth=int(c.get("xlora_depth", 1)),
        xlora_size=int(c.get("xlora_size", 2048)),
        xlora_dropout_p=float(c.get("xlora_dropout_p", 0.2)),
        use_trainable_adapters=as_bool(c.get("use_trainable_adapters"), False),
        softmax_temperature=float(c.get("softmax_temperature", 1.0)),
        top_k_lora=top_k_lora,
        global_scaling_weight=float(c.get("global_scaling_weight", 1.0)),
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