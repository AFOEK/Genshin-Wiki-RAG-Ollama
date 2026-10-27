import math
import re
import torch
from torch import nn
import torch.nn.functional as F

from configs import  MODEL_PROFILES
from HQ_LoRA import ModuleQuantumLoRALinear
from HQ_DVoRA import ModuleQuantumDVoRALinear


def get_model_profile(model_name: str) -> dict:
    if model_name not in MODEL_PROFILES:
        raise ValueError(f"Missing model profile for {model_name}")
    return MODEL_PROFILES[model_name]

def get_parent_module(root, module_name):
    parts = module_name.split(".")
    parent = root

    for part in parts[:-1]:
        parent = getattr(parent, part)

    return parent, parts[-1]


def get_layer_index(module_name):
    match = re.search(r"encoder\.layer\.(\d+)\.", module_name)
    return int(match.group(1)) if match else None


def get_num_encoder_layers(encoder):
    if hasattr(encoder.config, "num_hidden_layers"):
        return int(encoder.config.num_hidden_layers)

    indices = [
        i for name, _ in encoder.named_modules()
        if (i := get_layer_index(name)) is not None
    ]
    if not indices:
        raise RuntimeError("Could not infer number of encoder layers")

    return max(indices) + 1


def resolve_module_layer_indices(num_layers, layer_policy, last_k=2, selected_layers=None):
    if layer_policy == "all":
        return list(range(num_layers))

    if layer_policy == "last":
        return [num_layers - 1]

    if layer_policy == "last_k":
        if not 1 <= last_k <= num_layers:
            raise ValueError(f"last_k must be in [1, {num_layers}]")
        return list(range(num_layers - last_k, num_layers))

    if layer_policy == "selected":
        if not selected_layers:
            raise ValueError("selected_layers is required")

        indices = sorted(set(selected_layers))
        if any(i < 0 or i >= num_layers for i in indices):
            raise ValueError("Selected layer index out of range")
        return indices

    raise ValueError(f"Unknown layer policy: {layer_policy}")


def should_patch_module(
    module_name, target_suffixes, *,
    num_layers, layer_policy="all",
    last_k=2, selected_layers=None
):
    if not any(
        module_name.endswith(s)
        for s in target_suffixes
    ):
        return False

    index = get_layer_index(module_name)
    if index is None:
        return False

    if layer_policy == "all":
        return True
    if layer_policy == "last":
        return index == num_layers - 1
    if layer_policy == "last_k":
        return index >= num_layers - last_k
    if layer_policy == "selected":
        if selected_layers is None:
            raise ValueError("selected_layers is required")
        return index in selected_layers

    raise ValueError(f"Unknown layer policy: {layer_policy}")


def inject_quantum_peft_modules(
    encoder,
    adapter_mode,
    target_suffixes,
    rank=4,
    quantum_layers=1,
    alpha=8.0,
    dropout=0.05,
    quantum_dropout=0.10,
    shots=None,
    diff_method="parameter-shift",
    token_policy="cls",
    adapter_device=torch.device("cpu"),
    use_ibm_fake_noise=False,
    ibm_fake_backend_name="FakeSherbrooke",
    simulator_seed=None,
    layer_policy="all",
    last_k=2,
    selected_layers=None,
):
    if adapter_mode not in {
        "module_hq_lora", "module_hq_dvora"
    }:
        raise ValueError(f"Unsupported mode: {adapter_mode}")

    patched = []
    num_layers = get_num_encoder_layers(encoder)

    for name, module in list(encoder.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        if not should_patch_module(
            name,
            target_suffixes,
            num_layers=num_layers,
            layer_policy=layer_policy,
            last_k=last_k,
            selected_layers=selected_layers,
        ):
            continue

        parent, child = get_parent_module(encoder, name)

        adapter_cls = (
            ModuleQuantumLoRALinear
            if adapter_mode == "module_hq_lora"
            else ModuleQuantumDVoRALinear
        )

        replacement = adapter_cls(
            base_layer=module,
            rank=rank,
            quantum_layers=quantum_layers,
            alpha=alpha,
            dropout=dropout,
            quantum_dropout=quantum_dropout,
            shots=shots,
            diff_method=diff_method,
            token_policy=token_policy,
            adapter_device=adapter_device,
            use_ibm_fake_noise=use_ibm_fake_noise,
            ibm_fake_backend_name=ibm_fake_backend_name,
            simulator_seed=simulator_seed,
        )

        setattr(parent, child, replacement)
        patched.append(name)

    if not patched:
        raise RuntimeError(
            "No transformer modules patched. "
            "Check target preset and layer policy."
        )

    return patched
