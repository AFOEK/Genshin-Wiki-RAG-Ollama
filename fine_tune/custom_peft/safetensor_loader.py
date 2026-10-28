import json
import torch
import sys

from pathlib import Path
from safetensors.torch import load_file, load_model

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from HybridQuantumRetriever import HybridQuantumRetriever
from HQ_LoRA import ModuleQuantumLoRALinear
from HQ_DVoRA import ModuleQuantumDVoRALinear

def expected_adapter_keys(model):
    if not model.uses_module_peft:
        return {
            f"adapter.{key}"
            for key in model.adapter.state_dict()
        }

    expected = set()

    for prefix, module in model.named_modules():
        if not isinstance(
            module,
            (ModuleQuantumLoRALinear, ModuleQuantumDVoRALinear),
        ):
            continue

        expected.update(
            f"{prefix}.{key}"
            for key in module.state_dict()
            if not key.startswith("base_layer.")
        )

    return expected


def load_hq_retriever(
    export_dir,
    *,
    encoder_device="cpu",
    weights_kind="auto",
):
    folder = Path(export_dir).expanduser().resolve()
    meta_path = folder / "quantum_config.json"

    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)

    meta = json.loads(
        meta_path.read_text(encoding="utf-8")
    )

    if meta.get("architecture") != "HybridQuantumRetriever":
        raise ValueError("Invalid HQ retriever architecture")

    mode = meta["adapter_mode"]

    if (
        mode.endswith("dvora")
        and meta.get("reconstructed_frozen_parameters", False)
    ):
        raise RuntimeError(
            "DVoRA frozen projection weights were not "
            "verified during export. Refusing to load."
        )

    # Choose the weight file
    adapter_path = folder / "adapter.safetensors"
    full_path = folder / "model.safetensors"

    if weights_kind not in {"auto", "adapter", "full"}:
        raise ValueError("Invalid weights_kind")

    if weights_kind == "adapter" or (
        weights_kind == "auto" and adapter_path.exists()
    ):
        weights_path = adapter_path
        selected_kind = "adapter"
    else:
        weights_path = full_path
        selected_kind = "full"

    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    device = torch.device(encoder_device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Reconstruct the same model architecture
    model = HybridQuantumRetriever(
        model_name=meta["base_model"],
        adapter_mode=mode,
        rank=int(meta["rank"]),
        quantum_layers=int(meta["quantum_layers"]),
        alpha=float(meta["alpha"]),
        dropout=float(meta["dropout"]),
        quantum_dropout=float(meta["quantum_dropout"]),
        shots=meta.get("shots"),
        diff_method=meta["diff_method"],
        module_target_preset=(
            meta.get("module_target_preset") or "qv"
        ),
        module_token_policy=(
            meta.get("module_token_policy") or "cls"
        ),
        module_layer_policy=(
            meta.get("module_layer_policy") or "all"
        ),
        module_last_k=int(
            meta.get("module_last_k") or 2
        ),
        module_selected_layers=meta.get(
            "module_selected_layers"
        ),
        module_adapter_device=torch.device("cpu"),
        encoder_device=device,
        use_ibm_fake_noise=bool(
            meta.get("use_ibm_fake_noise", False)
        ),
        ibm_fake_backend_name=(
            meta.get("ibm_fake_backend_name")
            or "FakeSherbrooke"
        ),
        simulator_seed=meta.get("simulator_seed"),
    )

    # Verify the encoder profile matches training.
    for key in (
        "pooling_mode",
        "query_prefix",
        "document_prefix",
    ):
        if key in meta and getattr(model, key) != meta[key]:
            raise RuntimeError(
                f"Model profile mismatch for {key}"
            )

    if "max_length" in meta:
        model.model_max_length = int(meta["max_length"])

    if "embedding_dimension" in meta:
        expected_dim = int(meta["embedding_dimension"])
        actual_dim = int(model.encoder.config.hidden_size)

        if expected_dim != actual_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: "
                f"{expected_dim} != {actual_dim}"
            )

    # Restore tokenizer saved alongside the weights.
    if (folder / "tokenizer_config.json").is_file():
        from transformers import AutoTokenizer

        model.tokenizer = AutoTokenizer.from_pretrained(
            str(folder),
            local_files_only=True,
        )

    # Restore weights
    if selected_kind == "adapter":
        state = load_file(str(weights_path), device="cpu")
        expected = expected_adapter_keys(model)

        if not expected or set(state) != expected:
            raise RuntimeError(
                "Adapter checkpoint keys do not match.\n"
                f"Missing: {sorted(expected - set(state))[:10]}\n"
                f"Extra: {sorted(set(state) - expected)[:10]}"
            )

        result = model.load_state_dict(
            state,
            strict=False,
        )

        if result.unexpected_keys or (
            set(result.missing_keys) & expected
        ):
            raise RuntimeError(
                f"Adapter loading failed: {result}"
            )

    else:
        missing, unexpected = load_model(
            model,
            str(weights_path),
            strict=True,
        )

        if missing or unexpected:
            raise RuntimeError(
                f"Full model loading failed: "
                f"{missing}, {unexpected}"
            )

    model.eval()

    print("HQ retriever loaded")
    print("Mode:", mode)
    print("Weights:", selected_kind)
    print("Base:", meta["base_model"])
    print("Device:", device)
    print("Shots:", meta.get("shots"))
    print("IBM noise:", meta.get("use_ibm_fake_noise"))
    print("Embedding dimensions:", model.encoder.config.hidden_size)

    return model
