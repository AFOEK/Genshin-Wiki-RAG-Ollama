import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .configs import MODULE_TARGET_PRESETS, QUANTUM_DEVICE, ENCODER_DEVICE

from HQ_LoRA import HybridQuantumLoRAAdapter
from HQ_DVoRA import HybridQuantumDVoRAAdapter
from DVoRA import ClassicalDVoRAAdapter
from LoRA import ClassicalLoRAAdapter

from retriever_support import get_model_profile, get_num_encoder_layers, resolve_module_layer_indices, inject_quantum_peft_modules

class HybridQuantumRetriever(nn.Module):
    def __init__(
        self,
        model_name: str,
        adapter_mode: str = "hq_lora",
        rank: int = 4,
        quantum_layers: int = 1,
        alpha: float = 8.0,
        dropout: float = 0.05,
        quantum_dropout: float = 0.05,
        shots: int | None = None,
        diff_method: str = "parameter-shift",
        module_target_preset: str = "qv",
        module_token_policy: str = "cls",
        module_layer_policy: str = "all",
        module_last_k: int = 2,
        module_selected_layers: list[int] | None = None,
        module_adapter_device: torch.device = torch.device("cpu"),
        encoder_device: torch.device | None = None,
        use_ibm_fake_noise: bool = False,
        ibm_fake_backend_name: str = "FakeSherbrooke",
        simulator_seed: int | None = None,
    ):
        super().__init__()
        supported_modes = {
            "classical_lora",
            "classical_dvora",
            "hq_lora",
            "hq_dvora",
            "module_hq_lora",
            "module_hq_dvora",
        }

        if adapter_mode not in supported_modes:
            raise ValueError(f"Unknown adapter_mode: {adapter_mode}")
        self.adapter_mode = adapter_mode
        self.uses_module_peft = adapter_mode in {"module_hq_lora", "module_hq_dvora"}
        self.model_profile = get_model_profile(model_name)
        self.pooling_mode = self.model_profile["pooling_mode"]
        self.query_prefix = self.model_profile["query_prefix"]
        self.document_prefix = self.model_profile["document_prefix"]
        self.model_max_length = self.model_profile["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder_device = (encoder_device if encoder_device is not None else ENCODER_DEVICE)
        self.module_adapter_device = (module_adapter_device)
        self.encoder = AutoModel.from_pretrained(model_name, **self.model_profile["model_kwargs"],).to(self.encoder_device)
        num_layers = get_num_encoder_layers(self.encoder)
        self.module_trainable_layers = resolve_module_layer_indices(num_layers=num_layers, layer_policy=module_layer_policy, last_k=module_last_k, selected_layers=module_selected_layers,)
        self.module_cache_first_trainable_layer = min(self.module_trainable_layers)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        if self.uses_module_peft:
            target_suffixes = MODULE_TARGET_PRESETS[module_target_preset]
            self.patched_modules = inject_quantum_peft_modules(
                encoder=self.encoder,
                adapter_mode=adapter_mode,
                target_suffixes=target_suffixes,
                rank=rank,
                quantum_layers=quantum_layers,
                alpha=alpha,
                dropout=dropout,
                quantum_dropout=quantum_dropout,
                shots=shots,
                diff_method=diff_method,
                token_policy=module_token_policy,
                adapter_device=module_adapter_device,
                use_ibm_fake_noise=use_ibm_fake_noise,
                ibm_fake_backend_name=ibm_fake_backend_name,
                layer_policy=module_layer_policy,
                last_k=module_last_k,
                selected_layers=module_selected_layers,
                simulator_seed=simulator_seed,
            )
            self.adapter = nn.Identity()
        elif adapter_mode == "classical_lora":
            self.adapter = ClassicalLoRAAdapter(hidden_size=hidden_size, rank=rank, alpha=alpha, dropout=dropout).to(QUANTUM_DEVICE)
        elif adapter_mode == "classical_dvora":
            self.adapter = ClassicalDVoRAAdapter(hidden_size=hidden_size, rank=rank, alpha=alpha, dropout=dropout).to(QUANTUM_DEVICE)
        elif adapter_mode == "hq_lora":
            self.adapter = HybridQuantumLoRAAdapter(
                hidden_size=hidden_size,
                rank=rank,
                quantum_layers=quantum_layers,
                alpha=alpha,
                dropout=dropout,
                quantum_dropout=quantum_dropout,
                shots=shots,
                diff_method=diff_method,
                use_ibm_fake_noise=use_ibm_fake_noise,
                ibm_fake_backend_name=ibm_fake_backend_name,
                simulator_seed=simulator_seed,
            ).to(QUANTUM_DEVICE)
        elif adapter_mode == "hq_dvora":
            self.adapter = HybridQuantumDVoRAAdapter(
                hidden_size=hidden_size,
                rank=rank,
                quantum_layers=quantum_layers,
                alpha=alpha,
                dropout=dropout,
                quantum_dropout=quantum_dropout,
                shots=shots,
                diff_method=diff_method,
                use_ibm_fake_noise=use_ibm_fake_noise,
                ibm_fake_backend_name=ibm_fake_backend_name,
                simulator_seed=simulator_seed,
            ).to(QUANTUM_DEVICE)
        self.adapter = self.adapter.to(QUANTUM_DEVICE)

    @staticmethod
    def pool_embeddings(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling_mode: str) -> torch.Tensor:
        if pooling_mode == "cls":
            return hidden[:, 0, :]
    
        if pooling_mode == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
    
        raise ValueError(f"Unknown pooling_mode: {pooling_mode}")

    def encode_texts(self, texts: list[str], *, max_length: int | None = None, is_query: bool = False) -> torch.Tensor:
        if max_length is None:
            max_length = self.model_max_length
    
        if is_query:
            prefix = self.query_prefix
        else:
            prefix = self.document_prefix

        if prefix:
            texts = [f"{prefix}{text}" for text in texts]
    
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt",)
        tokens = {key: value.to(self.encoder_device) for key, value in tokens.items()}
        if self.uses_module_peft:
            outputs = self.encoder(**tokens)
            pooled = self.pool_embeddings(outputs.last_hidden_state, tokens["attention_mask"], pooling_mode=self.pooling_mode,)
            return F.normalize(pooled, p=2, dim=-1)
    
        with torch.no_grad():
            outputs = self.encoder(**tokens)
            pooled = self.pool_embeddings(outputs.last_hidden_state, tokens["attention_mask"], pooling_mode=self.pooling_mode,)
    
        return self.adapt_embeddings(pooled)

    def encode_from_module_cache(self, hidden_states, attention_mask,):
        if not self.uses_module_peft:
            raise RuntimeError("Module cache is only valid for module PEFT.")
    
        hidden_states = hidden_states.float().to(self.encoder_device)
        attention_mask = attention_mask.to(self.encoder_device)
        if hidden_states.dim() != 3:
            raise RuntimeError(f"Expected cached hidden states [B, S, H], got {tuple(hidden_states.shape)}")
    
        extended_attention_mask = (1.0 - attention_mask[:, None, None, :].to(hidden_states.dtype)) * torch.finfo(hidden_states.dtype).min
        for layer_idx in range(self.module_cache_first_trainable_layer, len(self.encoder.encoder.layer),):
            layer_output = self.encoder.encoder.layer[layer_idx](hidden_states, attention_mask=extended_attention_mask,)
    
            # Transformers versions may return either
            # a Tensor directly or a tuple/list.
            if torch.is_tensor(layer_output):
                hidden_states = layer_output
            elif isinstance(layer_output, (tuple, list)):
                hidden_states = layer_output[0]
            elif hasattr(layer_output, "last_hidden_state"):
                hidden_states = layer_output.last_hidden_state
            else:
                raise TypeError(f"Unexpected output type from encoder layer {layer_idx}: {type(layer_output)}")
    
            if hidden_states.dim() != 3:
                raise RuntimeError(f"Layer {layer_idx} returned shape {tuple(hidden_states.shape)}, expected [B, S, H].")
    
        pooled = self.pool_embeddings(hidden_states, attention_mask, pooling_mode=self.pooling_mode,)
        return F.normalize(pooled, p=2, dim=-1,)

    def adapt_embeddings(self, pooled: torch.Tensor,) -> torch.Tensor:
        if self.uses_module_peft:
            raise RuntimeError("Cached embeddings cannot be used with module-level PEFT.")
    
        pooled = pooled.float().to(QUANTUM_DEVICE)
        adapted = self.adapter(pooled)
        return F.normalize(adapted, p=2, dim=-1,)

    def forward(self, texts: list[str], max_length: int | None = None,) -> torch.Tensor:
        return self.encode_texts(texts, max_length=max_length, is_query=False,)