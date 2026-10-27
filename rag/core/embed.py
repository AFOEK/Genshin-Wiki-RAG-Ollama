import requests
import struct
import time
import threading
import logging
import torch

from typing import Literal, Any
from pathlib import Path
from functools import lru_cache

from fine_tune.custom_peft.safetensors_loader import load_hq_retriever
from core.paths import resolve_storage_root

log = logging.getLogger(__name__)
thread_local=threading.local()
_HQ_MODEL_LOCK = threading.RLock()
session = requests.Session()

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

class NonRetryableEmbedError(RuntimeError):
    pass

def embedding_model_name(cfg: dict, backend: str | None = None,) -> str:
    provider = normalize_backend_name(backend or cfg.get("runtime", {}).get("embedding_provider", "ollama"))
    if provider == "quantum_peft":
        qcfg = cfg.get("quantum_peft", {})
        return str(qcfg.get("model_id") or qcfg.get("model_dir", ""))

    if provider == "llamacpp":
        return str(cfg.get("llamacpp", {}).get("embedding_model", ""))

    return str(cfg.get("ollama", {}).get("embedding_model", ""))

def auto_embedding_profile(model_name: str) -> str:
    m = model_name.lower()

    if "e5" in m:
        return "e5"
    if "bge" in m:
        return "bge"
    if "nomic" in m:
        return "nomic"
    if "octen" in m:
        return "octen"
    if "snowflake" in m or "arctic" in m:
        return "snowflake"
    if "embeddinggemma" in m or "embedding-gemma" in m or "gemma" in m:
        return "embeddinggemma"
    if "minilm" in m or "all-minilm" in m:
        return "none"

    return "none"

def apply_embedding_prompt(cfg: dict, text_or_texts: str | list[str] | tuple[str,...], *, mode: str, backend: str | None = None, title: str | list[str] | tuple[str,...] | None=None) -> str | list[str]:   
    prompt_cfg = cfg.get("embedding_prompts", {}) or {}

    enabled = str(prompt_cfg.get("enabled", True)).strip().lower() in ("1", "true", "yes", "y", "on")
    if not enabled:
        return text_or_texts

    model_name = embedding_model_name(cfg, backend)
    profile_name = str(prompt_cfg.get("profile", "auto")).strip().lower()

    if profile_name == "auto":
        profile_name = auto_embedding_profile(model_name)

    profiles = prompt_cfg.get("profiles", {}) or {}
    profile = profiles.get(profile_name, profiles.get("none", {})) or {}

    if mode == "query":
        prefix = str(profile.get("query_prefix", ""))
    elif mode in ("passage", "document", "doc"):
        prefix = str(profile.get("passage_prefix", ""))
    else:
        prefix = ""

    if not prefix:
        return text_or_texts

    def one(x: str, item_title: str = "") -> str:
        x = x or ""
        item_title = item_title or ""
        if x.startswith(prefix):
            return x
        
        if "{text}" in prefix:
            return prefix.replace("{title}", item_title or "").replace("{text}", x)
        return prefix + x
    
    if isinstance(text_or_texts, (list, tuple)):
        if isinstance(title ,(list, tuple)):
            titles = list(title)
            if len(titles) != len(text_or_texts):
                raise ValueError(f"[EMBED] Embedding title count mismatch: texts={len(text_or_texts)} titles={len(titles)}")
        else:
            titles = [title] * len(text_or_texts)
        return [one(str(x), str(t or "")) for x, t in zip(text_or_texts, titles)]

    single_title = title[0] if isinstance(title, (list, tuple)) and title else title
    return one(str(text_or_texts) ,str(single_title or ""))

def pack_vec(vec: list[float]) -> tuple[bytes, int]:
    return struct.pack(f"<{len(vec)}f", *vec), len(vec)

def get_session() -> requests.Session:
    session = getattr(thread_local, "session", None)
    if session is None:
        session = requests.Session()
        thread_local.session = session
    return session

def post_json(url: str, payload: dict, timeout: int = 180) -> dict[str, Any]:
    r = get_session().post(url, json=payload, timeout=timeout)
    if 400 <= r.status_code < 500 and r.status_code not in RETRYABLE_STATUS:
        try:
            msg = r.json().get("error") or r.text
        except Exception:
            msg = r.text
        raise NonRetryableEmbedError(f"HTTP {r.status_code}: {msg[:300]}")
    
    if r.status_code in RETRYABLE_STATUS:
        raise requests.HTTPError(f"Retryable HTTP {r.status_code}: {r.text[:300]}", response=r)
    
    r.raise_for_status()
    return r.json()

def normalize_backend_name(name: str | None) -> str | None:
    if name is None:
        return None
    x = str(name).strip().lower()
    if x in {"llama.cpp", "llamacpp", "llama_cpp"}:
        return "llamacpp"
    if x == "ollama":
        return "ollama"
    if x in {"quantum_peft", "quantum", "hq_peft"}:
        return "quantum_peft"
    raise RuntimeError(f"Unknown embedding backend: {name}")

def embed_ollama(base_url: str, model: str, text_or_texts, keep_alive: str, timeout: int):
    is_batch = isinstance(text_or_texts, (list, tuple))
    payload_input = list(text_or_texts) if is_batch else text_or_texts

    data = post_json(
        f"{base_url}/api/embed",
        {
            "model": model,
            "input": payload_input,
            "truncate": True,
            "keep_alive": keep_alive
        }, timeout = timeout)
    embeddings = data["embeddings"]
    if not is_batch:
        vec = embeddings[0]
        return pack_vec(vec)

    return [pack_vec(vec) for vec in embeddings]

def embed_llamacpp(base_url: str, model: str, text_or_texts, timeout: int):
    is_batch = isinstance(text_or_texts, (list, tuple))
    payload_input = list(text_or_texts) if is_batch else [text_or_texts]

    data = post_json(
        f"{base_url}/v1/embeddings",
        {
            "model": model,
            "input": payload_input,
        }, timeout = timeout)
    rows = [item["embedding"] for item in data["data"]]
    if not is_batch:
        return pack_vec(rows[0])
    return [pack_vec(vec) for vec in rows]

@lru_cache(maxsize=1)
def _load_hq_model(export_dir: str, device: str, weights_kind: str):
    return load_hq_retriever(
        export_dir,
        encoder_device=device,
        weights_kind=weights_kind,
    )

def embed_quantum(cfg: dict, text_or_texts, *, mode: str = "passage"):
    qcfg = cfg.get("quantum_peft", {}) or {}

    model_dir = qcfg.get("model_dir")
    if not model_dir:
        raise ValueError("Missing quantum_peft.model_dir in config.yaml")

    # Resolve model directory
    folder = Path(model_dir).expanduser()
    if not folder.is_absolute():
        folder = resolve_storage_root(cfg) / folder

    folder = folder.resolve()
    requested = str(qcfg.get("device", "auto")).lower()
    if requested == "auto":
        device = ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = requested

    if (torch.device(device).type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested for HQ embeddings, but CUDA is unavailable")

    batch_size = int(qcfg.get("batch_size", 8))
    if batch_size < 1:
        raise ValueError("quantum_peft.batch_size must be positive")

    is_batch = isinstance(text_or_texts, (list, tuple))
    texts = ([str(t or "") for t in text_or_texts] if is_batch else [str(text_or_texts or "")])
    if not texts:
        return []

    if mode not in ("query", "passage", "document", "doc"):
        raise ValueError(f"Unsupported HQ embedding mode: {mode}")

    weights_kind = str(qcfg.get("weights_kind", "auto"))
    results = []
    with _HQ_MODEL_LOCK, torch.no_grad():
        model = _load_hq_model(str(folder), str(torch.device(device)), weights_kind,)
        max_length = qcfg.get("max_length")
        if max_length is not None:
            max_length = int(max_length)

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            vectors = model.encode_texts(batch, max_length=max_length, is_query=(mode == "query"),)
            if (vectors.ndim != 2 or vectors.shape[0] != len(batch)):
                raise RuntimeError("Unexpected HQ embedding shape: {tuple(vectors.shape)}")

            if not torch.isfinite(vectors).all().item():
                raise RuntimeError("HQ returned nonfinite embedding values")

            results.extend([pack_vec(row.detach().float().cpu().tolist()) for row in vectors])
    return results if is_batch else results[0]

def embed(cfg: dict, text_or_texts, backend: str | None = None, retries: int=10, backoff_s: float=1.0, mode: Literal["passage", "query"] = "passage", title: str | list[str] | tuple[str,...] | None = None):
    runtime = cfg.get("runtime", {})
    provider = normalize_backend_name(backend if backend is not None  else runtime.get("embedding_provider", "ollama"))
    if provider == "quantum_peft":
        return embed_quantum(cfg, text_or_texts, mode=mode,)

    text_or_texts = apply_embedding_prompt(cfg, text_or_texts, mode=mode, backend=provider, title=title,)
    last_err =  None
    for attempt in range(retries):
        try:
            if provider ==  "ollama":
                ollama = cfg["ollama"]
                return embed_ollama(
                    ollama["base_url"],
                    ollama["embedding_model"],
                    text_or_texts,
                    ollama.get("embed_keep_alive", "15s"),
                    int(ollama.get("timeout", "180"))
                )
            
            if provider == "llamacpp":
                llamacpp = cfg["llamacpp"]
                return embed_llamacpp(
                    llamacpp["embedding_url"],
                    llamacpp["embedding_model"],
                    text_or_texts,
                    int(llamacpp.get("timeout", "180"))
                )

            raise RuntimeError(f"Unknown embedding provider: {provider}")
        
        except NonRetryableEmbedError:
            raise
        except (requests.RequestException, KeyError, ValueError, RuntimeError) as e:
            last_err = str(e)
            time.sleep(backoff_s * (2 ** attempt))
            log.warning("Embedding failed attempt=%d/%d err=%s", attempt + 1, retries, last_err)
    
    raise RuntimeError(f"{provider} embeddings failed after {retries} retries. Last error: {last_err}")