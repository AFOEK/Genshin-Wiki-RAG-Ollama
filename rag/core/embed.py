import requests
import struct
import time
import logging
from typing import Literal

log = logging.getLogger(__name__)
session = requests.Session()

class NonRetryableEmbedError(RuntimeError):
    pass

def embedding_model_name(cfg: dict, backend: str | None = None) -> str:
    provider = (backend or cfg.get("runtime", {}).get("embedding_provider", "ollama")).strip().lower()
    if provider == "llama.cpp" or provider == "llamacpp":
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

def apply_embedding_prompt(cfg: dict, text_or_texts, *, mode: str, backend: str | None = None):
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

    def one(x: str) -> str:
        x = x or ""
        if x.startswith(prefix):
            return x
        
        if "{text}" in prefix:
            return prefix.replace("{title}", "").replace("{text}", x)
        return prefix + x
    
    if isinstance(text_or_texts, (list, tuple)):
        return [one(str(x)) for x in text_or_texts]
    return one(str(text_or_texts))

def pack_vec(vec: list[float]) -> tuple[bytes, int]:
    return struct.pack(f"<{len(vec)}f", *vec), len(vec)

def post_json(url: str, payload: dict, timeout: int = 180):
    r = session.post(url, json=payload, timeout=timeout)

    if r.status_code == 400:
        try:
            msg = r.json().get("error") or r.text
        except Exception:
            msg = r.text
        raise NonRetryableEmbedError(f"HTTP 400: {msg[:300]}")
    
    if r.status_code >= 500:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}", response=r)
    
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
        }
        ,
        timeout=timeout
    )
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
        },
        timeout=timeout
    )
    rows = [item["embedding"] for item in data["data"]]
    if not is_batch:
        return pack_vec(rows[0])
    return [pack_vec(vec) for vec in rows]

def embed(cfg: dict, text_or_texts, backend: str | None = None, retries: int = 10, backoff_s: float = 1.0, mode: Literal["passage", "query"] = "passage"):
    runtime = cfg.get("runtime", {})
    provider = normalize_backend_name(backend if backend is not None else runtime.get("embedding_provider", "ollama"))
    text_or_texts = apply_embedding_prompt(cfg, text_or_texts, mode=mode, backend=backend)
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