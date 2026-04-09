import os
import requests
import struct
import time
import logging

log = logging.getLogger(__name__)
session = requests.Session()

class NonRetryableEmbedError(RuntimeError):
    pass

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

def embed_ollama(base_url: str, model: str, text_or_texts, keep_alive: str, timeout: int):
    is_batch = isinstance(text_or_texts, (list, tuple))
    payload_input = list(text_or_texts) if is_batch else text_or_texts

    data = post_json(
        f"{base_url}/api/embed",
        json={
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

def embed_llamacpp(base_url: str, model: str, text_or_texts, keep_alive: str, timeout: int):
    is_batch = isinstance(text_or_texts, (list, tuple))
    payload_input = list(text_or_texts) if is_batch else [text_or_texts]

    data = post_json(
        f"{base_url}/v1/embeddings",
        {
            "model": model,
            "input": payload_input,
            "keep_alive": keep_alive
        },
        timeout=timeout
    )
    rows = [item["embedding"] for item in data["data"]]
    if not is_batch:
        return pack_vec(rows[0])
    return [pack_vec(vec) for vec in rows]

def embed(cfg: dict, text_or_texts, retries: int = 10, backoff_s: float = 1.0):
    runtime = cfg.get("runtime", {})
    provider = runtime.get("embedding_provider", "ollama").strip().lower()

    last_err =  None
    for attempt in range(retries):
        try:
            if provider ==  "ollama":
                ollama = cfg["ollama"]
                return embed_ollama(
                    ollama["base_url"],
                    ollama["embedding_model"],
                    text_or_texts,
                    ollama.get("embed_keep_alive", "15s")
                    ollama.get("timeout", "180")
                )
            
            if provider == "llamacpp":
                llamacpp = cfg["llamacpp"]
                return embed_llamacpp(
                    llamacpp["base_url"],
                    llamacpp["embedding_model"],
                    text_or_texts,
                    llamacpp.get("embed_keep_alive", None),
                    llamacpp.get("timeout", "180")
                )
            
            raise RuntimeError(f"Unknown embedding provider: {provider}")
        
        except NonRetryableEmbedError:
            raise
        except (requests.RequestException, KeyError, ValueError, RuntimeError):
            last_err = str(e)
            time.sleep(backoff_s * (2 ** attempt))
            log.warning("Embedding failed attempt=%d/%d err=%s", attempt + 1, retries, last_err)
    
    raise RuntimeError(f"{provider} embeddings failed after {retries} retries. Last error: {last_err}")