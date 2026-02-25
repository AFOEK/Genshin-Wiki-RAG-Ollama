import requests
import struct
import time
import logging

log = logging.getLogger(__name__)

def embed(ollama_base: str, model: str, text: str, retries: int = 5, backoff_s: float = 1.0) -> tuple[bytes, int]:
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{ollama_base}/api/embed",
                json={"model": model, "input": text, "truncate":True},
                timeout=180
            )
            if r.status_code == 400:
                try:
                    msg = r.json().get("error") or r.text
                except Exception:
                    msg = r.text
                    log.exception("Failed with connectiong to embed backend ollama")
                raise RuntimeError("HTTP 400: %s", msg[:300])
            if r.status_code >= 500:
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(backoff_s * (2 ** attempt))
                continue
            r.raise_for_status()
            data = r.json()
            vec = data["embeddings"][0]
            return struct.pack(f"<{len(vec)}f", *vec), len(vec)
        except (requests.RequestException, KeyError, ValueError) as e:
            last_err = str(e)
            time.sleep(backoff_s * (2 ** attempt))
            log.exception("Failed due to embedding failure, retrying")
    log.error("Ollama embeddings failed after %d retries. Last error: %s", retries, last_err)
    raise RuntimeError("Ollama embeddings failed after %d retries. Last error: %s", retries, last_err)