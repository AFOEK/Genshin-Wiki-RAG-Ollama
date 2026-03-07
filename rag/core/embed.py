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

def embed(ollama_base: str, model: str, text_or_texts, retries: int = 10, backoff_s: float = 1.0, keep_alive: str = "30s"):
    is_batch = isinstance(text_or_texts, (list, tuple))
    payload_input = list(text_or_texts) if is_batch else text_or_texts
    last_err = None

    for attempt in range(retries):
        try:
            r = session.post(
                f"{ollama_base}/api/embed",
                json={
                    "model": model,
                    "input": payload_input,
                    "truncate": True,
                    "keep_alive": keep_alive
                },
                timeout=180
            )

            if r.status_code == 400:
                try:
                    msg = r.json().get("error") or r.text
                except Exception:
                    msg = r.text

                msg = msg[:300]
                raise NonRetryableEmbedError(f"HTTP 400: {msg}")

            if r.status_code >= 500:
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(backoff_s * (2 ** attempt))
                continue

            r.raise_for_status()
            data = r.json()
            embeddings = data["embeddings"]

            if not is_batch:
                vec = embeddings[0]
                return pack_vec(vec)

            return [pack_vec(vec) for vec in embeddings]

        except NonRetryableEmbedError:
            raise
        except (requests.RequestException, KeyError, ValueError) as e:
            last_err = str(e)
            time.sleep(backoff_s * (2 ** attempt))
            log.warning("Embedding failed attempt=%d/%d err=%s", attempt + 1, retries, last_err)

    raise RuntimeError(f"Ollama embeddings failed after {retries} retries. Last error: {last_err}")