from __future__ import annotations

import requests

session = requests.Session()

def ollama_generate(base_url: str, model: str, prompt: str, retries:int = 3, keep_alive: str = "10m", timeout:int = 300) -> str:
    for attempt in range(retries):
        try:
            r = session.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": keep_alive,
                    "options":{
                        "temperature": 0,
                        "top_p":0.9,
                    },
                },
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()
        except requests.exceptions.ReadTimeout:
            if attempt == 0:
                continue
            raise

def llamacpp_generate(base_url: str, model: str, prompt: str, timeout: int, retries: int = 8) -> str:
    for attempt in range(retries):
        try:
            r = session.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0,
                    "top_p": 0.9,
                    "stream": False,
                },
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.ReadTimeout:
            if attempt == 0:
                continue
            raise

def generate(cfg: dict, prompt: str, retries: int = 8, timeout: int = 300) -> str:
    runtime = cfg.get("runtime", {})
    provider = runtime.get("qa_provider", "ollama").strip().lower()

    if provider == "ollama":
        ollama = cfg["ollama"]
        return ollama_generate(
            ollama["base_url"],
            ollama.get("qa_model", ollama.get("model", "llama3.2")),
            prompt,
            retries,
            ollama.get("qa_keep_alive", "10m"),
            timeout,
        )

    if provider == "llamacpp":
        lc = cfg["llamacpp"]
        return llamacpp_generate(
            lc["base_url"],
            lc["qa_model"],
            prompt,
            int(lc.get("timeout", timeout)),
            retries
        )

    raise RuntimeError(f"Unknown QA provider: {provider}")