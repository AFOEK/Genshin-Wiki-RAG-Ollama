from __future__ import annotations

import requests

def ollama_generate(base_url: str, model: str, prompt: str, retries:int = 3, keep_alive: str = "30s", timeout:int = 300) -> str:
    for attempt in range(retries):
        try:
            r = requests.post(
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