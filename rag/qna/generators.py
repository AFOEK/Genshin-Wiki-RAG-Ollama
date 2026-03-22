from __future__ import annotations

import requests

def ollama_generate(base_url: str, model: str, prompt: str, keep_alive: str = "30s") -> str:
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
        timeout=1000,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()