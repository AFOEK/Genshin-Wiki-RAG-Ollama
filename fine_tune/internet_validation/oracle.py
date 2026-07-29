import json
import requests
import logging

log = logging.getLogger(__name__)

ORACLE_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {
            "type": "boolean",
        },
        "answer": {
            "type": "string",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "evidence_judgements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "evidence_id": {
                        "type": "string",
                    },
                    "relevance": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "supports_answer": {
                        "type": "boolean",
                    },
                    "contradicts_answer": {
                        "type": "boolean",
                    },
                    "reason": {
                        "type": "string",
                    },
                },
                "required": [
                    "evidence_id",
                    "relevance",
                    "supports_answer",
                    "contradicts_answer",
                    "reason",
                ],
            },
        },
        "reason": {
            "type": "string",
        },
    },
    "required": [
        "answerable",
        "answer",
        "confidence",
        "evidence_judgements",
        "reason",
    ],
}

def ollama_structured(*, ollama_url: str, model: str, system: str, prompt: str, schema: dict, timeout_s: float = 600) -> dict:
    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "think": False,
            "format": schema,
            "keep_alive": -1,
            "options": {
                "temperature": 0.0,
                "seed": 40151652,
            },
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]
    return json.loads(content)

def run_blind_oracle(cfg: dict, *, question: str, evidence: list[dict]) -> dict:
    safe_evidence = [
        {
            "evidence_id": row["evidence_id"],
            "source": row["source"],
            "title": row["title"],
            "url": row["url"],
            "text": row["text"],
        } for row in evidence]

    prompt = (
        "Answer the question using only the "
        "independent internet evidence below.\n\n"
        f"Question:\n{question}\n\n"
        "Evidence:\n"
        + json.dumps(
            safe_evidence,
            ensure_ascii=False))

    validation_cfg = cfg["internet_validation"]

    return ollama_structured(
        ollama_url=validation_cfg["ollama_url"],
        model=validation_cfg["ollama_model"],
        system=("You are an independent Genshin Impact fact-checking oracle. Do not infer unsupported details."),
        prompt=prompt,
        schema=ORACLE_SCHEMA,
    )