from __future__ import annotations

import logging
from qna.generators import generate
from qna.utils import as_bool

def build_hyde_prompt(question: str) -> str:
    return f"""
Write a short hypothetical Genshin Impact reference passage that would
likely contain the information needed to answer the question below.

The passage is used only as a search query. It will not be shown to the
user and must not be treated as factual evidence.

Rules:
- Use terminology likely to appear in Genshin Impact guides or wiki pages.
- Include the relevant character, item, quest, location, weapon, artifact,
  mechanic, or version names when they can be inferred from the question.
- Include useful aliases and closely related terminology.
- Keep the passage between 80 and 180 words.
- Do not include analysis, citations, headings, or disclaimers.
- Return only the hypothetical passage.

Question:
{question}
""".strip()

def retrieve_hyde_candidate(cfg: dict, question: str, *, candidate_k: int, backend: str) -> list[dict]:
    hyde_cfg = cfg.get("hyde", {}) or {}
    hypothetical_documents = generate(cfg, build_hyde_prompt(question), model_override=str(hyde_cfg.get("model", "qwen3.5:9b")), think_override=hyde_cfg.get("think", False), options_override={
        "temperature": float(hyde_cfg.get("temperature", 0.0)),
        "top_p": float(hyde_cfg.get("top_p", 0.9)),
        "top_k": int(hyde_cfg.get("top_k", 40)),
        "num_predict": int(hyde_cfg.get("num_predict", 256))
    })
    return FaissRetriever.search(FaissRetriever, hypothetical_documents, candidate_k)