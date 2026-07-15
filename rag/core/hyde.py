from __future__ import annotations

import logging
from qna.generators import generate
from qna.utils import as_bool

log = logging.getLogger(__name__)

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

def generate_hyde_document(cfg: dict, question: str) -> str:
    hyde_cfg = cfg.get("hyde", {}) or {}

    if not as_bool(hyde_cfg.get("enabled", False)):
        return ""

    document = str(
        generate(cfg, build_hyde_prompt(question), model_override=str(hyde_cfg.get("model", "qwen3.5:9b")).strip(), timeout=int(hyde_cfg.get("timeout", 300)), think_override=hyde_cfg.get("think", False), options_override={
                "temperature": float(
                    hyde_cfg.get("temperature", 0.0)
                ),
                "top_p": float(
                    hyde_cfg.get("top_p", 0.9)
                ),
                "top_k": int(
                    hyde_cfg.get("top_k", 40)
                ),
                "min_p": float(
                    hyde_cfg.get("min_p", 0.05)
                ),
                "repeat_penalty": float(
                    hyde_cfg.get("repeat_penalty", 1.05)
                ),
                "num_predict": int(
                    hyde_cfg.get("num_predict", 256)
                ),
            },
        )
    ).strip()

    if not document:
        raise RuntimeError("HyDE generator returned an empty hypothetical document")
    log.info("[HYDE] generated hypothetical document question_chars=%d document_chars=%d", len(question), len(document))
    return document