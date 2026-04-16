from __future__ import annotations

import argparse
import yaml

from qna.utils import load_cfg
from qna.engine import answer_question
from utils.logging_setup import setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--config", default="rag/config.yaml")
    ap.add_argument("--retriever", choices=["faiss", "sqlite"], default="faiss")
    ap.add_argument("--direct_top_k", type=int, default=12)
    ap.add_argument("--broad_top_k", type=int, default=60)
    ap.add_argument("--summarize_batch_size", type=int, default=8)
    ap.add_argument("--backend", default=None, choices=["ollama", "llamacpp", "llama.cpp"])
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    setup_logging(
        cfg.get("logging", {}).get("file"),
        cfg.get("logging", {}).get("level", "INFO")
    )

    answer = answer_question(
        cfg,
        args.question,
        prefer_faiss=(args.retriever == "faiss"),
        direct_top_k=args.direct_top_k,
        broad_top_k=args.broad_top_k,
        summarize_batch_size=args.summarize_batch_size,
        backend = args.backend
    )

    print("\n=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()