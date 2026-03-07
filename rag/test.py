from __future__ import annotations

import argparse

from qna.utils import setup_basic_logging, load_cfg
from qna.engine import answer_question


def main():
    setup_basic_logging()

    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--config", default="rag/config.yaml")
    ap.add_argument("--retriever", choices=["faiss", "sqlite"], default="faiss")
    ap.add_argument("--direct_top_k", type=int, default=12)
    ap.add_argument("--broad_top_k", type=int, default=60)
    ap.add_argument("--summarize_batch_size", type=int, default=8)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    answer = answer_question(
        cfg,
        args.question,
        prefer_faiss=(args.retriever == "faiss"),
        direct_top_k=args.direct_top_k,
        broad_top_k=args.broad_top_k,
        summarize_batch_size=args.summarize_batch_size,
    )

    print("\n=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()