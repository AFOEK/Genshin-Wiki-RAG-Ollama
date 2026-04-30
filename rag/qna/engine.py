from __future__ import annotations

import logging, textwrap
from core.embed import embed
from core.paths import resolve_db_path, resolve_faiss_dir
from .utils import read_only_connect, normalize_query_vec, is_broad_question, chunk_batch, rerank_chunks, dedupe_chunks, detect_intent, filter_by_intent_source, rrf_fuse
from .retrievers import FaissRetriever, SqliteEmbeddingRetriever, BM25Retriever
from .db_fetch import fetch_chunks
from .prompts import build_context, summarize_chunk_group, synthesize_final_answer
from .generators import generate
from .cross_encoder import cross_encoder_rerank


log = logging.getLogger(__name__)

def answer_question(
    cfg: dict,
    question: str,
    *,
    retriever_name: str = "hybrid",
    direct_top_k: int = 12,
    broad_top_k: int = 60,
    summarize_batch_size: int = 8,
    backend: str | None = None) -> str:
    db_path = resolve_db_path(cfg)
    faiss_dir = resolve_faiss_dir(cfg)
    conn = read_only_connect(str(db_path))

    runtime = cfg.get("runtime", {})
    provider = runtime.get("qa_provider", "ollama").strip().lower()
    log.info("[QNA] Use provider: %s", provider)

    reranker_cfg = cfg.get("reranker", {})
    reranker_mode = str(reranker_cfg.get("mode", "feature")).strip().lower()

    if provider == "llamacpp":
        qa_timeout = int(cfg.get("llamacpp", {}).get("timeout", 300))
    else:
        qa_timeout = int(cfg["ollama"].get("timeout", 1800))

    intent = detect_intent(question)
    broad = is_broad_question(question)
    top_k = broad_top_k if broad else direct_top_k
    candidate_k = max(top_k * 5, 85)

    retriever_name = retriever_name.strip().lower()
    if retriever_name == "sql":
        retriever_name = "sqlite"

    def search_embedding_retriever(ret, k: int):
        q_blob, q_dims = embed(cfg, question, backend=backend)
        if q_dims != ret.dims:
            raise RuntimeError(
                f"query embedding dims mismatch: query={q_dims} retriever={ret.dims}"
            )
        q_vec = normalize_query_vec(q_blob, q_dims)
        return ret.search(q_vec, k)

    def search_hybrid(k: int):
        faiss_ret = FaissRetriever(faiss_dir)
        bm25_ret = BM25Retriever(conn)

        q_blob, q_dims = embed(cfg, question, backend=backend)
        if q_dims != faiss_ret.dims:
            raise RuntimeError(
                f"query embedding dims mismatch: query={q_dims} retriever={faiss_ret.dims}"
            )

        q_vec = normalize_query_vec(q_blob, q_dims)
        faiss_results = faiss_ret.search(q_vec, k)
        bm25_results = bm25_ret.search(question.lower(), k)

        return rrf_fuse(faiss_results, bm25_results, k=60)

    if retriever_name == "faiss":
        log.info("[QNA] using FAISS retriever")
        retriever = FaissRetriever(faiss_dir)
        results = search_embedding_retriever(retriever, candidate_k)
    elif retriever_name == "sqlite":
        log.info("[QNA] using SQLite brute-force retriever")
        retriever = SqliteEmbeddingRetriever(conn)
        results = search_embedding_retriever(retriever, candidate_k)
    elif retriever_name == "bm25":
        log.info("[QNA] using SQLite BM25 retriever")
        retriever = BM25Retriever(conn)
        results = retriever.search(question.lower(), candidate_k)
    elif retriever_name == "hybrid":
        log.info("[QNA] using HYBRID retriever (FAISS + BM25)")
        results = search_hybrid(candidate_k)
    else:
        raise RuntimeError(f"Unknown retriever: {retriever_name}")

    chunk_ids = [cid for cid, score in results]
    initial_scores = {cid: score for cid, score in results}

    filtered_ids = filter_by_intent_source(
        conn,
        chunk_ids,
        intent,
        min_required=5,
        max_fallback=20,
    )

    if intent in ("build", "mechanic", "lore", "biography", "location") and len(filtered_ids) < 3:
        deep_k = candidate_k * 5
        log.info("[QNA] intent filter returned too few chunks; deep search k=%d", deep_k)

        if retriever_name == "faiss":
            deep_results = search_embedding_retriever(retriever, deep_k)
        elif retriever_name == "sqlite":
            deep_results = search_embedding_retriever(retriever, deep_k)
        elif retriever_name == "bm25":
            deep_results = retriever.search(question.lower(), deep_k)
        elif retriever_name == "hybrid":
            deep_results = search_hybrid(deep_k)
        else:
            raise RuntimeError(f"Unknown retriever: {retriever_name}")

        deep_ids = [cid for cid, _ in deep_results]
        filtered_ids = filter_by_intent_source(
            conn,
            deep_ids,
            intent,
            min_required=5,
            max_fallback=40,
        )

        deep_scores = {cid: score for cid, score in deep_results}
        results = [(cid, deep_scores[cid]) for cid in filtered_ids if cid in deep_scores]
        chunk_ids = [cid for cid, _ in results]
        initial_scores = {cid: score for cid, score in results}

    else:
        id_set = set(filtered_ids)
        results = [(cid, score) for cid, score in results if cid in id_set]
        chunk_ids = [cid for cid, _ in results]
        initial_scores = {cid: score for cid, score in results}

    chunks = fetch_chunks(conn, chunk_ids)
    max_per_doc = 3 if intent in ("biography", "location") else 2
    chunks = dedupe_chunks(chunks, initial_scores, max_per_doc=max_per_doc)
    chunks = rerank_chunks(question, chunks, initial_scores)

    if reranker_mode == "cross_encoder":
        chunks = cross_encoder_rerank(question, chunks, model_name=reranker_cfg.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"), top_n=int(reranker_cfg.get("cross_encoder_top_n", 32)), batch_size=int(reranker_cfg.get("cross_encoder_batch_size", 8)), max_pair_text_chars=int(reranker_cfg.get("max_pair_text_chars", 1200)),),


    if not chunks:
        try:
            conn.close()
        except Exception:
            pass
        return "I couldn't retrieve any relevant chunks from the knowledge base."

    for row in chunks[:direct_top_k]:
        log.info(
            "[QNA] chunk_id=%s title=%s source=%s preview=%s",
            row["chunk_id"],
            row["title"],
            row["source"],
            (row["text"][:200] if row["text"] else "").replace("\n", " "),
        )

    if not broad:
        context = build_context(chunks[:direct_top_k])
        prompt = textwrap.dedent(f"""
            You are a retrieval-grounded Genshin Impact assistant.
            Answer the question using ONLY the provided context.
            If the answer is not explicitly supported by the context, say:
            "I don't have enough evidence in the retrieved context."

            Rules:
            - Use ONLY information present in the context.
            - You MAY infer reasonable conclusions directly supported by the context.
            - Cite chunk IDs inline like [chunk_id=123].
            - Cite source name like [source_name=xyz].
            - If the context genuinely does not contain enough information, say so briefly.
            - Do not guess facts not supported by the context.

            Question:
            {question}

            Context:
            {context}
        """).strip()
        result = generate(cfg, prompt)
        try:
            conn.close()
        except Exception:
            pass
        return result
    
    notes = []
    for group in chunk_batch(chunks, summarize_batch_size):
        notes.append(summarize_chunk_group(cfg, question, group))

    result = synthesize_final_answer(cfg, question, notes, qa_timeout)
    try:
        conn.close()
    except Exception:
        pass
    return result