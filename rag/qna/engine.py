from __future__ import annotations

import logging, textwrap
from core.embed import embed
from core.paths import resolve_db_path, resolve_faiss_dir
from .utils import read_only_connect, normalize_query_vec, is_broad_question, chunk_batch, rerank_chunks, dedupe_chunks, detect_intent, filter_by_intent_source, rrf_fuse, as_bool
from .retrievers import FaissRetriever, SqliteEmbeddingRetriever, BM25Retriever
from .db_fetch import fetch_chunks
from .prompts import build_context, summarize_chunk_group, synthesize_final_answer
from .generators import generate
from .cross_encoder import cross_encoder_rerank
from .context_expand import expand_context_windows
from .parent_child import fetch_parent_context_chunks


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
    log.info("[QNA] Use reranker: %s", reranker_mode)

    retrieval_cfg = cfg.get("retrieval", {}) or {}
    candidate_k_cfg = int(retrieval_cfg.get("candidate_k", 85))
    deep_candidate_multiplier = int(retrieval_cfg.get("deep_candidate_multiplier", 5))
    dedup_max_per_doc = int(retrieval_cfg.get("dedup_max_per_doc", 2))

    rrf_k = int(retrieval_cfg.get("rrf_k", 60))

    if provider == "llamacpp":
        qa_timeout = int(cfg.get("llamacpp", {}).get("timeout", 300))
    else:
        qa_timeout = int(cfg["ollama"].get("timeout", 1800))

    intent = detect_intent(question)
    broad = is_broad_question(question)
    top_k = broad_top_k if broad else direct_top_k
    candidate_k = max(top_k * 5, candidate_k_cfg)
    log.info(
        "[QNA] retrieval cfg: top_k=%d candidate_k=%d deep_multiplier=%d dedup_max_per_doc=%d intent=%s broad=%s",
        top_k,
        candidate_k,
        deep_candidate_multiplier,
        dedup_max_per_doc,
        intent,
        broad,
    )

    faiss_ret_cache = None
    bm25_ret_cache = None
    q_vec_cache = None
    q_dims_cache = None

    def get_q_vec(ret):
        nonlocal q_vec_cache, q_dims_cache

        if q_vec_cache is None:
            q_blob, q_dims = embed(cfg, question, backend=backend)

            if q_dims != ret.dims:
                raise RuntimeError(
                    f"query embedding dims mismatch: query={q_dims} retriever={ret.dims}"
                )

            q_vec_cache = normalize_query_vec(q_blob, q_dims)
            q_dims_cache = q_dims
            return q_vec_cache

        if q_dims_cache != ret.dims:
            raise RuntimeError(
                f"cached query embedding dims mismatch: query={q_dims_cache} retriever={ret.dims}"
            )

        return q_vec_cache


    def get_faiss_ret():
        nonlocal faiss_ret_cache

        if faiss_ret_cache is None:
            faiss_ret_cache = FaissRetriever(faiss_dir)

        return faiss_ret_cache


    def get_bm25_ret():
        nonlocal bm25_ret_cache

        if bm25_ret_cache is None:
            bm25_ret_cache = BM25Retriever(conn)

        return bm25_ret_cache


    def search_embedding_retriever(ret, k: int):
        q_vec = get_q_vec(ret)
        return ret.search(q_vec, k)


    def search_hybrid(k: int):
        faiss_ret = get_faiss_ret()
        bm25_ret = get_bm25_ret()

        q_vec = get_q_vec(faiss_ret)

        faiss_results = faiss_ret.search(q_vec, k)
        bm25_results = bm25_ret.search(question.lower(), k)

        return rrf_fuse(faiss_results, bm25_results, k=rrf_k)

    retriever_name = retriever_name.strip().lower()
    if retriever_name == "sql":
        retriever_name = "sqlite"

    if retriever_name == "faiss":
        log.info("[QNA] using FAISS retriever")
        retriever = get_faiss_ret()
        results = search_embedding_retriever(retriever, candidate_k)
    elif retriever_name == "sqlite":
        log.info("[QNA] using SQLite brute-force retriever")
        retriever = SqliteEmbeddingRetriever(conn)
        results = search_embedding_retriever(retriever, candidate_k)
    elif retriever_name == "bm25":
        log.info("[QNA] using SQLite BM25 retriever")
        retriever = get_bm25_ret()
        results = retriever.search(question.lower(), candidate_k)
    elif retriever_name == "hybrid":
        log.info("[QNA] using HYBRID retriever (FAISS + BM25)")
        retriever = None
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
        deep_k = candidate_k * deep_candidate_multiplier
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
    max_per_doc = dedup_max_per_doc
    if intent in ("biography", "location"):
        max_per_doc = max(dedup_max_per_doc, 2)
    chunks = dedupe_chunks(chunks, initial_scores, max_per_doc=max_per_doc)

    if reranker_mode in ("feature", "cross_encoder"):
        chunks = rerank_chunks(question, chunks, initial_scores)
    
    if reranker_mode == "cross_encoder":
        chunks = cross_encoder_rerank(
            question, 
            chunks, 
            model_name=reranker_cfg.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"), 
            top_n=int(reranker_cfg.get("cross_encoder_top_n", 32)),
            batch_size=int(reranker_cfg.get("cross_encoder_batch_size", 8)),
            max_pair_text_chars=int(reranker_cfg.get("max_pair_text_chars", 1200))
            )
    elif reranker_mode not in ("none", "feature", "cross_encoder"):
        raise RuntimeError(f"Unknown reranker mode: {reranker_mode}")

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
        context_cfg = cfg.get("context_expansion", {}) or {}
        ctx_enabled = as_bool(context_cfg.get("enabled", False))

        selected_chunks = chunks[:direct_top_k]

        parent_cfg = cfg.get("parent_child", {}) or {}
        parent_enabled = as_bool(parent_cfg.get("enabled", False))


        if parent_enabled:
            selected_chunks = fetch_parent_context_chunks(
                conn,
                selected_chunks,
                max_parents=int(parent_cfg.get("max_parents", 8)),
                max_total_chunks=int(parent_cfg.get("max_total_chunks", 32)),
            )

            log.info("[PARENT] expanded selected chunks to parent context chunks=%d", len(selected_chunks))

        if ctx_enabled:
            ctx_max_total = int(context_cfg.get("max_total_chunks_after_parent", context_cfg.get("max_total_chunks", 30)))

            selected_chunks = expand_context_windows(
                conn,
                selected_chunks,
                before=int(context_cfg.get("before", 1)),
                after=int(context_cfg.get("after", 1)),
                max_total_chunks=ctx_max_total,
            )

            log.info("[CTX_EXPAND] expanded context chunks=%d before=%s after=%s", len(selected_chunks), context_cfg.get("before", 1), context_cfg.get("after", 1))

        context = build_context(selected_chunks)
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