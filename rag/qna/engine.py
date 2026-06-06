from __future__ import annotations

import logging, textwrap

from pathlib import Path

from core.embed import embed
from core.paths import resolve_db_path, resolve_faiss_dir
from .utils import read_only_connect, normalize_query_vec, is_broad_question, chunk_batch, rerank_chunks, dedupe_chunks, detect_intent, filter_by_intent_source, as_bool, get_kqm_news_fetch_version_baseline, prefer_entity_seed_chunks, build_hybrid_signal, expected_model_from_cfg, make_intent_fts5_query, get_bm25_weights, detect_build_subtypes
from .retrievers import FaissRetriever, SqliteEmbeddingRetriever, BM25Retriever, TurboVecRetriever
from .db_fetch import fetch_chunks
from .prompts import build_context, summarize_chunk_group, synthesize_final_answer
from .generators import generate
from .cross_encoder import cross_encoder_rerank
from .context_expand import expand_context_windows
from .parent_child import fetch_parent_context_chunks


log = logging.getLogger(__name__)

def answer_question(cfg: dict, question: str, *, retriever_name: str = "hybrid", direct_top_k: int = 12, broad_top_k: int = 60, summarize_batch_size: int = 8, backend: str | None = None) -> str:
    db_path = resolve_db_path(cfg)
    faiss_dir = resolve_faiss_dir(cfg)
    tv_cfg = cfg.get("turbovec", {}) or {}
    tv_raw = Path(str(tv_cfg.get("path", "data/turbovec")))
    turbovec_dir = tv_raw if tv_raw.is_absolute() else db_path.parent.parent / tv_raw
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
    build_subtypes = (detect_build_subtypes(question) if intent == "build" else set())
    broad = is_broad_question(question)
    top_k = broad_top_k if broad else direct_top_k
    bm25_weights = get_bm25_weights(intent)
    candidate_k = max(top_k * 5, candidate_k_cfg)
    log.info(
        "[QNA] retrieval cfg: top_k=%d candidate_k=%d deep_multiplier=%d dedup_max_per_doc=%d intent=%s subtypes=%s broad=%s",
        top_k,
        candidate_k,
        deep_candidate_multiplier,
        dedup_max_per_doc,
        intent,
        sorted(build_subtypes),
        broad,
    )

    faiss_ret_cache = None
    bm25_ret_cache = None
    turbovec_ret_cache = None
    q_vec_cache = None
    q_dims_cache = None

    def get_q_vec(ret):
        nonlocal q_vec_cache, q_dims_cache

        if q_vec_cache is None:
            q_blob, q_dims = embed(cfg, question, backend=backend, mode="query")

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

    expected_faiss_model = expected_model_from_cfg(cfg, backend=backend)
    faiss_mismatch_policy = str(retrieval_cfg.get("faiss_model_mismatch", "error")).strip().lower()
    
    def build_single_channel_results(raw_results: list[tuple[int, float]], channel: str) -> tuple[list[tuple[int, float]], dict[int, dict]]:
        scale = float(retrieval_cfg.get("rrf_scale", 10.0))
        signals: dict[int, dict] = {}

        for rank, (cid, raw_score) in enumerate(raw_results, start=1):
            cid = int(cid)

            signal = {
                "rrf_score": scale / (rrf_k + rank),
                "faiss_score": 0.0,
                "bm25_score": 0.0,
                "faiss_rank": None,
                "bm25_rank": None,
                "in_faiss": False,
                "in_bm25": False,
            }

            if channel == "bm25":
                signal["bm25_score"] = float(raw_score)
                signal["bm25_rank"] = rank
                signal["in_bm25"] = True
            else:
                # Treat FAISS, SQLite-vector and TurboVec as semantic channels.
                signal["faiss_score"] = float(raw_score)
                signal["faiss_rank"] = rank
                signal["in_faiss"] = True

            signals[cid] = signal

        ranked = sorted(
            (
                (cid, signal["rrf_score"])
                for cid, signal in signals.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked, signals

    def get_faiss_ret():
        nonlocal faiss_ret_cache

        if faiss_ret_cache is None:
            faiss_ret_cache = FaissRetriever(faiss_dir, expected_model=expected_faiss_model, mismatch_policy=faiss_mismatch_policy)
        return faiss_ret_cache


    def get_bm25_ret():
        nonlocal bm25_ret_cache

        if bm25_ret_cache is None:
            bm25_ret_cache = BM25Retriever(conn)

        return bm25_ret_cache

    def merge_ranked_results(primary: list[tuple[int, float]], fallback: list[tuple[int, float]], k: int) -> list[tuple[int, float]]:
        merged = []
        seen = set()

        for cid, score in primary + fallback:
            cid = int(cid)

            if cid in seen:
                continue

            seen.add(cid)
            merged.append((cid, float(score)))

            if len(merged) >= k:
                break

        return merged
    
    def search_bm25(k: int):
        bm25_ret = get_bm25_ret()

        strict_query = make_intent_fts5_query(question, intent)
        log.info("[BM25] intent=%s weights=%s", intent, bm25_weights)

        if not strict_query:
            return bm25_ret.search(question, k, weights=bm25_weights)

        log.info("[BM25] strict intent query=%s", strict_query)

        strict_results = bm25_ret.search_fts(strict_query, k, weights=bm25_weights)
        minimum_strict = min(5, k)

        if len(strict_results) >= minimum_strict:
            return strict_results

        log.info("[BM25] strict query returned only %d rows; adding broad fallback", len(strict_results))
        broad_results = bm25_ret.search(question, k, weights=bm25_weights)

        return merge_ranked_results(strict_results, broad_results, k)

    def get_turbovec_ret():
        nonlocal turbovec_ret_cache

        if turbovec_ret_cache is None:
            tv_cfg = cfg.get("turbovec", {}) or {}
            expected_model = expected_model_from_cfg(cfg, backend=backend, source=str(tv_cfg.get("model_source", "runtime")))

            turbovec_ret_cache = TurboVecRetriever(turbovec_dir, expected_model=expected_model, mismatch_policy=str(tv_cfg.get("model_mismatch", "error")))

        return turbovec_ret_cache

    def search_embedding_retriever(ret, k: int):
        q_vec = get_q_vec(ret)
        return ret.search(q_vec, k)


    def search_hybrid(k: int):
        faiss_ret = get_faiss_ret()
        q_vec = get_q_vec(faiss_ret)

        faiss_results = faiss_ret.search(q_vec, k)
        bm25_results = search_bm25(k)

        signals = build_hybrid_signal(faiss_results, bm25_results, rrf_k=rrf_k, rrf_scale=float(retrieval_cfg.get("rrf_scale", 10.0)))

        results = sorted(((cid, sig["rrf_score"]) for cid, sig in signals.items()), key=lambda x: x[1], reverse=True)

        return results, signals

    def search_hybrid_turbovec(k: int):
        tv_ret = get_turbovec_ret()
        q_vec = get_q_vec(tv_ret)

        tv_results = tv_ret.search(q_vec, k)
        bm25_results = search_bm25(k)

        signals = build_hybrid_signal(tv_results, bm25_results, rrf_k=rrf_k, rrf_scale=float(retrieval_cfg.get("rrf_scale", 10.0)))

        results = sorted(((cid, sig["rrf_score"]) for cid, sig in signals.items()), key=lambda x: x[1], reverse=True)

        return results, signals

    retriever_name = retriever_name.strip().lower()
    if retriever_name == "sql":
        retriever_name = "sqlite"

    if retriever_name == "faiss":
        log.info("[QNA] using FAISS retriever")
        retriever = get_faiss_ret()
        raw_results = search_embedding_retriever(retriever, candidate_k)
        results, retrieval_signals = build_single_channel_results(raw_results, "semantic")
    elif retriever_name == "sqlite":
        log.info("[QNA] using SQLite brute-force retriever")
        retriever = SqliteEmbeddingRetriever(conn)
        results = search_embedding_retriever(retriever, candidate_k)
        retrieval_signals = {cid: score for cid, score in results}
    elif retriever_name == "bm25":
        log.info("[QNA] using SQLite BM25 retriever")
        retriever = get_bm25_ret()
        raw_results = search_bm25(candidate_k)
        results, retrieval_signals = build_single_channel_results(raw_results, "bm25")
    elif retriever_name == "hybrid":
        log.info("[QNA] using HYBRID retriever (FAISS + BM25)")
        retriever = None
        results, retrieval_signals = search_hybrid(candidate_k)
    elif retriever_name == "turbovec":
        log.info("[QNA] using TurboVec retriever")
        retriever = get_turbovec_ret()
        results = search_embedding_retriever(retriever, candidate_k)
        retrieval_signals = {cid: score for cid, score in results}
    elif retriever_name == "hybrid_turbovec":
        log.info("[QNA] using HYBRID retriever (TurboVec + BM25)")
        retriever = None
        results, retrieval_signals = search_hybrid_turbovec(candidate_k)
    else:
        raise RuntimeError(f"Unknown retriever: {retriever_name}")

    chunk_ids = [cid for cid, score in results]
    rank_scores = {cid: (float(sig.get("rrf_score", 0.0)) if isinstance(sig, dict) else float(sig)) for cid, sig in retrieval_signals.items()}
    initial_scores = rank_scores

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
            retrieval_signals = {cid: score for cid, score in deep_results}
        elif retriever_name == "sqlite":
            deep_results = search_embedding_retriever(retriever, deep_k)
            retrieval_signals = {cid: score for cid, score in deep_results}
        elif retriever_name == "bm25":
            raw_deep_results = search_bm25(deep_k)
            deep_results, retrieval_signals = build_single_channel_results(raw_deep_results, "bm25")
        elif retriever_name == "hybrid":
            deep_results, retrieval_signals = search_hybrid(deep_k)
        elif retriever_name == "turbovec":
            deep_results = search_embedding_retriever(retriever, deep_k)
            retrieval_signals = {cid: score for cid, score in deep_results}
        elif retriever_name == "hybrid_turbovec":
            deep_results, retrieval_signals = search_hybrid_turbovec(deep_k)
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

        deep_scores = {cid: (float(sig.get("rrf_score", 0.0)) if isinstance(sig, dict) else float(sig)) for cid, sig in retrieval_signals.items()}
        results = [(cid, deep_scores[cid]) for cid in filtered_ids if cid in deep_scores]
        chunk_ids = [cid for cid, _ in results]
        initial_scores = deep_scores

    else:
        id_set = set(filtered_ids)
        results = [(cid, score) for cid, score in results if cid in id_set]
        chunk_ids = [cid for cid, _ in results]
        initial_scores = {cid: score for cid, score in results}

    chunks = fetch_chunks(conn, chunk_ids)
    max_per_doc = dedup_max_per_doc
    if intent == "build":
        max_per_doc = max(dedup_max_per_doc, 6)
    elif intent in ("biography", "location"):
        max_per_doc = max(dedup_max_per_doc, 2)

    baseline_label, baseline_ord = get_kqm_news_fetch_version_baseline(conn)
    log.info(
        "[QNA] current version baseline from kqm_news: label=%s ord=%s",
        baseline_label,
        baseline_ord,
    )

    if reranker_mode in ("feature", "cross_encoder"):
        chunks = rerank_chunks(
            question,
            chunks,
            retrieval_signals,
            baseline_ord,
        )
        dedupe_scores = {int(row["chunk_id"]): float(row["_rerank_score"]) for row in chunks}
    else:
        dedupe_scores = initial_scores

    chunks = dedupe_chunks(chunks, dedupe_scores, max_per_doc=max_per_doc)
    
    if reranker_mode == "cross_encoder":
        chunks = cross_encoder_rerank(
            question, 
            chunks, 
            model_name=reranker_cfg.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"), 
            top_n=int(reranker_cfg.get("cross_encoder_top_n", 32)),
            batch_size=int(reranker_cfg.get("cross_encoder_batch_size", 8)),
            max_pair_text_chars=int(reranker_cfg.get("max_pair_text_chars", 1200)))
    elif reranker_mode not in ("none", "feature", "cross_encoder"):
        raise RuntimeError(f"Unknown reranker mode: {reranker_mode}")

    for row in chunks:
        row.pop("_rerank_score", None)

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
        if intent in ("biography", "location"):
            selected_chunks = prefer_entity_seed_chunks(question, selected_chunks, min_keep=3)
            selected_chunks = selected_chunks[:direct_top_k]

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
        log.info("[CONTEXT] final chunk IDs=%s", [int(row["chunk_id"]) for row in selected_chunks])
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