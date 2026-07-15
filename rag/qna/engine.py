from __future__ import annotations

import logging, re

from pathlib import Path

from core.embed import embed
from core.paths import resolve_db_path, resolve_faiss_dir, resolve_storage_root
from core.hyde import generate_hyde_document
from .utils import read_only_connect, normalize_query_vec, is_broad_question, chunk_batch, rerank_chunks, dedupe_chunks, detect_intent, filter_by_intent_source, as_bool, get_kqm_news_fetch_version_baseline, prefer_entity_seed_chunks, build_hybrid_signal, build_hybrid_hyde_signal, expected_model_from_cfg, make_intent_fts5_query, get_bm25_weights, detect_build_subtypes, extract_lookup_entity, make_retrieval_cache_key, retrieval_result_from_cache, retrieval_result_to_cache
from .retrievers import FaissRetriever, SqliteEmbeddingRetriever, BM25Retriever, TurboVecRetriever
from .retrieval_cache import RetrievalCache
from .db_fetch import fetch_chunks
from .prompts import build_context, summarize_chunk_group, synthesize_final_answer
from .generators import generate
from .cross_encoder import cross_encoder_rerank
from .context_expand import expand_context_windows
from .parent_child import fetch_parent_context_chunks
from .types import RetrievalResult

log = logging.getLogger(__name__)
_RETRIEVAL_CACHE : RetrievalCache | None = None

def get_retrieval_cache(cfg: dict) -> RetrievalCache | None:
    global _RETRIEVAL_CACHE
    cache_cfg = cfg.get("retrieval_cache", {}) or {}
    if not as_bool(cache_cfg.get("enabled", False)):
        return None
    
    if _RETRIEVAL_CACHE is None:
        root = resolve_storage_root(cfg)
        rel = Path(str(cache_cfg.get("path", "data/cache/retrieval_cache.sqlite")))
        path = rel if rel.is_absolute() else root / rel
        _RETRIEVAL_CACHE = RetrievalCache(path, ttl_seconds=int(cache_cfg.get("ttl_seconds", 86400)), max_entries=int(cache_cfg.get("max_entries", 50000)))
    return _RETRIEVAL_CACHE

def retrieve_question_context_uncached(cfg: dict, question: str, *, retriever_name: str = "hybrid", direct_top_k: int = 12, broad_top_k: int = 60, backend: str | None = None) -> RetrievalResult:
    strict_fts_query_used: str | None = None
    db_path = resolve_db_path(cfg)
    faiss_dir = resolve_faiss_dir(cfg)
    tv_cfg = cfg.get("turbovec", {}) or {}
    tv_raw = Path(str(tv_cfg.get("path", "data/turbovec")))
    turbovec_dir = tv_raw if tv_raw.is_absolute() else db_path.parent.parent / tv_raw
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

    intent = detect_intent(question)
    build_subtypes = (detect_build_subtypes(question) if intent == "build" else set())
    broad = is_broad_question(question)
    top_k = broad_top_k if broad else direct_top_k
    bm25_weights = get_bm25_weights(intent)
    candidate_k = max(top_k, candidate_k_cfg)
    log.info("[QNA] retrieval cfg: top_k=%d candidate_k=%d deep_multiplier=%d dedup_max_per_doc=%d intent=%s subtypes=%s broad=%s", top_k, candidate_k, deep_candidate_multiplier, dedup_max_per_doc, intent, sorted(build_subtypes), broad)

    faiss_ret_cache = None
    bm25_ret_cache = None
    turbovec_ret_cache = None
    q_vec_cache = None
    q_dims_cache = None
    hyde_document_cache: str | None = None
    hyde_used_for_request = False
    hyde_fallback_reason: str | None = None
    hyde_error: str | None = None
    
    hyde_cfg = cfg.get("hyde", {}) or {}
    hyde_enabled = as_bool(hyde_cfg.get("enabled", False))
    hyde_mode = str(hyde_cfg.get("mode", "fallback")).strip().lower()
    if hyde_mode not in {"always", "fallback", "off", "disabled", "never"}:
        raise ValueError(f"Unsupported HyDE mode: {hyde_mode!r}")

    def get_q_vec(ret):
        nonlocal q_vec_cache, q_dims_cache
        if q_vec_cache is None:
            q_blob, q_dims = embed(cfg, question, backend=backend, mode="query")

            if q_dims != ret.dims:
                raise RuntimeError(f"query embedding dims mismatch: query={q_dims} retriever={ret.dims}")

            q_vec_cache = normalize_query_vec(q_blob, q_dims)
            q_dims_cache = q_dims
            return q_vec_cache

        if q_dims_cache != ret.dims:
            raise RuntimeError(f"cached query embedding dims mismatch: query={q_dims_cache} retriever={ret.dims}")

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

    def search_hyde(k: int) -> list[tuple[int, float]]:
        nonlocal hyde_document_cache
        hyde_cfg = cfg.get("hyde", {}) or {}
        if not as_bool(hyde_cfg.get("enabled", False)):
            return []
        
        faiss_ret = get_faiss_ret()
        if hyde_document_cache is None:
            hyde_document_cache = generate_hyde_document(cfg, question)
        
        if not hyde_document_cache:
            return []
        
        hyde_blob, hyde_dims = embed(cfg, hyde_document_cache, backend=backend, mode="query")
        if hyde_dims != faiss_ret.dims:
            raise RuntimeError("HyDE embedding dimension mismatch: " f"query={hyde_dims} faiss={faiss_ret.dims}")
        
        hyde_vec = normalize_query_vec(hyde_blob, hyde_dims)
        configured_k = int(hyde_cfg.get("candidate_k", k))
        effective_k = min(k, configured_k)
        results = faiss_ret.search(hyde_vec, effective_k)
        log.info("[HYDE] retrieved candidates=%d requested_k=%d", len(results), effective_k)
        return results

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
        nonlocal strict_fts_query_used
        strict_query = make_intent_fts5_query(question, intent)
        strict_fts_query_used = strict_query
        bm25_ret = get_bm25_ret()
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
    
    def search_hybrid_all(k: int):
        faiss_ret = get_faiss_ret()
        tv_ret = get_turbovec_ret()
        q_vec = get_q_vec(faiss_ret)

        if faiss_ret.dims != tv_ret.dims:
            raise RuntimeError(f"FAISS/TurboVec dimension mismatch: faiss={faiss_ret.dims} turbovec={tv_ret.dims}")

        faiss_results = faiss_ret.search(q_vec, k)
        tv_results = tv_ret.search(q_vec, k)
        bm25_results = search_bm25(k)

        signals = build_three_way_signal(
            faiss_results,
            tv_results,
            bm25_results,
            rrf_k=rrf_k,
            rrf_scale=float(retrieval_cfg.get("rrf_scale", 10.0)),
        )

        results = sorted(((cid, sig["rrf_score"]) for cid, sig in signals.items()), key=lambda x: x[1], reverse=True)
        return results, signals
    
    def should_trigger_hyde(faiss_results: list[tuple[int, float]], bm25_results: list[tuple[int, float]]) -> tuple[bool, str]:
        if not hyde_enabled:
            return False, "hyde_disabled"
        
        if hyde_mode == "always":
            return True, "mode_always"
        
        if hyde_mode in {"off", "disabled", "never"}:
            return False, "mode_disabled"
        
        if not faiss_results and not bm25_results:
            return True, "both_channels_empty"

        if not faiss_results:
            return True, "faiss_empty"

        if not bm25_results:
            return True, "bm25_empty"

        top_n = max(1, int(hyde_cfg.get("fallback_top_n", 12)))

        faiss_chunk_ids = [int(chunk_id) for chunk_id, _ in faiss_results[:top_n]]
        bm25_chunk_ids = [int(chunk_id) for chunk_id, _ in bm25_results[:top_n]]

        top_chunk_ids = list(
            dict.fromkeys(
                faiss_chunk_ids + bm25_chunk_ids
            )
        )

        rows = fetch_chunks(conn, top_chunk_ids)

        chunk_to_doc = {
            int(row["chunk_id"]): int(row["doc_id"])
            for row in rows
        }

        faiss_doc_ids = {
            chunk_to_doc[chunk_id]
            for chunk_id in faiss_chunk_ids
            if chunk_id in chunk_to_doc
        }

        bm25_doc_ids = {
            chunk_to_doc[chunk_id]
            for chunk_id in bm25_chunk_ids
            if chunk_id in chunk_to_doc
        }

        shared_doc_ids = (
            faiss_doc_ids & bm25_doc_ids
        )

        minimum_shared_docs = max(
            0,
            int(
                hyde_cfg.get(
                    "fallback_min_shared_docs",
                    1,
                )
            ),
        )

        if len(shared_doc_ids) < minimum_shared_docs:
            return (
                True,
                "low_channel_agreement:"
                f"shared_docs={len(shared_doc_ids)}"
                f"<{minimum_shared_docs}",
            )

        return (
            False,
            "normal_retrieval_sufficient:"
            f"shared_docs={len(shared_doc_ids)}",
        )

    
    def search_hybrid_hyde(k: int, *, force_hyde: bool = False, force_reason: str | None = None,) -> tuple[list[tuple[int, float]], dict[int, dict]]:
        nonlocal hyde_used_for_request
        nonlocal hyde_fallback_reason
        nonlocal hyde_error

        faiss_ret = get_faiss_ret()
        query_vec = get_q_vec(faiss_ret)
        faiss_results = faiss_ret.search(query_vec, k)
        bm25_results = search_bm25(k)
        use_hyde = False
        reason = "not_evaluated"

        if force_hyde and hyde_enabled:
            use_hyde = True
            reason = force_reason or "forced_fallback"
        else:
            use_hyde, reason = should_trigger_hyde(faiss_results, bm25_results)

        hyde_results: list[tuple[int, float]] = []

        if use_hyde:
            try:
                hyde_results = search_hyde(k)
                hyde_used_for_request = True
                hyde_fallback_reason = reason
            except Exception as exc:
                hyde_error = (f"{type(exc).__name__}: {exc}")
                hyde_fallback_reason = (f"{reason};hyde_failed")
                log.warning(
                    "[HYDE] fallback generation/search failed continuing with normal hybrid retrieval: %s", hyde_error)
        else:
            if hyde_fallback_reason is None:
                hyde_fallback_reason = reason

        signals = build_hybrid_hyde_signal(
            faiss_results,
            bm25_results,
            hyde_results,
            rrf_k=rrf_k,
            rrf_scale=float(retrieval_cfg.get("rrf_scale", 10.0)), 
            hyde_weight=float(hyde_cfg.get("rrf_weight", 0.75)))

        results = sorted(((chunk_id, signal["rrf_score"]) for chunk_id, signal in signals.items()), key=lambda item: item[1], reverse=True)
        log.info("[HYDE] mode=%s used=%s reason=%s normal_faiss=%d bm25=%d hyde=%d fused=%d", hyde_mode, bool(hyde_results), reason, len(faiss_results), len(bm25_results), len(hyde_results), len(results))
        return results, signals

    try:
        conn = read_only_connect(str(db_path))
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
            raw_results = search_embedding_retriever(retriever, candidate_k)
            results, retrieval_signals = build_single_channel_results(raw_results, "semantic")
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
            raw_results = search_embedding_retriever(retriever, candidate_k)
            results, retrieval_signals = build_single_channel_results(raw_results, "semantic")
        elif retriever_name == "hybrid_turbovec":
            log.info("[QNA] using HYBRID retriever (TurboVec + BM25)")
            retriever = None
            results, retrieval_signals = search_hybrid_turbovec(candidate_k)
        elif retriever_name in {"hybrid_all", "hybrid_faiss_turbovec"}:
            log.info("[QNA] using HYBRID retriever (FAISS + TurboVec + BM25)")
            retriever = None
            results, retrieval_signals = search_hybrid_all(candidate_k)
        elif retriever_name == "hybrid_hyde":
            log.info("[QNA] using HYBRID HYDE retriever (FAISS + BM25 + HyDE FAISS)")
            retriever = None
            results, retrieval_signals = (search_hybrid_hyde(candidate_k))
        else:
            raise RuntimeError(f"Unknown retriever: {retriever_name}")

        chunk_ids = [cid for cid, score in results]
        rank_scores = {cid: (float(sig.get("rrf_score", 0.0)) if isinstance(sig, dict) else float(sig)) for cid, sig in retrieval_signals.items()}
        initial_scores = rank_scores
        filtered_ids = filter_by_intent_source(conn, chunk_ids, intent, min_required=5, max_fallback=20)

        if intent in ("build", "mechanic", "lore", "biography", "location") and len(filtered_ids) < 3:
            deep_k = candidate_k * deep_candidate_multiplier
            log.info("[QNA] intent filter returned too few chunks; deep search k=%d", deep_k)

            if retriever_name == "faiss":
                raw_deep_results = search_embedding_retriever(retriever, deep_k)
                deep_results, retrieval_signals = build_single_channel_results(raw_deep_results, "semantic")
            elif retriever_name == "sqlite":
                raw_deep_results = search_embedding_retriever(retriever, deep_k)
                deep_results, retrieval_signals = build_single_channel_results(raw_deep_results, "semantic")
            elif retriever_name == "bm25":
                raw_deep_results = search_bm25(deep_k)
                deep_results, retrieval_signals = build_single_channel_results(raw_deep_results, "bm25")
            elif retriever_name == "hybrid":
                deep_results, retrieval_signals = search_hybrid(deep_k)
            elif retriever_name == "turbovec":
                raw_deep_results = search_embedding_retriever(retriever, deep_k)
                deep_results, retrieval_signals = build_single_channel_results(raw_deep_results, "semantic")
            elif retriever_name == "hybrid_turbovec":
                deep_results, retrieval_signals = search_hybrid_turbovec(deep_k)
            elif retriever_name in {"hybrid_all", "hybrid_faiss_turbovec"}:
                deep_results, retrieval_signals = search_hybrid_all(deep_k)
            elif retriever_name == "hybrid_hyde":
                deep_results, retrieval_signals = (search_hybrid_hyde(deep_k, force_hyde=True, force_reason=(f"intent_filter_too_few: count={len(filtered_ids)}")))
            else:
                raise RuntimeError(f"Unknown retriever: {retriever_name}")

            deep_ids = [cid for cid, _ in deep_results]
            filtered_ids = filter_by_intent_source(conn, deep_ids, intent, min_required=5, max_fallback=40)
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
        elif intent == "lookup":
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

        hyde_enabled = as_bool(hyde_cfg.get("enabled", False))
        hyde_signal_count = sum(1 for signal in (retrieval_signals or {}).values() if isinstance(signal, dict) and signal.get("in_hyde"))
        hyde_used = hyde_signal_count > 0

        if not chunks:
            return RetrievalResult(
                question=question,
                intent=intent,
                build_subtypes=set(build_subtypes),
                broad=broad,
                candidate_chunks=[],
                selected_chunks=[],
                context="",
                retrieval_signals=retrieval_signals,
                baseline_label=baseline_label,
                baseline_ord=baseline_ord,
                strict_fts_query=strict_fts_query_used,
                diagnostics={
                    "retriever": retriever_name,
                    "candidate_k": candidate_k,
                    "top_k": top_k,
                    "candidate_chunk_ids": [],
                    "selected_chunk_ids": [],
                    "hyde_enabled": hyde_enabled,
                    "hyde_used": hyde_used,
                    "hyde_fallback_reason": hyde_fallback_reason,
                    "hyde_error": hyde_error,
                    "hyde_candidate_count": hyde_signal_count,
                },
            )

        for row in chunks[:top_k]:
            log.info(
                "[QNA] chunk_id=%s title=%s source=%s preview=%s",
                row["chunk_id"],
                row["title"],
                row["source"],
                (row["text"][:200] if row["text"] else "").replace("\n", " "),
            )

        if intent == "lookup":
            lookup_entity = extract_lookup_entity(question)
            if (lookup_entity and not has_lookup_phrase(chunks, lookup_entity)):
                log.info(
                    "[LOOKUP] No reliable phrase match entity=%r",
                    lookup_entity,
                )
                chunks = []

        context_max_per_doc = (8 if intent == "lookup" else 4)
        candidate_chunks = [dict(row) for row in chunks]
        if broad:
            selected_chunks = candidate_chunks
        else:
            context_cfg = cfg.get("context_expansion", {}) or {}
            ctx_enabled = as_bool(context_cfg.get("enabled", False))
            selected_chunks = chunks[:direct_top_k]
            if intent in ("biography", "location", "lookup"):
                selected_chunks = prefer_entity_seed_chunks(question, selected_chunks, min_keep=3)
                selected_chunks = selected_chunks[:direct_top_k]

            parent_cfg = cfg.get("parent_child", {}) or {}
            parent_enabled = as_bool(parent_cfg.get("enabled", False))

            if parent_enabled:
                seed_chunks = list(selected_chunks)
                parent_max_total = max(
                    int(parent_cfg.get("max_total_chunks", 16)),
                    len(seed_chunks),
                )
                parent_chunks = fetch_parent_context_chunks(
                    conn,
                    seed_chunks,
                    max_parents=int(parent_cfg.get("max_parents", 8)),
                    max_total_chunks=parent_max_total,
                )
                selected_chunks = merge_context_preserving_seeds(
                    seed_chunks,
                    parent_chunks,
                    max_total=parent_max_total,
                    max_per_doc=context_max_per_doc,
                )

                log.info("[PARENT] expanded selected chunks to parent context chunks=%d", len(selected_chunks))

            if ctx_enabled:
                seed_chunks = list(selected_chunks)
                ctx_max_total = max(
                    int(
                        context_cfg.get(
                            "max_total_chunks_after_parent",
                            context_cfg.get("max_total_chunks", 20),
                        )
                    ),
                    len(seed_chunks),
                )
                expanded_chunks = expand_context_windows(
                    conn,
                    seed_chunks,
                    before=int(context_cfg.get("before", 1)),
                    after=int(context_cfg.get("after", 1)),
                    max_total_chunks=ctx_max_total,
                )
                selected_chunks = merge_context_preserving_seeds(
                    seed_chunks,
                    expanded_chunks,
                    max_total=ctx_max_total,
                    max_per_doc=context_max_per_doc,
                )

                log.info("[CTX_EXPAND] expanded context chunks=%d before=%s after=%s", len(selected_chunks), context_cfg.get("before", 1), context_cfg.get("after", 1))

        selected_chunks = [dict(row) for row in selected_chunks]
        answer_context_cfg = cfg.get("answer_context", {}) or {}
        if intent == "lookup":
            default_max_chunks = 6
            default_max_chars = 9000
        elif intent == "build":
            default_max_chunks = 8
            default_max_chars = 14000
        else:
            default_max_chunks = 8
            default_max_chars = 12000

        selected_chunks = trim_chunks_to_context_budget(selected_chunks, max_chunks=int(answer_context_cfg.get(f"{intent}_max_chunks", default_max_chunks)), max_chars=int(answer_context_cfg.get(f"{intent}_max_chars", default_max_chars)), max_chars_per_chunk=int(answer_context_cfg.get("max_chars_per_chunk", 2200)))
        hyde_selected_count = sum(1 for row in selected_chunks if (retrieval_signals.get(int(row["chunk_id"]), {}).get("in_hyde", False)))
        hyde_used = hyde_selected_count > 0
        context = build_context(selected_chunks)
        log.info("[CONTEXT] final chunk IDs=%s", [int(row["chunk_id"]) for row in selected_chunks])
        return RetrievalResult(
            question=question,
            intent=intent,
            build_subtypes=set(build_subtypes),
            broad=broad,
            candidate_chunks=candidate_chunks,
            selected_chunks=selected_chunks,
            context=context,
            retrieval_signals=retrieval_signals,
            baseline_label=baseline_label,
            baseline_ord=baseline_ord,
            strict_fts_query=strict_fts_query_used,
            diagnostics={
                "retriever": retriever_name,
                "candidate_k": candidate_k,
                "top_k": top_k,
                "candidate_chunk_ids": [int(row["chunk_id"]) for row in candidate_chunks],
                "selected_chunk_ids": [int(row["chunk_id"]) for row in selected_chunks],
                "hyde_enabled": hyde_enabled,
                "hyde_used": hyde_used,
                "hyde_candidate_count": hyde_signal_count,
                "hyde_selected_count": hyde_selected_count,
            })
    finally:
        conn.close()

def retrieve_question_context(cfg: dict, question: str, *, retriever_name: str = "hybrid", direct_top_k: int = 12, broad_top_k: int = 60, backend: str | None = None) -> RetrievalResult:
    intent = detect_intent(question)
    build_subtypes = detect_build_subtypes(question) if intent == "build" else set()
    cache = get_retrieval_cache(cfg)

    if cache is None:
        return retrieve_question_context_uncached(cfg, question, retriever_name=retriever_name, direct_top_k=direct_top_k, broad_top_k=broad_top_k, backend=backend)

    db_path = resolve_db_path(cfg)
    cache_cfg = cfg.get("retrieval_cache", {}) or {}
    cache_version = int(cache_cfg.get("version", 1))
    cache_key = make_retrieval_cache_key(question=question, retriever_name=retriever_name, backend=backend, direct_top_k=direct_top_k, intent=f"{intent}:v{cache_version}", subtypes=build_subtypes, db_path=db_path, index_meta={})

    cached = cache.get(cache_key)
    if cached is not None:
        log.info("[RETRIEVAL_CACHE] hit key=%s question=%r", cache_key[:12], question)
        return retrieval_result_from_cache(cached)

    log.info("[RETRIEVAL_CACHE] miss key=%s question=%r", cache_key[:12], question)
    result = retrieve_question_context_uncached(cfg, question, retriever_name=retriever_name, direct_top_k=direct_top_k, broad_top_k=broad_top_k, backend=backend)
    cache.set(cache_key, retrieval_result_to_cache(result))
    log.info("[RETRIEVAL_CACHE] stored key=%s chunks=%d context_chars=%d", cache_key[:12], len(result.selected_chunks), len(result.context))
    return result

def build_three_way_signal(faiss_results: list[tuple[int, float]], turbovec_results: list[tuple[int, float]], bm25_results: list[tuple[int, float]], *, rrf_k: int = 60, rrf_scale: float = 10.0) -> dict[int, dict]:
    signals: dict[int, dict] = {}

    def ensure(cid: int) -> dict:
        if cid not in signals:
            signals[cid] = {
                "rrf_score": 0.0,
                "faiss_score": 0.0,
                "turbovec_score": 0.0,
                "bm25_score": 0.0,
                "faiss_rank": None,
                "turbovec_rank": None,
                "bm25_rank": None,
                "in_faiss": False,
                "in_turbovec": False,
                "in_bm25": False,
            }
        return signals[cid]

    for rank, (cid, score) in enumerate(faiss_results, start=1):
        cid = int(cid)
        s = ensure(cid)
        s["faiss_score"] = float(score)
        s["faiss_rank"] = rank
        s["in_faiss"] = True
        s["rrf_score"] += 1.0 / (rrf_k + rank)

    for rank, (cid, score) in enumerate(turbovec_results, start=1):
        cid = int(cid)
        s = ensure(cid)
        s["turbovec_score"] = float(score)
        s["turbovec_rank"] = rank
        s["in_turbovec"] = True
        s["rrf_score"] += 1.0 / (rrf_k + rank)

    for rank, (cid, score) in enumerate(bm25_results, start=1):
        cid = int(cid)
        s = ensure(cid)
        s["bm25_score"] = float(score)
        s["bm25_rank"] = rank
        s["in_bm25"] = True
        s["rrf_score"] += 1.0 / (rrf_k + rank)

    for s in signals.values():
        s["rrf_score"] *= rrf_scale

    return signals

def merge_context_preserving_seeds(seed_chunks: list[dict], extra_chunks: list[dict], *, max_total: int, max_per_doc: int = 4) -> list[dict]:
    max_total = max(int(max_total), len(seed_chunks))
    output: list[dict] = []
    seen_chunk_ids: set[int] = set()
    doc_counts: dict[int, int] = {}

    for row in seed_chunks:
        chunk_id = int(row["chunk_id"])

        if chunk_id in seen_chunk_ids:
            continue

        output.append(row)
        seen_chunk_ids.add(chunk_id)

        doc_id = int(row["doc_id"])
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

    for row in extra_chunks:
        if len(output) >= max_total:
            break

        chunk_id = int(row["chunk_id"])

        if chunk_id in seen_chunk_ids:
            continue

        doc_id = int(row["doc_id"])

        if doc_counts.get(doc_id, 0) >= max_per_doc:
            continue

        output.append(row)
        seen_chunk_ids.add(chunk_id)
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

    return output

def trim_chunks_to_context_budget(chunks: list[dict], *, max_chunks: int, max_chars: int, max_chars_per_chunk: int) -> list[dict]:
    output: list[dict] = []
    total_chars = 0

    for row in chunks:
        if len(output) >= max_chunks:
            break

        text = str(row.get("text") or "").strip()
        if not text:
            continue

        remaining = max_chars - total_chars
        if remaining < 200:
            break

        allowed = min(len(text), max_chars_per_chunk, remaining)

        copied = dict(row)
        copied["text"] = text[:allowed]

        output.append(copied)
        total_chars += allowed

    return output

def build_grounded_answer_prompt(question: str, context: str, *, intent: str | None =  None, build_subtypes: set[str] | None = None, max_recommendations: int = 5) -> str:
    subtypes = set(build_subtypes or ())
    format_rules = ""

    if intent == "build" and "weapon" in subtypes:
        format_rules = f"""
This is a weapon recommendation question.

Answer format:
1. Give the top recommendation first.
2. Then list up to {max_recommendations} explicitly supported weapon options in ranked order.
3. For each weapon, explain why it is recommended using only evidence from the context.
4. Mention the relevant role, stat, passive, utility, or trade-off only when the context supports it.
5. If the context provides a ranking but no reason, state that it is ranked by the source and do not invent a reason.
6. Even if the question says "best weapon" in the singular, include supported alternatives after the top choice.
7. Do not infer weapon stats, passives, damage, Energy Recharge, Elemental Mastery, or role from prior knowledge.
8. Every explanation must be explicitly supported by the supplied context.
9. If the context contains only a ranked list, say: "Ranked #N by the source; the retrieved context does not provide a reason."
"""

    elif intent == "build" and "artifact" in subtypes:
        format_rules = f"""
This is an artifact recommendation question.

Answer format:
1. Give the top artifact set first.
2. Then list up to {max_recommendations} explicitly supported artifact options in ranked order.
3. For each option, explain its use case, set effect, role, or trade-off only when supported by the context.
4. Distinguish full sets from mixed 2-piece combinations when the context does so.
5. If the context provides only a ranking and no explanation, state that clearly instead of inventing a reason.
6. Even if the question says "best artifact set" in the singular, include supported alternatives after the top choice.
7. An artifact may be recommended only when the context explicitly associates that artifact with the requested character.
8. Do not recommend artifacts merely because they appear in a generic artifact mechanics page.
9. Preserve the ranking from the character's build section.
"""

    elif intent == "build" and "team" in subtypes:
        format_rules = f"""
This is a team recommendation question.

List up to {max_recommendations} supported team compositions in ranked order.
For each team, identify the members and briefly explain their roles and synergy using only the context.

Anser format:
1. List a team only when the context explicitly presents those characters together as one team, party, lineup, or team composition.
2. Do not construct a team by combining character names found in separate passages.
3. Do not infer synergy solely from isolated descriptions of individual characters.
4. If no explicit Zhongli team composition appears in the context, state that the retrieved context does not contain a supported team.
"""

    elif intent == "build" and "talent" in subtypes:
        format_rules = """
This is a talent-priority question.

Give the priority as an ordered sequence such as:
1. Elemental Skill
2. Elemental Burst
3. Normal Attack

Explain each priority only when the context provides enough evidence.
"""

    return f"""
You are a retrieval-grounded Genshin Impact assistant.

Answer the question using only the supplied context.

General rules:
- Do not invent unsupported facts.
- Treat headings, numbered rankings, bullet lists, tables, and item descriptions as explicit evidence.
- Preserve the ranking order shown in the context.
- Before refusing, inspect all headings, lists, tables, and descriptions.
- Cite supporting chunk IDs where practical.
- If the context supports fewer than {max_recommendations} recommendations, list only those supported.
- If the context contains no answer, say that there is not enough evidence.

{format_rules}

Question:
{question}

Context:
{context}
""".strip()

def normalized_phrase(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

def has_lookup_phrase(chunks: list[dict], entity: str) -> bool:
    entity_key = normalized_phrase(entity)

    if not entity_key:
        return False

    for row in chunks:
        title_key = normalized_phrase(str(row.get("title") or ""))
        text_key = normalized_phrase(str(row.get("text") or "")[:2500])

        if entity_key in title_key:
            return True

        if entity_key in text_key:
            return True

    return False

def answer_question(cfg: dict, question: str, *, retriever_name: str = "hybrid", direct_top_k: int = 12, broad_top_k: int = 60, summarize_batch_size: int = 8, backend: str | None = None) -> str:
    intent = detect_intent(question)
    build_subtypes = detect_build_subtypes(question) if intent == "build" else set()
    cache = get_retrieval_cache(cfg)
    cache_key = None

    if cache is not None:
        db_path = resolve_db_path(cfg)
        cache_cfg = cfg.get("retrieval_cache", {}) or {}
        cache_version = int(cache_cfg.get("version", 1))
        cache_key = make_retrieval_cache_key(question=question, retriever_name=retriever_name, backend=backend, direct_top_k=direct_top_k, intent=f"{intent}:v{cache_version}", subtypes=build_subtypes, db_path=db_path, index_meta={})
        cached = cache.get(cache_key)
        if cached is not None:
            log.info("[RETRIEVAL_CACHE] hit key=%s question=%r", cache_key[:12], question)
            result = retrieval_result_from_cache(cached)
        else:
            log.info("[RETRIEVAL_CACHE] miss key=%s question=%r", cache_key[:12], question)
            result = retrieve_question_context(cfg, question, retriever_name=retriever_name, direct_top_k=direct_top_k, broad_top_k=broad_top_k, backend=backend)
            cache.set(cache_key, retrieval_result_to_cache(result))
            log.info("[RETRIEVAL_CACHE] stored key=%s chunks=%d context_chars=%d", cache_key[:12], len(result.selected_chunks), len(result.context))
    else:
        result = retrieve_question_context(cfg, question, retriever_name=retriever_name, direct_top_k=direct_top_k, broad_top_k=broad_top_k, backend=backend)

    if not result.selected_chunks:
        return "I couldn't retrieve any relevant chunks from the knowledge base."

    if result.broad:
        runtime = cfg.get("runtime", {}) or {}
        provider = str(runtime.get("qa_provider", "ollama")).strip().lower()
        if provider == "llamacpp":
            qa_timeout = int(cfg.get("llamacpp", {}).get("timeout", 300))
        else:
            qa_timeout = int(cfg.get("ollama", {}).get("timeout", 1800))

        notes = []
        for group in chunk_batch(result.candidate_chunks, summarize_batch_size):
            notes.append(summarize_chunk_group(cfg, question, group))
        return synthesize_final_answer(cfg, question, notes, qa_timeout)
    
    answer_style_cfg = cfg.get("answer_style", {}) or {} 
    prompt = build_grounded_answer_prompt(question, result.context, intent=result.intent, build_subtypes=result.build_subtypes, max_recommendations=int(answer_style_cfg.get("max_build_recommendations", 5)))
    answer = generate(cfg, prompt).strip()

    if len(answer.split()) < 3:
        log.warning("[QNA] Generator returned suspiciously short output: %r", answer)

        if result.intent == "lookup":
            return ("I couldn't produce a reliable answer from the retrieved context.")

    return answer