from __future__ import annotations
import math
from statistics import mean, median
from typing import Any

def rank(doc_ids: list[int], positive_doc_id: int) -> int | None:
    for rank, doc_id in enumerate(doc_ids, start=1):
        if int(doc_id) == int(positive_doc_id):
            return rank
    return None

def recall(ranks: list[int | None], k: int) -> float:
    return sum(rank is not None and rank <= k for rank in ranks) / len(ranks) if ranks else 0.0

def mrr(ranks:list[int | None], k:int)->float:
    return sum(1.0 / rank if rank is not None and rank <= k else 0.0 for rank in ranks) / len(ranks) if ranks else 0.0

def ndcg(ranks: list[int | None], k: int) -> float:
    return sum(1.0 / math.log2(rank+1) if rank is not None and rank<=k else 0.0 for rank in ranks) / len(ranks) if ranks else 0.0

def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) -1) * p
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))

    if lo == hi:
        return float(values[lo])

    return float(values[lo] + (values[hi] - values[lo]) * (pos - lo))

def rank_stats(ranks: list[int | None]) -> dict[str, float |  int | None]:
    found = [rank for rank in ranks if rank is not None]
    return {
        "found":  len(found),
        "missing":  len(ranks) - len(found),
        "mean":  mean(found) if found else None,
        "median":  median(found) if found else None,
        "p90":  percentile([float(x) for x in found], 0.90),
        "p95":  percentile([float(x) for x in found], 0.95),
    }

def make_retrieval_metric_record(retrieval: Any, positive_doc_id: int, *, latency_ms: float | None = None) -> dict[str, Any]:
    d = retrieval.diagnostics or {}
    if "metric_candidate_doc_ids" not in d:
        raise RuntimeError("Retrieval diagnostics missing metric_candidate_doc_ids; enable retrieval_metrics in engine")

    candidate=[int(x) for x in d.get("metric_candidate_doc_ids", [])]
    reranked=[int(x) for x in d.get("metric_reranked_doc_ids", [])]
    context=[int(x) for x in d.get("metric_context_doc_ids", [])]
    return {
        "positive_doc_id": int(positive_doc_id),
        "candidate_rank": rank(candidate, positive_doc_id),
        "reranked_rank": rank(reranked, positive_doc_id),
        "context_rank": rank(context, positive_doc_id),
        "candidate_count": len(candidate),
        "reranked_count": len(reranked),
        "context_count": len(context),
        "intent": str(retrieval.intent),
        "broad": bool(retrieval.broad),
        "query_expansion_used": bool(d.get("query_expansion_used", False)),
        "query_decomposition_used": bool(d.get("query_decomposition_used", False)),
        "multi_hop_used": bool(d.get("multi_hop_used", False)),
        "hyde_used": bool(d.get("hyde_used", False)),
        "latency_ms": latency_ms,
    }

class RetrievalMetrics:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.errors = 0

    def __len__(self) -> int:
        return len(self.rows)

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def extend(self, rows: list[dict[str, Any]]) -> None:
        self.rows.extend(rows)

    def add_error(self, count: int = 1) -> None:
        self.errors += count

    def summary(self) -> dict[str, Any]:
        rows = self.rows
        candidate = [row["candidate_rank"] for row in rows]
        reranked = [row["reranked_rank"] for row in rows]
        context = [row["context_rank"] for row in rows]
        latency = [float(row["latency_ms"]) for row in rows if row.get("latency_ms")]
        candidate_hits = sum(rank is not None for rank in candidate)
        reranked_hits = sum(rank is not None for rank in reranked)
        context_hits = sum(rank is not None for rank in context)
        return {
            "queries": len(rows),
            "errors": self.errors,
            "candidate_recall@10": recall(candidate,10),
            "candidate_recall@32": recall(candidate,32),
            "candidate_recall@100": recall(candidate,100),
            "candidate_recall@300": recall(candidate,300),
            "candidate_recall@950": recall(candidate,950),
            "candidate_mrr@10": mrr(candidate,10),
            "candidate_ndcg@10": ndcg(candidate,10),
            "candidate_rank": rank_stats(candidate),
            "selected_recall@1": recall(reranked,1),
            "selected_recall@3": recall(reranked,3),
            "selected_recall@5": recall(reranked,5),
            "selected_recall@8": recall(reranked,8),
            "selected_recall@10": recall(reranked,10),
            "selected_recall@32": recall(reranked,32),
            "mrr@10": mrr(reranked,10),
            "ndcg@10": ndcg(reranked,10),
            "reranked_rank": rank_stats(reranked),
            "contextrecall": context_hits / len(rows) if rows else 0.0,
            "candidate_misses": len(rows) - candidate_hits,
            "ranking_losses": sum(c is not None and r is None for c, r in zip(candidate, reranked)),
            "context_losses": sum(r is not None and c is None for r, c in zip(reranked, context)),
            "candidate_docs_mean": mean(row["candidate_count"] for row in rows) if rows else 0.0,
            "reranked_docs_mean": mean(row["reranked_count"] for row in rows) if rows else 0.0,
            "context_docs_mean": mean(row["context_count"] for row in rows) if rows else 0.0,
            "expansion_usage": sum(row["query_expansion_used"] for row in rows) / len(rows) if rows else 0.0,
            "decomposition_usage": sum(row["query_decomposition_used"] for row in rows) / len(rows) if rows else 0.0,
            "multi_hop_usage": sum(row["multi_hop_used"] for row in rows) / len(rows) if rows else 0.0,
            "hyde_usage": sum(row["hyde_used"] for row in rows) / len(rows) if rows else 0.0,
            "latency_mean_ms": mean(latency) if latency else None,
            "latency_p50_ms": percentile(latency, 0.50),
            "latency_p95_ms": percentile(latency, 0.95),
        }

    def print(self)->None:
        s=self.summary()
        cr=s["candidate_rank"]
        rr=s["reranked_rank"]
        def f(value:Any,  digits:int=4) -> str:
            return "n/a" if value is None else f"{float(value):.{digits}f}"
        print("\n========== RETRIEVAL METRICS ==========")
        print(f"Queries measured        : {s['queries']}")
        print(f"Metric errors           : {s['errors']}")
        print("---------- Candidate retrieval ----------")
        print(f"Candidate Recall@10     : {f(s['candidate_recall@10'])}")
        print(f"Candidate Recall@32     : {f(s['candidate_recall@32'])}")
        print(f"Candidate Recall@100    : {f(s['candidate_recall@100'])}")
        print(f"Candidate Recall@300    : {f(s['candidate_recall@300'])}")
        print(f"Candidate Recall@950    : {f(s['candidate_recall@950'])}")
        print(f"Candidate MRR@10        : {f(s['candidate_mrr@10'])}")
        print(f"Candidate NDCG@10       : {f(s['candidate_ndcg@10'])}")
        print(f"Positive rank median    : {f(cr['median'],2)}")
        print(f"Positive rank mean      : {f(cr['mean'],2)}")
        print(f"Positive rank P90/P95   : {f(cr['p90'],2)} / {f(cr['p95'],2)}")
        print(f"Candidate misses        : {s['candidate_misses']}")
        print("---------- Final ranking ----------")
        print(f"Selected Recall@1       : {f(s['selected_recall@1'])}")
        print(f"Selected Recall@3       : {f(s['selected_recall@3'])}")
        print(f"Selected Recall@5       : {f(s['selected_recall@5'])}")
        print(f"Selected Recall@8       : {f(s['selected_recall@8'])}")
        print(f"Selected Recall@10      : {f(s['selected_recall@10'])}")
        print(f"Selected Recall@32      : {f(s['selected_recall@32'])}")
        print(f"MRR@10                  : {f(s['mrr@10'])}")
        print(f"NDCG@10                 : {f(s['ndcg@10'])}")
        print(f"Positive rank median    : {f(rr['median'],2)}")
        print(f"Positive rank mean      : {f(rr['mean'],2)}")
        print(f"Positive rank P90/P95   : {f(rr['p90'],2)} / {f(rr['p95'],2)}")
        print("---------- Context ----------")
        print(f"Context Recall          : {f(s['context_recall'])}")
        print(f"Candidate -> rank losses: {s['ranking_losses']}")
        print(f"Rank -> context losses  : {s['context_losses']}")
        print("---------- Pipeline ----------")
        print(f"Avg candidate docs      : {f(s['candidate_docs_mean'],1)}")
        print(f"Avg reranked docs       : {f(s['reranked_docs_mean'],1)}")
        print(f"Avg context docs        : {f(s['context_docs_mean'],1)}")
        print(f"Expansion usage         : {f(s['expansion_usage'])}")
        print(f"Decomposition usage     : {f(s['decomposition_usage'])}")
        print(f"Multi-hop usage         : {f(s['multi_hop_usage'])}")
        print(f"HyDE usage              : {f(s['hyde_usage'])}")
        if s["latency_mean_ms"] is not None:
            print(f"Retrieval latency mean  : {f(s['latency_mean_ms'],1)} ms")
            print(f"Retrieval latency P50   : {f(s['latency_p50_ms'],1)} ms")
            print(f"Retrieval latency P95   : {f(s['latency_p95_ms'],1)} ms")
        print("=======================================\n")