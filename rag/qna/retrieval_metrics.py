from __future__ import annotations

import math
import csv
import json

from datetime import datetime, timezone
from pathlib import Path
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
        "question":str(retrieval.question),
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

def flatten_metadata(data:dict[str,Any], prefix:str="meta")->dict[str,Any]:
        out={}
        for key,value in data.items():
            name=f"{prefix}_{key}"
            if isinstance(value,dict):
                out.update(flatten_metadata(value, name))
            elif isinstance(value,(list,tuple,set)):
                out[name]=json.dumps(list(value),ensure_ascii=False)
            elif isinstance(value,Path):
                out[name]=str(value)
            else:
                out[name]=value
        return out

def make_dense_metric_record(*, query_id: str, question: str, positive_chunk_id: int, positive_doc_id: int, retrieved_chunk_ids: list[int], retrieved_doc_ids: list[int], embedding_latency_ms: float, search_latency_ms: float) -> dict[str, Any]:
    if len(retrieved_chunk_ids) != len(retrieved_doc_ids):
        raise ValueError("Chunk/document result count mismatch")

    unique_docs = list(dict.fromkeys(retrieved_doc_ids))
    return {
        "query_id": query_id,
        "question": question,
        "positive_chunk_id": positive_chunk_id,
        "positive_doc_id": positive_doc_id,
        "chunk_rank": rank(retrieved_chunk_ids, positive_chunk_id,),
        "document_in_chunk_rank": rank(retrieved_doc_ids, positive_doc_id,),
        "unique_document_rank": rank(unique_docs, positive_doc_id,),
        "candidate_count": len(retrieved_chunk_ids),
        "unique_document_count": len(unique_docs),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_doc_ids": retrieved_doc_ids,
        "embedding_latency_ms": embedding_latency_ms,
        "search_latency_ms": search_latency_ms,
        "latency_ms": (embedding_latency_ms + search_latency_ms),
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
            "context_recall": context_hits / len(rows) if rows else 0.0,
            "candidate_misses": len(rows) - candidate_hits,
            "ranking_losses": sum(c is not None and r is None for c, r in zip(candidate, reranked)),
            "context_losses": sum(r is not None and c is None for r, c in zip(reranked, context)),
            "candidate_docs_mean": mean(row["candidate_count"] for row in rows) if rows else 0.0,
            "reranked_docs_mean": mean(row["reranked_count"] for row in rows) if rows else 0.0,
            "reranked_misses":len(rows)-reranked_hits,
            "context_misses":len(rows)-context_hits,
            "ranking_survival_rate":reranked_hits/candidate_hits if candidate_hits else 0.0,
            "context_survival_rate":context_hits/reranked_hits if reranked_hits else 0.0,
            "context_docs_mean": mean(row["context_count"] for row in rows) if rows else 0.0,
            "expansion_usage": sum(row["query_expansion_used"] for row in rows) / len(rows) if rows else 0.0,
            "decomposition_usage": sum(row["query_decomposition_used"] for row in rows) / len(rows) if rows else 0.0,
            "multi_hop_usage": sum(row["multi_hop_used"] for row in rows) / len(rows) if rows else 0.0,
            "hyde_usage": sum(row["hyde_used"] for row in rows) / len(rows) if rows else 0.0,
            "latency_mean_ms": mean(latency) if latency else None,
            "latency_p50_ms": percentile(latency, 0.50),
            "latency_p95_ms": percentile(latency, 0.95),
        }

    def save_jsonl(self,path:str|Path,*,run_id:str|None=None,metadata:dict[str,Any]|None=None)->None:
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        run_id=run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        metadata=dict(metadata or {})
        with path.open("w",encoding="utf-8") as f:
            for row in self.rows:
                output={
                    "run_id":run_id,
                    "metadata":metadata,
                    **row,
                }
                f.write(json.dumps(output,ensure_ascii=False,default=str)+"\n")

    def save_csv(self,path:str|Path,*,run_id:str|None=None,metadata:dict[str,Any]|None=None)->None:
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        if not self.rows:
            return
        run_id=run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        flat_metadata=flatten_metadata(metadata or {})
        rows=[
            {
                "run_id":run_id,
                **flat_metadata,
                **row,
            }
            for row in self.rows
        ]
        fieldnames=list(dict.fromkeys(key for row in rows for key in row.keys()))
        with path.open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fieldnames,extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def save_summary(self,path:str|Path,*,run_id:str|None=None,metadata:dict[str,Any]|None=None)->None:
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        run_id=run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output={
            "run_id":run_id,
            "generated_at":datetime.now(timezone.utc).isoformat(),
            "metadata":dict(metadata or {}),
            "metrics":self.summary(),
        }
        path.write_text(
            json.dumps(output,ensure_ascii=False,indent=2,default=str),
            encoding="utf-8",
        )

    def save(self,output_dir:str|Path,*,run_id:str|None=None,metadata:dict[str,Any]|None=None)->None:
        output_dir=Path(output_dir)
        output_dir.mkdir(parents=True,exist_ok=True)
        run_id=run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.save_jsonl(
            output_dir/f"retrieval_metrics_{run_id}.jsonl",
            run_id=run_id,
            metadata=metadata,
        )
        self.save_csv(
            output_dir/f"retrieval_metrics_{run_id}.csv",
            run_id=run_id,
            metadata=metadata,
        )
        self.save_summary(
            output_dir/f"retrieval_metrics_summary_{run_id}.json",
            run_id=run_id,
            metadata=metadata,
        )

    def print(self)->None:
        s=self.summary()
        cr=s["candidate_rank"]
        rr=s["reranked_rank"]
        def f(value:Any,  digits:int=6) -> str:
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
        print(f"Reranked misses         : {s['reranked_misses']}")
        print(f"Context misses          : {s['context_misses']}")
        print(f"Ranking survival        : {f(s['ranking_survival_rate'])}")
        print(f"Context survival        : {f(s['context_survival_rate'])}")
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

class DenseRetrievalMetrics(RetrievalMetrics):
    def __init__(self, top_k: int = 100):
        super().__init__()
        self.top_k = int(top_k)
        if self.top_k < 1:
            raise ValueError("top_k must be positive")

    def summary(self) -> dict[str, Any]:
        rows = self.rows
        output = {
            "queries": len(rows),
            "errors": self.errors,
            "search_depth": self.top_k,
        }
        rank_fields = {
            "chunk": "chunk_rank",
            "document_in_chunk_ranking": "document_in_chunk_rank",
        }
        for prefix, field in rank_fields.items():
            ranks = [row[field] for row in rows]
            for k in [1, 3, 5, 10, 20, 50, 100, 300, 950]:
                if k > self.top_k:
                    continue
                output[f"{prefix}_recall@{k}"] = (recall(ranks, k))
                output[f"{prefix}_mrr@{k}"] = (mrr(ranks, k))
                output[f"{prefix}_ndcg@{k}"] = (ndcg(ranks, k))

            output[f"{prefix}_rank_statistics"] = (rank_stats(ranks))

        latency_fields = ["embedding_latency_ms", "search_latency_ms", "latency_ms",]
        for field in latency_fields:
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            output[f"{field}_mean"] = (mean(values) if values else None)
            for p in [0.50, 0.90, 0.95, 0.99]:
                output[f"{field}_p{int(p * 100)}"] = percentile(values, p)

        return output

    def print(self):
        print("\n===== DENSE RETRIEVAL METRICS =====")
        for key, value in self.summary().items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")