from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

FEATURE_NAMES=(
    "rrf_score","manual_score",
    "faiss_score","bm25_score","splade_score","turbovec_score","hyde_score",
    "faiss_rr","bm25_rr","splade_rr","turbovec_rr","hyde_rr",
    "in_faiss","in_bm25","in_splade","in_turbovec","in_hyde",
    "query_hits","query_best_rr",
    "title_overlap","text_overlap","exact_title_match",
    "source_weight","chunk_log_len",
)

def _f(value:Any,default:float=0.0)->float:
    try:
        value=float(value)
        return value if math.isfinite(value) else default
    except (TypeError,ValueError):
        return default

def _rr(rank:Any)->float:
    try:
        rank=int(rank)
        return 1.0/(60.0+rank) if rank>0 else 0.0
    except (TypeError,ValueError):
        return 0.0

def _terms(text:str)->set[str]:
    return set(re.findall(r"[a-z0-9]+",str(text).casefold()))

def _overlap(query:str,text:str)->float:
    q=_terms(query)
    if not q:
        return 0.0
    return len(q&_terms(text))/len(q)

def source_weight(cfg:dict,source:str)->float:
    for row in cfg.get("sources",[]) or []:
        if str(row.get("name") or "")==source:
            return _f(row.get("weight"),1.0)
    return 1.0

def extract_ltr_features(cfg:dict,question:str,row:dict,signal:dict|None=None)->list[float]:
    signal=signal or {}
    title=str(row.get("title") or "")
    text=str(row.get("text") or "")
    qkey=" ".join(question.casefold().split())
    tkey=" ".join(title.casefold().split())
    return [
        _f(signal.get("rrf_score")),_f(row.get("_rerank_score")),
        _f(signal.get("faiss_score")),_f(signal.get("bm25_score")),_f(signal.get("splade_score")),_f(signal.get("turbovec_score")),_f(signal.get("hyde_score")),
        _rr(signal.get("faiss_rank")),_rr(signal.get("bm25_rank")),_rr(signal.get("splade_rank")),_rr(signal.get("turbovec_rank")),_rr(signal.get("hyde_rank")),
        float(bool(signal.get("in_faiss"))),float(bool(signal.get("in_bm25"))),float(bool(signal.get("in_splade"))),float(bool(signal.get("in_turbovec"))),float(bool(signal.get("in_hyde"))),
        _f(signal.get("query_hits",signal.get("decomposition_hits",0))),_rr(signal.get("query_best_rank",signal.get("decomposition_best_rank"))),
        _overlap(question,title),_overlap(question,text),float(bool(qkey and qkey==tkey)),
        source_weight(cfg,str(row.get("source") or "")),math.log1p(len(text)),
    ]

class LambdaMARTRanker:
    def __init__(self,path:str|Path):
        self.path=Path(path)
        self.model=lgb.Booster(model_file=str(self.path))

    def rerank(self,cfg:dict,question:str,chunks:list[dict],signals:dict[int,dict])->list[dict]:
        if not chunks:
            return chunks
        matrix=np.asarray([extract_ltr_features(cfg,question,row,signals.get(int(row["chunk_id"]),{})) for row in chunks],dtype=np.float32)
        scores=self.model.predict(matrix)
        for row,score in zip(chunks,scores):
            row["_ltr_score"]=float(score)
        return sorted(chunks,key=lambda row:row["_ltr_score"],reverse=True)