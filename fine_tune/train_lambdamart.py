from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))

from qna.engine import retrieve_question_context
from qna.lambdamart import FEATURE_NAMES,extract_ltr_features

def load_jsonl(path:str|Path)->list[dict]:
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_training_data(cfg:dict,pairs:list[dict])->tuple[np.ndarray,np.ndarray,list[int]]:
    X=[]
    y=[]
    groups=[]

    retrieval_cfg=copy.deepcopy(cfg)
    retrieval_cfg.setdefault("lambdamart",{})["enabled"]=False
    retrieval_cfg.setdefault("retrieval_cache",{})["enabled"]=False

    ds_cfg=cfg.get("dataset_creation",{}) or {}
    retriever=str(ds_cfg.get("retriever","hybrid_all"))
    backend=str(ds_cfg.get("backend","ollama"))

    for i,pair in enumerate(pairs,1):
        question=str(pair["query"]).strip()
        candidates=[(pair["positive"],2)]
        candidates.extend((row,0) for row in pair.get("hard_negatives",[]))
        candidates.extend((row,0) for row in pair.get("easy_negatives",[]))

        if len(candidates)<2:
            continue

        retrieval=retrieve_question_context(retrieval_cfg,question,retriever_name=retriever,direct_top_k=75,backend=backend)
        signals=retrieval.retrieval_signals

        for row,label in candidates:
            signal=signals.get(int(row["chunk_id"]),{})
            X.append(extract_ltr_features(cfg,question,row,signal))
            y.append(label)

        groups.append(len(candidates))

        if i%100==0:
            print(f"[LTR] queries={i} rows={len(X)}")

    return np.asarray(X,dtype=np.float32),np.asarray(y,dtype=np.int32),groups

def train_ranker(cfg:dict,X:np.ndarray,y:np.ndarray,groups:list[int])->lgb.LGBMRanker:
    lcfg=cfg.get("lambdamart",{}) or {}
    model=lgb.LGBMRanker(
        objective=str(lcfg.get("objective","lambdarank")),
        metric=str(lcfg.get("metric","ndcg")),
        n_estimators=int(lcfg.get("n_estimators",500)),
        learning_rate=float(lcfg.get("learning_rate",0.03)),
        num_leaves=int(lcfg.get("num_leaves",31)),
        max_depth=int(lcfg.get("max_depth",-1)),
        min_child_samples=int(lcfg.get("min_child_samples",20)),
        feature_fraction=float(lcfg.get("feature_fraction",0.85)),
        bagging_fraction=float(lcfg.get("bagging_fraction",0.85)),
        bagging_freq=int(lcfg.get("bagging_freq",1)),
        random_state=int(lcfg.get("seed",1337)),
        n_jobs=-1,
    )
    model.fit(X,y,group=groups,eval_at=tuple(lcfg.get("eval_at",[5,10,20])))
    return model

def main()->None:
    with open("rag/config.yaml","r",encoding="utf-8") as f:
        cfg=yaml.safe_load(f)

    lcfg=cfg.get("lambdamart",{}) or {}
    pairs=load_jsonl(lcfg["training_pairs"])
    X,y,groups=build_training_data(cfg,pairs)

    print(f"[LTR] queries={len(groups)} rows={len(y)} features={X.shape[1]}")
    model=train_ranker(cfg,X,y,groups)

    out=Path(lcfg["model_path"])
    out.parent.mkdir(parents=True,exist_ok=True)
    model.booster_.save_model(str(out))

    for name,value in sorted(zip(FEATURE_NAMES,model.feature_importances_),key=lambda x:x[1],reverse=True):
        print(f"{name:24s} {value}")

if __name__=="__main__":
    main()