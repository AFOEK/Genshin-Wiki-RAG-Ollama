from __future__ import annotations

import json
import logging
import re
from typing import Any

from .generators import generate

log = logging.getLogger(__name__)

def extract_json_object(text: str) -> dict[str, Any]:
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*","",text,flags=re.IGNORECASE)
    text = re.sub(r"\s*```$","",text).strip()
    start, end=text.find("{"),text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found")
    data = json.loads(text[start:end+1])
    if not isinstance(data,dict):
        raise ValueError("Expected JSON object")
    return data

def validate_answer_claims(cfg: dict[str, Any],question: str,answer: str,context: str) -> dict[str, Any]:
    vcfg = cfg.get("unsupported_claim_detection", {}) or {}
    if not answer.strip():
        return {"passed":False,"claims":[],"unsafe_claims":[],"reason":"empty_answer"}
    prompt=f"""You are a strict grounding verifier for a Genshin Impact RAG system.
Use ONLY the supplied context. Do not use prior knowledge.

Question:
{question}

Answer:
{answer}

Context:
{context}

Break the answer into atomic factual claims and judge each claim.

Rules:
- "supported": the context directly supports the claim.
- "unsupported": the context does not contain enough evidence, contradicts it, or the answer makes a stronger claim than the evidence.
- "uncertain": support is ambiguous.
- Exact numbers, dates, names, ranks, relationships, versions, stats, materials, costs, locations, and causal claims require explicit support.
- Do not treat two separate facts as proof of a stronger combined fact.
- Ignore purely stylistic or transitional text.
- Return JSON only.

Format:
{{
  "claims":[
    {{"claim":"...","verdict":"supported","confidence":0.99,"reason":"..."}}
  ]
}}"""
    model=str(vcfg.get("model") or "").strip() or None
    raw=generate(cfg,prompt,model_override=model,think_override=False,options_override={"temperature":float(vcfg.get("temperature",0.0)),"top_p":float(vcfg.get("top_p",0.8)),"top_k":int(vcfg.get("top_k",20)),"num_predict":int(vcfg.get("num_predict",768))})
    try:
        data=extract_json_object(raw)
    except (ValueError,json.JSONDecodeError) as exc:
        log.warning("[CLAIM_CHECK] invalid JSON err=%s raw=%r",exc,str(raw)[:500])
        return {"passed":False,"claims":[],"unsafe_claims":[],"reason":"validator_invalid_json"}

    claims=data.get("claims",[])
    if not isinstance(claims,list):
        return {"passed":False,"claims":[],"unsafe_claims":[],"reason":"validator_invalid_claims"}

    min_conf=float(vcfg.get("min_confidence",0.85))
    normalized=[]
    for row in claims:
        if not isinstance(row,dict):
            continue

        claim = str(row.get("claim") or "").strip()
        if not claim:
            continue
        
        verdict = str(row.get("verdict") or "uncertain").strip().lower()
        try:
            confidence = max(0.0, min(1.0, float(row.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0

        if verdict not in {"supported","unsupported","uncertain"}:
            verdict = "uncertain"

        normalized.append({
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "reason": str(row.get("reason") or "").strip()
        })

    if not normalized and answer.strip():
        return {
            "passed":False,
            "claims":[],
            "unsafe_claims":[],
            "reason":"validator_returned_no_claims",
        }

    unsafe = [row for row in normalized if row["verdict"] != "supported" or row["confidence"] < min_conf]
    return {
        "passed":not unsafe,
        "claims":normalized,
        "unsafe_claims":unsafe,
        "reason":"ok" if not unsafe else "unsupported_or_uncertain_claims",
    }

def build_broad_validation_context(chunks:list[dict],max_chunks:int=30,max_chars:int=30000,max_chars_per_chunk:int=1500)->str:
    parts=[]
    total=0
    for row in chunks[:max_chunks]:
        text=str(row.get("text") or "").strip()[:max_chars_per_chunk]
        if not text:
            continue
        block=f"[Source: {row.get('source')} | Title: {row.get('title')} | Chunk: {row.get('chunk_id')}]\n{text}"
        if total+len(block)>max_chars:
            break
        parts.append(block)
        total+=len(block)
    return "\n\n".join(parts)

def repair_answer_claims(cfg: dict[str, Any],question: str,answer: str,context: str,report: dict[str, Any]) -> str:
    vcfg=cfg.get("unsupported_claim_detection",{}) or {}
    unsafe=report.get("unsafe_claims",[])
    if not unsafe:
        return answer
    unsafe_text="\n".join(f"- {row.get('claim','')}: {row.get('reason','')}" for row in unsafe)
    prompt=f"""Rewrite the answer so every factual claim is directly supported by the supplied context.

Question:
{question}

Original answer:
{answer}

Claims that failed grounding:
{unsafe_text}

Context:
{context}

Rules:
- Remove unsupported claims.
- Correct contradicted claims only when the context provides the correction.
- Preserve supported information.
- Do not add new facts.
- Do not use prior knowledge.
- Keep the answer concise and directly answer the question.
- If the context cannot support part of the requested answer, omit that part or explicitly say the retrieved context does not provide it.
- Return only the corrected answer."""
    model=str(vcfg.get("repair_model") or vcfg.get("model") or "").strip() or None
    return str(generate(cfg,prompt,model_override=model,think_override=False,options_override={"temperature":float(vcfg.get("repair_temperature",0.0)),"top_p":float(vcfg.get("top_p",0.8)),"top_k":int(vcfg.get("top_k",20)),"num_predict":int(vcfg.get("repair_num_predict",768))})).strip()

def enforce_claim_support(cfg: dict[str, Any],question: str,answer: str,context: str) -> tuple[str,dict[str, Any]]:
    vcfg=cfg.get("unsupported_claim_detection",{}) or {}
    if not bool(vcfg.get("enabled",False)):
        return answer,{"passed":True,"disabled":True}
    report=validate_answer_claims(cfg,question,answer,context)
    if report["passed"]:
        return answer,report
    action=str(vcfg.get("action","repair")).strip().lower()
    if action=="warn":
        return answer,report
    if action=="reject":
        return str(vcfg.get("fallback_message","I couldn't produce a sufficiently grounded answer from the retrieved context.")),report
    if action!="repair":
        raise ValueError(f"Unsupported unsupported_claim_detection.action: {action!r}")

    repaired=repair_answer_claims(cfg,question,answer,context,report)
    second=validate_answer_claims(cfg,question,repaired,context)
    second["initial_report"]=report
    if second["passed"]:
        log.info("[CLAIM_CHECK] repaired answer successfully unsafe_before=%d",len(report["unsafe_claims"]))
        return repaired,second

    log.warning("[CLAIM_CHECK] repair still contains unsafe claims count=%d",len(second.get("unsafe_claims",[])))
    if bool(vcfg.get("fail_closed",True)):
        return str(vcfg.get("fallback_message","I couldn't produce a sufficiently grounded answer from the retrieved context.")),second
    return repaired,second