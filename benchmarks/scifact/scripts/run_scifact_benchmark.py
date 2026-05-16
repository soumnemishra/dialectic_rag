import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.graph.workflow import build_workflow
from src.models.state import GraphState

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CLAIMS_PATH = PROJECT_ROOT / "benchmarks" / "scifact" / "data" / "claims_dev.jsonl"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "scifact"
    / "predictions"
    / "predictions_dev.jsonl"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Load SciFact claims
# ---------------------------------------------------------------------
def load_claims(path: Path, limit: int = None) -> List[Dict]:
    claims = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            claims.append(json.loads(line))
            if limit and len(claims) >= limit:
                break
    return claims


# ---------------------------------------------------------------------
# Map DIALECTIC-RAG verdicts to SciFact labels
# ---------------------------------------------------------------------
def map_label(verdict: str) -> str:
    if not verdict:
        return "NOT_ENOUGH_INFO"

    verdict = verdict.lower()

    mapping = {
        "support": "SUPPORT",
        "supported": "SUPPORT",
        "refute": "CONTRADICT",
        "refuted": "CONTRADICT",
        "contradict": "CONTRADICT",
        "uncertain": "NOT_ENOUGH_INFO",
        "insufficient": "NOT_ENOUGH_INFO",
        "not_enough_info": "NOT_ENOUGH_INFO",
    }

    return mapping.get(verdict, "NOT_ENOUGH_INFO")


# ---------------------------------------------------------------------
# DIALECTIC-RAG pipeline integration
# ---------------------------------------------------------------------
def _extract_verdict(result: Dict) -> str:
    evidence_pool = result.get("evidence_pool", [])
    support_score = 0
    oppose_score = 0

    for item in evidence_pool:
        stance = getattr(item, "stance", None)
        if stance is None and isinstance(item, dict):
            stance = item.get("stance")

        stance_text = str(stance).upper()
        if "SUPPORT" in stance_text:
            support_score += 1
        elif "OPPOSE" in stance_text or "CONTRADICT" in stance_text:
            oppose_score += 1

    if support_score > oppose_score:
        return "support"
    if oppose_score > support_score:
        return "refute"
    return "uncertain"


def _extract_evidence_doc_ids(result: Dict) -> List[str]:
    retrieved_docs = result.get("retrieved_docs", {})
    pmid_seen = set()
    evidence_doc_ids: List[str] = []

    for articles in retrieved_docs.values():
        for article in articles or []:
            if hasattr(article, "model_dump"):
                article = article.model_dump()
            pmid = article.get("pmid") if isinstance(article, dict) else None
            if pmid is None:
                continue
            pmid_str = str(pmid)
            if pmid_str not in pmid_seen:
                pmid_seen.add(pmid_str)
                evidence_doc_ids.append(pmid_str)

    return evidence_doc_ids


async def _run_dialectic_rag_async(claim: str) -> Dict:
    workflow = build_workflow()

    initial_state: GraphState = {
        "original_question": claim,
        "mcq_options": None,
        "intent": "informational",
        "risk_level": "low",
        "pico": None,
        "evidence_pool": [],
        "retrieved_docs": {},
        "step_notes": [],
        "claim_clusters": [],
        "temporal_result": None,
        "consensus_state": None,
        "epistemic_result": None,
        "abstention_rationale": None,
        "candidate_answers": [],
        "candidate_answer": "",
        "final_reasoning": "",
        "abstention_triggered": False,
        "extracted_claims": [],
        "candidate_stances": {},
        "fused_beliefs": {},
        "temporal_shift": {},
        "epistemic_state": None,
        "safety_flags": [],
        "trace_events": [],
        "trace_id": str(uuid.uuid4()),
        "trace_created_at": datetime.utcnow().isoformat(),
    }

    result = await workflow.ainvoke(initial_state)
    verdict = _extract_verdict(result)
    evidence_doc_ids = _extract_evidence_doc_ids(result)

    return {
        "verdict": verdict,
        "evidence_doc_ids": evidence_doc_ids,
        "evidence_sentences": {},
    }


def run_dialectic_rag(claim: str) -> Dict:
    try:
        return asyncio.run(_run_dialectic_rag_async(claim))
    except Exception:
        return {
            "verdict": "uncertain",
            "evidence_doc_ids": [],
            "evidence_sentences": {},
        }


# ---------------------------------------------------------------------
# Convert pipeline output to SciFact format
# ---------------------------------------------------------------------
def build_prediction(claim_id: int, result: Dict) -> Dict:
    label = map_label(result.get("verdict", "uncertain"))
    evidence_doc_ids = result.get("evidence_doc_ids", [])

    evidence = {}

    # Only SUPPORT and CONTRADICT may appear as predicted labels.
    if label in {"SUPPORT", "CONTRADICT"}:
        for doc_id in evidence_doc_ids[:5]:
            evidence[str(doc_id)] = {
                "label": label,
                "sentences": [],
            }

    return {
        "id": claim_id,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------
def main(limit: int = 10):
    claims = load_claims(CLAIMS_PATH, limit=limit)

    predictions = []

    for idx, item in enumerate(claims, start=1):
        claim_id = item["id"]
        claim_text = item["claim"]

        print(f"[{idx}/{len(claims)}] Claim {claim_id}: {claim_text[:80]}")

        result = run_dialectic_rag(claim_text)
        prediction = build_prediction(claim_id, result)

        predictions.append(prediction)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"\nSaved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main(limit=10)
