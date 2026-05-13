#!/usr/bin/env python3
"""Offline claim clustering and applicability review using the scored evidence pool."""

import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path
import os
import sys
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.epistemic.applicability_scorer import ApplicabilityScorer
from src.models.schemas import PICO
from src.nodes.contrastive_retrieval import contrastive_retrieval_node
from src.nodes.epistemic_scoring import epistemic_scoring_node
from src.nodes.pico_extraction import pico_extraction_node
from src.nodes.claim_clustering import claim_clustering_node

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEST_QUESTION = """
A 21-year-old sexually active male with recent travel to Vietnam and Cambodia presents with fatigue,
 worsening headache, malaise, arthralgia (pain in hands and wrists), fever, maculopapular rash,
 and laboratory findings including leukopenia and thrombocytopenia.
Options:
A) Chikungunya
B) Dengue fever
C) Epstein-Barr virus
D) Hepatitis A
"""


def build_population_profile(claims):
    population_sentences = []
    for claim in claims:
        if claim.get("population_claim"):
            text = claim.get("text", "").strip()
            if text:
                population_sentences.append(text)
    if not population_sentences:
        return ""
    return " ; ".join(population_sentences[:10])


def token_overlap_score(left_text, right_text):
    left_tokens = set(re.findall(r"[a-z]{3,}", (left_text or "").lower()))
    right_tokens = set(re.findall(r"[a-z]{3,}", (right_text or "").lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def select_profile_claims(claims, patient_text):
    if not claims:
        return []
    scored = []
    for claim in claims:
        text = claim.get("text", "")
        overlap = token_overlap_score(patient_text, text)
        scored.append((overlap, text))
    scored.sort(reverse=True, key=lambda item: item[0])
    selected = [text for score, text in scored if score > 0.0][:5]
    if selected:
        return selected
    return [claim.get("text", "") for claim in claims[:3] if claim.get("text")]


async def run_offline_review():
    state = {"original_question": TEST_QUESTION, "trace_events": []}

    pico_res = await pico_extraction_node(state)
    state.update(pico_res)

    retrieval_res = await contrastive_retrieval_node(state)
    state.update(retrieval_res)

    scoring_res = await epistemic_scoring_node(state)
    state.update(scoring_res)

    clustering_res = await claim_clustering_node(state)

    evidence_pool = scoring_res.get("evidence_pool", [])
    extracted_claims = clustering_res.get("extracted_claims", [])
    claim_clusters = clustering_res.get("claim_clusters", [])
    trace_events = state.get("trace_events", []) + clustering_res.get("trace_events", [])

    pico_obj = state.get("pico") or {}
    if isinstance(pico_obj, dict):
        patient_pico = PICO(
            population=pico_obj.get("population", "unknown"),
            intervention=pico_obj.get("intervention", "unknown"),
            comparator=pico_obj.get("comparator", "standard care"),
            outcome=pico_obj.get("outcome", "unknown"),
            risk_level=pico_obj.get("risk_level", "unknown"),
        )
    else:
        patient_pico = pico_obj

    scorer = ApplicabilityScorer()
    claim_index_by_pmid = defaultdict(list)
    for claim in extracted_claims:
        claim_index_by_pmid[str(claim.get("pmid"))].append(claim)

    patient_text = f"{patient_pico.population} {patient_pico.intervention} {patient_pico.outcome}"

    revised_applicability = []
    for item in evidence_pool:
        pmid = str(item.get("pmid"))
        selected_claims = select_profile_claims(claim_index_by_pmid.get(pmid, []), patient_text)
        profile = build_population_profile(
            [{"text": claim_text, "population_claim": True} for claim_text in selected_claims]
        )
        if not profile:
            profile = item.get("abstract", "")[:2000]
        score = scorer.compute(
            patient_pico,
            study_abstract=item.get("abstract", ""),
            study_population_profile=profile,
        )
        revised_applicability.append({
            "pmid": pmid,
            "title": item.get("title", ""),
            "original_applicability": item.get("applicability_score", 0.0),
            "revised_applicability": score,
            "population_profile": profile,
        })

    revised_scores = [row["revised_applicability"] for row in revised_applicability]
    mean_revised = sum(revised_scores) / len(revised_scores) if revised_scores else 0.0

    logger.info("=== OFFLINE CLUSTERING REVIEW ===")
    logger.info("Passed evidence items: %d", len(evidence_pool))
    logger.info("Extracted claims: %d", len(extracted_claims))
    logger.info("Clusters formed: %d", len(claim_clusters))
    logger.info("Revised applicability mean: %.3f", mean_revised)

    logger.info("=== SAMPLE CLAIMS ===")
    for claim in extracted_claims[:5]:
        logger.info(
            "PMID %s | population=%s | %s",
            claim.get("pmid"),
            claim.get("population_claim"),
            claim.get("text", "")[:180],
        )

    for keyword in ("Chikungunya", "Dengue"):
        matches = [row for row in revised_applicability if keyword.lower() in row["title"].lower()]
        if matches:
            logger.info("=== %s REVIEWS ===", keyword)
            for row in matches[:3]:
                logger.info(
                    "PMID %s | original=%.3f | revised=%.3f",
                    row["pmid"],
                    row["original_applicability"],
                    row["revised_applicability"],
                )

    logger.info("=== SAMPLE CLUSTERS ===")
    for cluster in claim_clusters[:5]:
        logger.info(
            "Cluster %s | size=%s | PMIDs=%s | representative=%s",
            cluster.get("cluster_id"),
            cluster.get("size"),
            cluster.get("source_pmids"),
            cluster.get("representative_claim", "")[:160],
        )

    output = {
        "evidence_pool_size": len(evidence_pool),
        "total_claims": len(extracted_claims),
        "cluster_count": len(claim_clusters),
        "mean_revised_applicability": round(mean_revised, 3),
        "revised_applicability": revised_applicability,
        "sample_claims": extracted_claims[:5],
        "sample_clusters": claim_clusters[:5],
        "trace_events": trace_events,
    }

    output_path = Path("results/offline_clustering_review.json")
    output_path.parent.mkdir(exist_ok=True)

    def _json_default(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)

    output_path.write_text(json.dumps(output, indent=2, default=_json_default), encoding="utf-8")
    logger.info("Saved review output to %s", output_path)


if __name__ == "__main__":
    asyncio.run(run_offline_review())
