import logging
from typing import Dict, Any, List
import asyncio
import numpy as np
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.core.registry import ModelRegistry, safe_ainvoke
from src.models.state import GraphState
from src.models.schemas import EvidenceItem, PICO
from src.epistemic.applicability_scorer import ApplicabilityScorer
from collections import defaultdict

logger = logging.getLogger(__name__)

ATOMIC_CLAIM_EXTRACTION_PROMPT = """
You are a clinical research assistant. Extract all atomic claims from the provided abstract.
Each claim should be a single, declarative sentence about a specific finding or relationship.
Focus on: diagnosis, intervention, outcome, symptom, lab finding, or epidemiological claim.

Return ONLY a JSON array of claim objects. Each object must have:
- "claim_id": integer starting from 1
- "text": the full claim sentence
- "subject": the entity or subject (e.g., "Dengue", "Chikungunya", "fever")
- "predicate": the relationship (e.g., "presents_with", "is_associated_with", "causes")
- "object": the target entity (e.g., "arthralgia", "thrombocytopenia")
- "population_claim": true if the claim describes the study population, cohort, enrollment criteria, demographics, or setting; otherwise false

Example output:
[
    {{"claim_id": 1, "text": "Dengue presents with fever and rash", "subject": "Dengue", "predicate": "presents_with", "object": "fever and rash", "population_claim": false}},
    {{"claim_id": 2, "text": "The study included adults with acute dengue infection", "subject": "study population", "predicate": "included", "object": "adults with acute dengue infection", "population_claim": true}}
]

Abstract: {abstract}
"""

def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if v1 is None or v2 is None:
        return 0.0
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def greedy_cluster(claims_with_embeddings: List[Dict[str, Any]], threshold: float = 0.8) -> List[List[int]]:
    """
    Greedy clustering of claims based on embedding similarity.
    Returns list of clusters, where each cluster is a list of claim indices.
    """
    if not claims_with_embeddings:
        return []
    
    clusters = []
    assigned = set()
    
    for i, claim_i in enumerate(claims_with_embeddings):
        if i in assigned:
            continue
        
        cluster = [i]
        assigned.add(i)
        
        for j in range(i + 1, len(claims_with_embeddings)):
            if j in assigned:
                continue
            
            sim = _cosine_similarity(claim_i["embedding"], claims_with_embeddings[j]["embedding"])
            if sim >= threshold:
                cluster.append(j)
                assigned.add(j)
        
        clusters.append(cluster)
    
    return clusters

async def claim_clustering_node(state: GraphState) -> Dict[str, Any]:
    """
    Node to extract atomic claims and group them into semantic clusters.
    
    Process:
    1. Extract atomic claims from each abstract.
    2. Encode each claim with embeddings.
    3. Cluster claims using cosine similarity threshold.
    4. Return clusters with source PMIDs and original claims.
    """
    evidence_pool = state.get("evidence_pool", [])
    if not evidence_pool:
        return {"claim_clusters": [], "extracted_claims": []}

    # Reconstruct EvidenceItem objects if needed
    items = []
    for raw in evidence_pool:
        if isinstance(raw, dict):
            try:
                item = EvidenceItem(**raw)
            except Exception:
                item = raw
        else:
            item = raw
        items.append(item)

    # 1. Extract atomic claims from each abstract
    llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)
    prompt = ChatPromptTemplate.from_template(ATOMIC_CLAIM_EXTRACTION_PROMPT)
    
    all_claims = []  # List of {pmid, claim_id, text, subject, predicate, object, embedding}
    
    for item in items:
        try:
            res = await safe_ainvoke(prompt | llm | JsonOutputParser(), {"abstract": item.abstract})
            
            if isinstance(res, dict) and "claim_id" in res:
                # LLM returned single claim instead of array
                res = [res]
            elif not isinstance(res, list):
                res = []
            
            for claim_obj in res:
                claim_text = claim_obj.get("text", "")
                population_flag = bool(claim_obj.get("population_claim", False))
                if not population_flag:
                    population_flag = bool(re.search(
                        r"\b(patient|patients|participant|participants|subject|subjects|cohort|enrolled|included|adults?|children|infants?|women|men|female|male|pregnant|outpatient|inpatient|emergency|travel|resident|population|setting|clinic|hospital|age)\b",
                        claim_text,
                        flags=re.IGNORECASE,
                    ))
                claim_dict = {
                    "pmid": item.pmid,
                    "claim_id": claim_obj.get("claim_id"),
                    "text": claim_text,
                    "subject": claim_obj.get("subject", ""),
                    "predicate": claim_obj.get("predicate", ""),
                    "object": claim_obj.get("object", ""),
                    "population_claim": population_flag,
                    "is_population_claim": population_flag,
                    "confidence": round(float(getattr(item, "reproducibility_score", 0.5) or 0.5), 3),
                    "rps": item.reproducibility_score,
                    "applicability": item.applicability_score
                }
                all_claims.append(claim_dict)
            
            logger.info(f"Extracted {len(res)} claims from PMID {item.pmid}")
        except Exception as e:
            logger.warning(f"Atomic claim extraction failed for PMID {item.pmid}: {e}")

    # 2. Encode claims with embeddings
    encoder = ModelRegistry.get_sentence_transformer("all-MiniLM-L6-v2")
    
    for claim in all_claims:
        try:
            emb = encoder.encode(claim["text"])
            claim["embedding"] = emb
        except Exception as e:
            logger.warning(f"Embedding failed for claim '{claim['text'][:50]}': {e}")
            claim["embedding"] = None

    # 3. Cluster claims using greedy similarity-based clustering
    claims_with_emb = [c for c in all_claims if c.get("embedding") is not None]
    
    if not claims_with_emb:
        logger.warning("No claims with valid embeddings; returning empty clusters")
        return {
            "claim_clusters": [],
            "extracted_claims": all_claims,
            "trace_events": [{"node": "claim_clustering", "error": "No valid embeddings"}]
        }

    cluster_indices = greedy_cluster(claims_with_emb, threshold=0.8)

    # 4. Build output clusters with metadata
    output_clusters = []
    for cluster_idx, indices in enumerate(cluster_indices):
        cluster_claims = [claims_with_emb[i] for i in indices]
        
        # Representative claim: the one with highest RPS
        rep_claim = max(cluster_claims, key=lambda c: c.get("rps", 0))
        
        cluster_dict = {
            "cluster_id": cluster_idx,
            "representative_claim": rep_claim["text"],
            "size": len(cluster_claims),
            "source_pmids": list(set(c["pmid"] for c in cluster_claims)),
            "claims": cluster_claims
        }
        output_clusters.append(cluster_dict)
        
        logger.info(f"Cluster {cluster_idx}: {len(cluster_claims)} claims from {len(cluster_dict['source_pmids'])} studies")

    # 5. Refinement pass: re-score applicability using extracted population profiles
    try:
        patient_pico = state.get("pico")
        if patient_pico and all_claims:
            # Convert evidence pool dicts back to list for refinement
            refined_evidence = await refine_applicability_with_claims(
                evidence_pool,
                all_claims,
                patient_pico
            )
            logger.info(f"Updated applicability scores via claim-based population profiling")
        else:
            refined_evidence = evidence_pool
            logger.info("Skipping applicability refinement: missing patient_pico or claims")
    except Exception as e:
        logger.warning(f"Applicability refinement failed: {e}; using original evidence pool")
        refined_evidence = evidence_pool

    # 6. Trace event
    trace_event = {
        "node": "claim_clustering",
        "section": "claim_clustering",
        "input": {"total_studies": len(items)},
        "output": {
            "total_claims_extracted": len(all_claims),
            "population_claims": sum(1 for claim in all_claims if claim.get("population_claim")),
            "total_clusters": len(output_clusters),
            "avg_cluster_size": round(len(all_claims) / len(output_clusters), 1) if output_clusters else 0,
            "applicability_refinement": "applied" if patient_pico and all_claims else "skipped"
        }
    }

    return {
        "extracted_claims": all_claims,
        "claim_clusters": output_clusters,
        "evidence_pool": refined_evidence,
        "trace_events": [trace_event]
    }


def _select_profile_claims(claims: List[Dict[str, Any]], patient_text: str) -> List[str]:
    """Select highest-overlap claims for population profile construction."""
    if not claims:
        return []
    scored = []
    for claim in claims:
        text = claim.get("text", "")
        left_tokens = set(re.findall(r"[a-z]{3,}", (patient_text or "").lower()))
        right_tokens = set(re.findall(r"[a-z]{3,}", (text or "").lower()))
        if left_tokens and right_tokens:
            overlap = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
        else:
            overlap = 0.0
        scored.append((overlap, text))
    scored.sort(reverse=True, key=lambda item: item[0])
    selected = [text for score, text in scored if score > 0.0][:5]
    return selected if selected else [claim.get("text", "") for claim in claims[:3] if claim.get("text")]


def _build_population_profile(claims_for_pmid: List[Dict[str, Any]]) -> str:
    """Build a population profile string from extracted population claims."""
    population_sentences = []
    for claim in claims_for_pmid:
        if claim.get("population_claim"):
            text = claim.get("text", "").strip()
            if text:
                population_sentences.append(text)
    if not population_sentences:
        return ""
    return " ; ".join(population_sentences[:5])


async def refine_applicability_with_claims(
    evidence_pool: List[Dict[str, Any]],
    extracted_claims: List[Dict[str, Any]],
    patient_pico: Any,  # Can be PICO or dict
) -> List[Dict[str, Any]]:
    """
    Refinement pass: re-score applicability using extracted population profiles.
    Called after claim extraction to leverage population-specific claims for better weighting.
    """
    # Convert dict to PICO if needed
    if isinstance(patient_pico, dict):
        try:
            patient_pico = PICO(**patient_pico)
        except Exception as e:
            logger.warning(f"Failed to convert dict to PICO: {e}; skipping refinement")
            return evidence_pool
    
    claim_index_by_pmid = defaultdict(list)
    for claim in extracted_claims:
        pmid_str = str(claim.get("pmid", ""))
        if pmid_str:
            claim_index_by_pmid[pmid_str].append(claim)

    scorer = ApplicabilityScorer()
    patient_text = f"{patient_pico.population} {patient_pico.intervention} {patient_pico.outcome}".strip()

    refined_pool = []
    for item in evidence_pool:
        # Support both dict-like evidence and EvidenceItem objects
        if hasattr(item, "pmid"):
            pmid = str(getattr(item, "pmid", ""))
            abstract_text = str(getattr(item, "abstract", ""))[:2000]
        else:
            pmid = str(item.get("pmid", ""))
            abstract_text = str(item.get("abstract", ""))[:2000]

        claims_for_pmid = claim_index_by_pmid.get(pmid, [])

        # Build population profile from extracted claims
        if claims_for_pmid:
            profile = _build_population_profile(claims_for_pmid)
        else:
            profile = ""

        # If no population profile extracted, fall back to abstract
        if not profile:
            profile = abstract_text

        # Re-compute applicability with the population profile
        refined_score = scorer.compute(
            patient_pico,
            study_abstract=abstract_text,
            study_population_profile=profile,
        )

        # Write back refined score to either object or dict
        try:
            if hasattr(item, "applicability_score"):
                setattr(item, "applicability_score", float(refined_score))
            else:
                item["applicability_score"] = float(refined_score)
        except Exception:
            try:
                item["applicability_score"] = float(refined_score)
            except Exception:
                pass

        refined_pool.append(item)
    
    logger.info(f"Refined applicability scores for {len(refined_pool)} evidence items using extracted population claims")
    return refined_pool
