#!/usr/bin/env python3
"""Test script for claim clustering node using validated evidence pool from Step 3."""

import asyncio
import json
import logging
from pathlib import Path
from src.graph.workflow import build_workflow
from src.models.state import GraphState

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def test_claim_clustering():
    """Test claim clustering with single-question validated flow."""
    
    # Build workflow
    app = build_workflow()
    
    # Question from Step 3 validation
    test_question = """
    A 45-year-old male with a history of hypertension and diabetes mellitus presents 
    to the emergency department with severe left-sided chest pain that started 4 hours ago. 
    The pain is sharp, increases with breathing, and radiates to the left shoulder. 
    On examination, he has a respiratory rate of 22/min, heart rate of 98/min, and 
    blood pressure of 155/95 mmHg. Auscultation reveals reduced breath sounds on the 
    left side with dullness to percussion. Chest X-ray shows a large left pleural effusion 
    with mediastinal shift. Laboratory studies show elevated D-dimer (2.5 mcg/mL), 
    elevated troponin (0.15 ng/mL), and elevated N-terminal pro-BNP (450 pg/mL). 
    Electrocardiography shows sinus tachycardia with slight ST-segment elevation in leads II, III, and aVF.
    
    Options:
    A) Acute myocardial infarction with cardiogenic shock
    B) Acute pulmonary embolism with pleural infarction
    C) Acute aortic dissection with mediastinal hemorrhage
    D) Acute tension pneumothorax with acute coronary syndrome
    E) Acute pericarditis with constrictive effusion
    """
    
    candidates = ["A", "B", "C", "D", "E"]
    
    # Build initial input state for graph
    initial_state = GraphState(
        question=test_question,
        candidates=candidates,
        pico=None,
        candidate_answers=candidates,
        retrieved_docs={},
        evidence_pool=[],
        extracted_claims=[],
        claim_clusters=[],
        candidate_stances={},
        temporal_shift={},
        fused_beliefs={},
        epistemic_state="UNKNOWN",
        abstention_triggered=False,
        trace_events=[]
    )
    
    # Step 1: Run PICO extraction through claim clustering
    logger.info("=" * 80)
    logger.info("CLAIM CLUSTERING TEST: Running Steps 1-3 to build evidence pool")
    logger.info("=" * 80)
    
    try:
        input_state = {"question": test_question}
        
        # Run graph asynchronously
        result = await app.ainvoke(
            input_state,
            config={"recursion_limit": 25}
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("CLAIM CLUSTERING RESULTS")
        logger.info("=" * 80)
        
        # Extract claim clustering results
        extracted_claims = result.get("extracted_claims", [])
        claim_clusters = result.get("claim_clusters", [])
        evidence_pool = result.get("evidence_pool", [])
        trace_events = result.get("trace_events", [])
        
        # Summary metrics
        logger.info(f"\n✓ Evidence Pool Size: {len(evidence_pool)}")
        logger.info(f"✓ Total Claims Extracted: {len(extracted_claims)}")
        logger.info(f"✓ Clusters Formed: {len(claim_clusters)}")
        
        if claim_clusters:
            cluster_sizes = [c.get("size", 0) for c in claim_clusters]
            logger.info(f"✓ Cluster Sizes: {cluster_sizes}")
            logger.info(f"✓ Average Cluster Size: {sum(cluster_sizes)/len(cluster_sizes):.1f}")
        
        # Log sample claims
        logger.info(f"\n--- Sample Extracted Claims (first 5) ---")
        for i, claim in enumerate(extracted_claims[:5]):
            logger.info(f"  Claim {i+1}:")
            logger.info(f"    PMID: {claim.get('pmid')}")
            logger.info(f"    Text: {claim.get('text', 'N/A')[:100]}")
            logger.info(f"    S-P-O: {claim.get('subject', 'N/A')} → {claim.get('predicate', 'N/A')} → {claim.get('object', 'N/A')}")
        
        # Log sample clusters
        logger.info(f"\n--- Sample Clusters (first 3) ---")
        for i, cluster in enumerate(claim_clusters[:3]):
            logger.info(f"  Cluster {cluster.get('cluster_id', i)}:")
            logger.info(f"    Representative: {cluster.get('representative_claim', 'N/A')[:100]}")
            logger.info(f"    Size: {cluster.get('size')} claims")
            logger.info(f"    Source PMIDs: {cluster.get('source_pmids', [])}")
        
        # Trace summary
        claim_clustering_traces = [t for t in trace_events if t.get("node") == "claim_clustering"]
        logger.info(f"\n--- Trace Events (claim_clustering) ---")
        for trace in claim_clustering_traces:
            logger.info(f"  Input: {trace.get('input')}")
            logger.info(f"  Output: {trace.get('output')}")
        
        # Validation checks
        logger.info(f"\n--- Validation Checks ---")
        checks = {
            "evidence_pool_non_empty": len(evidence_pool) > 0,
            "claims_extracted": len(extracted_claims) > 0,
            "clusters_formed": len(claim_clusters) > 0,
            "clusters_less_than_claims": len(claim_clusters) < len(extracted_claims),
            "all_claims_in_clusters": all(
                all(c.get("pmid") for c in cluster.get("claims", []))
                for cluster in claim_clusters
            )
        }
        
        for check_name, result in checks.items():
            status = "✓" if result else "✗"
            logger.info(f"  {status} {check_name}: {result}")
        
        # Save detailed output
        output_file = Path("results/test_clustering_output.json")
        output_file.parent.mkdir(exist_ok=True)
        
        output = {
            "question_snippet": test_question[:200],
            "evidence_pool_size": len(evidence_pool),
            "total_claims_extracted": len(extracted_claims),
            "total_clusters": len(claim_clusters),
            "sample_claims": extracted_claims[:3],
            "sample_clusters": [
                {
                    "cluster_id": c.get("cluster_id"),
                    "representative_claim": c.get("representative_claim"),
                    "size": c.get("size"),
                    "source_pmids": c.get("source_pmids")
                }
                for c in claim_clusters[:3]
            ],
            "validation_checks": checks,
            "trace_events": claim_clustering_traces
        }
        
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"\n✓ Output saved to {output_file}")
        
        # Final summary
        logger.info("\n" + "=" * 80)
        all_passed = all(checks.values())
        if all_passed:
            logger.info("✓ ALL VALIDATION CHECKS PASSED")
        else:
            logger.info("⚠ Some validation checks failed")
            for check_name, result in checks.items():
                if not result:
                    logger.warning(f"  ✗ {check_name}")
        logger.info("=" * 80)
        
        return all_passed
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = asyncio.run(test_claim_clustering())
    exit(0 if success else 1)
