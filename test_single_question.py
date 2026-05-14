#!/usr/bin/env python3
"""
Test script to run a single question (0228) through the full graph.
Captures trace, state, and validates applicability/clustering output.
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
import uuid

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph.workflow import build_workflow
from src.models.state import GraphState
from src.config import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


async def main():
    logger.info("Initializing workflow for single-question test...")
    workflow = build_workflow()
    
    # Test question
    test_question = (
        "Two weeks after undergoing an emergency cardiac catherization with stenting for unstable angina pectoris, a 61-year-old man has decreased urinary output and malaise. He has type 2 diabetes mellitus and osteoarthritis of the hips. Prior to admission, his medications were insulin and naproxen. He was also started on aspirin, clopidogrel, and metoprolol after the coronary intervention. His temperature is 38°C (100.4°F), pulse is 93/min, and blood pressure is 125/85 mm Hg. Examination shows mottled, reticulated purplish discoloration of the feet. Laboratory studies show: Hemoglobin count 14 g/dL, Leukocyte count 16,400/mm3, Segmented neutrophils 56%, Eosinophils 11%, Lymphocytes 31%, Monocytes 2%, Platelet count 260,000/mm3, Erythrocyte sedimentation rate 68 mm/h, Serum Urea nitrogen 25 mg/dL, Creatinine 4.2 mg/dL. Renal biopsy shows intravascular spindle-shaped vacuoles. Which of the following is the most likely cause of this patient's symptoms?"
    )
    
    logger.info(f"\n{'='*80}")
    logger.info("Running single-question evaluation")
    logger.info(f"Question: {test_question[:100]}...")
    logger.info(f"{'='*80}\n")
    
    try:
        # Initialize the graph state
        initial_state: GraphState = {
            "original_question": test_question,
            "mcq_options": None,
            "intent": "clinical_question",
            "risk_level": "moderate",
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
        
        logger.info("Invoking workflow with initial state...")
        result = await workflow.ainvoke(initial_state)
        
        logger.info(f"\n{'='*80}")
        logger.info("WORKFLOW EXECUTION COMPLETED")
        logger.info(f"{'='*80}\n")
        
        # Log final state
        logger.info("Final State Summary:")
        for key in ["original_question", "final_reasoning", "candidate_answer", "epistemic_state"]:
            value = result.get(key, "N/A")
            if isinstance(value, str):
                logger.info(f"  {key}: {value[:100]}")
            else:
                logger.info(f"  {key}: {type(value).__name__}")
        
        # Extract and validate key metrics
        logger.info(f"\n{'='*80}")
        logger.info("VALIDATION METRICS")
        logger.info(f"{'='*80}\n")
        
        # Check evidence pool
        evidence_pool = result.get("evidence_pool", [])
        if evidence_pool:
            logger.info(f"Evidence Pool Size: {len(evidence_pool)} items")
            scores = []
            for item in evidence_pool:
                if isinstance(item, dict) and "applicability_score" in item:
                    scores.append(item["applicability_score"])
            
            if scores:
                mean_app = sum(scores) / len(scores)
                min_app = min(scores)
                max_app = max(scores)
                logger.info(f"Applicability Scores:")
                logger.info(f"  - Mean: {mean_app:.3f}")
                logger.info(f"  - Min: {min_app:.3f}")
                logger.info(f"  - Max: {max_app:.3f}")
                logger.info(f"  - Sample (first 3): {scores[:3]}")
                
                if mean_app > 0.3:
                    logger.info("  ✓ PASS: Mean applicability > 0.3")
                else:
                    logger.warning("  ✗ FAIL: Mean applicability ≤ 0.3")
        else:
            logger.warning("No evidence pool in final state")
        
        # Check clusters
        claim_clusters = result.get("claim_clusters", [])
        extracted_claims = result.get("extracted_claims", [])
        logger.info(f"\nClaim Extraction & Clustering:")
        logger.info(f"  - Extracted claims: {len(extracted_claims)}")
        logger.info(f"  - Claim clusters: {len(claim_clusters)}")
        
        if claim_clusters:
            sizes = []
            for cluster in claim_clusters:
                if isinstance(cluster, list):
                    sizes.append(len(cluster))
                elif isinstance(cluster, dict) and "size" in cluster:
                    sizes.append(cluster["size"])
            
            if sizes:
                logger.info(f"  - Cluster sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")
                logger.info(f"  ✓ PASS: Populated claim_clusters")
        else:
            logger.info(f"  ⚠ Note: No clusters generated (may indicate small evidence pool)")
        
        # Check trace events
        trace_events = result.get("trace_events", [])
        logger.info(f"\nTrace Events: {len(trace_events)} events recorded")
        
        node_events = {}
        for event in trace_events:
            node_name = event.get("node", "unknown")
            if node_name not in node_events:
                node_events[node_name] = 0
            node_events[node_name] += 1
        
        for node_name, count in sorted(node_events.items()):
            logger.info(f"  - {node_name}: {count} events")
        
        # Save detailed trace to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path("results") / f"test_single_question_{timestamp}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        trace_output = {
            "question": test_question,
            "final_reasoning": result.get("final_reasoning", ""),
            "candidate_answer": result.get("candidate_answer", ""),
            "evidence_pool_size": len(evidence_pool),
            "extracted_claims": len(extracted_claims),
            "claim_clusters": len(claim_clusters),
            "trace_events": trace_events,
        }
        
        with open(output_file, "w") as f:
            json.dump(trace_output, f, indent=2, default=str)
        
        logger.info(f"\nFull trace saved to: {output_file}")
        
        logger.info(f"\n{'='*80}")
        logger.info("✓ Single-question test completed successfully")
        logger.info(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"Error running single-question test: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
