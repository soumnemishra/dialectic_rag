import asyncio
import os
import sys
import logging

# Setup path to include project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.workflow import build_workflow
from src.models.enums import ResponseTier

async def test_non_clinical():
    workflow = build_workflow()
    
    # Non-clinical query
    initial_state = {
        "original_question": "What is the capital of France?",
        "evidence_pool": [],
        "retrieved_docs": {},
        "step_notes": [],
        "claim_clusters": [],
        "extracted_claims": [],
        "candidate_stances": {},
        "fused_beliefs": {},
        "temporal_shift": {},
        "safety_flags": [],
        "trace_events": []
    }
    
    print("Running non-clinical query test...")
    result = await workflow.ainvoke(initial_state)
    
    print(f"\nFinal Answer:\n{result.get('candidate_answer')}")
    
    # Check if we successfully bypassed retrieval
    trace_nodes = [e["node"] for e in result.get("trace_events", [])]
    print(f"\nNodes visited: {trace_nodes}")
    
    if "contrastive_retrieval" not in trace_nodes:
        print("\nSUCCESS: Retrieval was correctly bypassed.")
    else:
        print("\nFAILURE: Retrieval was NOT bypassed.")

if __name__ == "__main__":
    asyncio.run(test_non_clinical())
