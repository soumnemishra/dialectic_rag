import asyncio
import json
import logging
import os
import sys

# Add root folder to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nodes.pico_extraction import pico_extraction_node
from src.nodes.contrastive_retrieval import contrastive_retrieval_node
from src.nodes.epistemic_scoring import epistemic_scoring_node
from src.models.state import GraphState

logging.basicConfig(level=logging.INFO)

async def run_test():
    question = """A 21-year-old sexually active male with recent travel to Vietnam and Cambodia presents with fatigue, worsening headache, malaise, arthralgia (pain in hands and wrists), fever, maculopapular rash, and laboratory findings including leukopenia and thrombocytopenia.
    Options:
    A) Chikungunya
    B) Dengue fever
    C) Epstein-Barr virus
    D) Hepatitis A
    """
    
    state: GraphState = {"original_question": question, "candidate_answers": [], "trace_events": []}
    
    # 1. PICO
    pico_res = await pico_extraction_node(state)
    state.update(pico_res)
    
    # 2. Retrieval
    retrieval_res = await contrastive_retrieval_node(state)
    state.update(retrieval_res)
    
    # 3. Scoring
    score_res = await epistemic_scoring_node(state)
    
    print("=== SCORING OUTPUT STATS ===")
    for event in score_res.get("trace_events", []):
        print(json.dumps(event, indent=2))
        
    pool = score_res.get("evidence_pool", [])
    if pool:
        print("\n=== SAMPLE EVIDENCE ITEM ===")
        print(json.dumps(pool[0], indent=2))

if __name__ == "__main__":
    asyncio.run(run_test())