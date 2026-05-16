#!/usr/bin/env python3
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure project path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.workflow import build_workflow
from src.config import configure_logging
from scripts.evaluate import expected_calibration_error, brier

configure_logging()
logger = logging.getLogger(__name__)


def map_prediction_to_option(pred_text: str, options: dict) -> str:
    """Strictly extract answer by searching for 'Final Answer: [A-Z|ABSTAIN|UNKNOWN]' pattern.
    
    Priority order:
    1. Look for exact "Final Answer:" pattern with ABSTAIN/UNKNOWN
    2. Look for exact "Final Answer:" pattern with letter (A-D)
    3. Other fallback patterns
    
    Returns:
    - Option key (A/B/C/D) if letter matched
    - "ABSTAIN" if Final Answer: ABSTAIN found
    - "UNKNOWN" if Final Answer: UNKNOWN found
    - Empty string if nothing matched
    """
    if not pred_text:
        return ""
    
    import re
    
    # Pattern 1: **Final Answer: ABSTAIN** or Final Answer: ABSTAIN
    if re.search(r'\*\*Final Answer:\s*ABSTAIN\*\*', pred_text, re.IGNORECASE):
        return "ABSTAIN"
    if re.search(r'Final Answer:\s*ABSTAIN', pred_text, re.IGNORECASE):
        return "ABSTAIN"
    
    # Pattern 2: **Final Answer: UNKNOWN** or Final Answer: UNKNOWN
    if re.search(r'\*\*Final Answer:\s*UNKNOWN\*\*', pred_text, re.IGNORECASE):
        return "UNKNOWN"
    if re.search(r'Final Answer:\s*UNKNOWN', pred_text, re.IGNORECASE):
        return "UNKNOWN"
    
    # Pattern 3: **Final Answer: X** (markdown bold)
    match = re.search(r'\*\*Final Answer:\s*([A-D])\*\*', pred_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 4: Final Answer: X (plain text)
    match = re.search(r'Final Answer:\s*([A-D])', pred_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 5: Final answer: X (case-insensitive)
    match = re.search(r'final answer[:\s]+([A-D])', pred_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # FALLBACK: Try other patterns if "Final Answer" not found
    txt = pred_text.strip().upper()
    
    # direct single letter at end
    for k in options.keys():
        if txt.endswith(k):
            return k
    
    # "Answer: X" pattern
    match = re.search(r'Answer[:\s]+([A-D])', pred_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Contains letter with parens/brackets
    for k in options.keys():
        if f"({k})" in txt or f"[{k}]" in txt or f" {k} " in txt:
            return k
    
    return ""


async def run_batch(n_questions: int = 10):
    logger.info("Building workflow")
    workflow = build_workflow()

    # Load dataset
    data_path = Path("data") / "benchmark.json"
    with open(data_path, "r", encoding="utf-8") as f:
        ds = json.load(f)

    medqa = ds.get("medqa", {})
    ids = sorted(medqa.keys())[:n_questions]

    probs = []
    labels = []
    exacts = []
    per_item = []

    for qid in ids:
        entry = medqa[qid]
        question = entry["question"]
        options = entry.get("options", {})
        gold = entry.get("answer", "")

        logger.info(f"Running question {qid}: {question[:80]}...")

        initial_state = {
            "original_question": question,
            "mcq_options": options,
            "intent": "clinical_question",
            "pico": None,
            "trace_id": qid,
        }

        result = await workflow.ainvoke(initial_state)

        # Extract prediction text
        pred_text = result.get("candidate_answer") or result.get("final_reasoning") or ""

        # Map to option
        pred_option = map_prediction_to_option(pred_text, options)
        
        # Determine correctness: ABSTAIN/UNKNOWN are recorded but not counted as correct/incorrect
        if pred_option in ["ABSTAIN", "UNKNOWN"]:
            is_correct = None  # Mark as abstained
            p_belief = 0.0  # Abstention should have minimal belief
        else:
            is_correct = 1 if pred_option and pred_option == gold else 0

        # Extract pignistic belief from trace events (uncertainty_propagation)
        p_belief_from_trace = None
        for ev in result.get("trace_events", []):
            if ev.get("node") == "uncertainty_propagation":
                out = ev.get("output", {})
                p_belief_from_trace = out.get("pignistic_belief")
                break
        
        # Use trace belief if available, else use 0.0 for abstention
        if p_belief_from_trace is not None:
            p_belief = float(p_belief_from_trace)
        elif is_correct is None:  # abstained
            p_belief = 0.0
        else:
            p_belief = 0.5

        probs.append(p_belief)
        labels.append(1 if is_correct == 1 else 0)  # Only count True positives
        exacts.append(is_correct == 1 if is_correct is not None else False)  # Abstention counts as incorrect

        per_item.append({
            "id": qid,
            "question": question,
            "gold": gold,
            "pred_option": pred_option,
            "pred_text": pred_text,
            "pignistic_belief": float(p_belief),
            "correct": is_correct,
            "abstained": is_correct is None
        })

        logger.info(f"Q{qid} gold={gold} pred={pred_option} prob={p_belief} correct={is_correct}")

    probs_arr = np.array(probs)
    labels_arr = np.array(labels)

    em = float(np.mean(exacts))
    brier_score = brier(probs_arr, labels_arr)
    ece = expected_calibration_error(probs_arr, labels_arr, n_bins=10)
    
    # Count abstentions
    abstentions = [item for item in per_item if item.get("abstained", False)]
    abstention_rate = len(abstentions) / len(per_item) if per_item else 0.0
    
    results = {
        "n_questions": n_questions,
        "exact_match": round(em, 4),
        "brier_score": brier_score,
        "ece": ece,
        "abstention_rate": round(abstention_rate, 4),
        "n_abstained": len(abstentions),
        "per_item": per_item,
        "timestamp": datetime.utcnow().isoformat(),
    }

    out_file = Path("results") / f"batch_medqa_{n_questions}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Batch results saved to {out_file}")
    logger.info(f"Exact Match: {results['exact_match']}, Brier: {results['brier_score']}, ECE: {results['ece']}")
    logger.info(f"Abstentions: {results['n_abstained']}/{results['n_questions']} ({results['abstention_rate']*100:.1f}%)")

    return results


if __name__ == "__main__":
    asyncio.run(run_batch(5))
