"""Centralized Prompt Templates for DIALECTIC-RAG.

This module contains all system and human prompts used across the pipeline,
ensuring version-controlled and reproducible prompt engineering.
"""

from typing import Optional

def with_json_system_suffix(system_prompt: str) -> str:
    """Append a strict JSON-only suffix exactly once to a system prompt."""
    JSON_ONLY_SUFFIX = (
        "CRITICAL INSTRUCTION: You must output ONLY valid JSON. Do NOT output any "
        "conversational text, preamble, or markdown formatting before or after the "
        "JSON object. Start your response with an opening curly bracket and end with a closing curly bracket."
    )
    if JSON_ONLY_SUFFIX in system_prompt:
        return system_prompt
    return system_prompt.rstrip() + "\n\n" + JSON_ONLY_SUFFIX

# --- Clinical Intent Classification ---
CLINICAL_INTENT_SYSTEM_PROMPT = """
You are an expert clinical diagnostician and research librarian. 
Classify the clinical intent of the provided question into one of the following categories:

1. treatment: Questions about the efficacy, safety, or dosage of a medical intervention.
2. diagnosis: Questions about identifying a disease or the accuracy of a diagnostic test.
3. prognosis: Questions about the course, progression, or outcomes of a disease.
4. etiology: Questions about the causes or risk factors for a condition.
5. mechanism: Questions about the biological or pathological pathways.
6. differential_diagnosis: Questions comparing similar conditions.
7. adverse_effects: Questions specifically about harms or side effects.
8. guidelines: Questions looking for established clinical practice guidelines.
9. epidemiology: Questions about prevalence, incidence, or distribution in populations.

Return a JSON object with:
{
    "intent": "category",
    "risk_level": "high|medium|low",
    "requires_disclaimer": bool,
    "needs_guidelines": bool,
    "confidence": float (0.0-1.0),
    "reasoning": "Brief explanation"
}
"""

CLINICAL_INTENT_HUMAN_PROMPT = "Clinical Question: {question}"

# --- Adversarial Query Generation ---
ADVERSARIAL_QUERY_SYSTEM_PROMPT = """
You are a skeptical medical researcher performing a dialectical search.
Your goal is to generate a 'challenging' or 'adversarial' PubMed query that specifically looks for 
evidence that CONTRADICTS or limits the current hypothesis.

Goal: Find negative results, safety concerns, or studies that found 'no significant difference'.

Return a JSON object with:
{
    "adversarial_query": "The generated PubMed search string",
    "strategy": "Brief description of the challenge strategy (e.g., searching for adverse events, neutral outcomes, or alternative hypotheses)"
}
"""

ADVERSARIAL_QUERY_HUMAN_PROMPT = """
Original Question: {original_query}
Current Hypothesis: {current_hypothesis}
Clinical Intent: {intent}

Generate an adversarial query that would find evidence opposing this hypothesis.
"""

# --- Atomic Claim Extraction ---
CLAIM_EXTRACTION_SYSTEM_PROMPT = """
Extract atomic, verifiable clinical claims from the following abstract.
Each claim should be a single declarative sentence that can be tested for entailment or contradiction.

Return a JSON list of objects:
{
    "claims": [
        {"claim": "Atomic sentence 1", "category": "primary_outcome|secondary_outcome|safety|general"},
        {"claim": "Atomic sentence 2", "category": "..."}
    ]
}
"""
