import logging
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.core.registry import ModelRegistry, safe_ainvoke
from src.models.state import GraphState
from src.models.enums import ResponseTier, EpistemicState

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """
You are a senior clinical research lead. Synthesize a clinical answer based on the provided evidence and epistemic analysis.
Strictly follow the structure below.

Question: {question}
Multiple-Choice Options:
{mcq_options}
Epistemic Result: {epistemic_result}

Structured Evidence Balance:
Supporting Studies:
{support_summary}

Opposing Studies:
{oppose_summary}

Suggested Confidence Phrase: {verbal_belief}

Structure:
## Clinical Question
[Restate the original clinical question clearly]

## Bottom Line
[One-line clinical takeaway, MUST start with the Suggested Confidence Phrase]

## Direct Answer
[Clear answer, grounded in the provided Evidence Balance. If opposing evidence dominates, state that the intervention is not supported.]

## Evidence Balance
- **Supporting**: [Summarize the supporting evidence provided above]
- **Opposing**: [Summarize the opposing evidence provided above]

## Temporal Evolution
[Describe if consensus is settled or evolving]

## Confidence Assessment
- State: {state_label}
- Reliability: [belief percentage]
- Conflict: [Low/Moderate/High]

## Caveats
[Specific limitations and what would change this assessment]

**Final Answer: [X]**
(Replace X with the predicted letter A, B, C, or D using the options above, or "UNKNOWN" if no conclusion is supported)
"""

async def response_generation_node(state: GraphState) -> Dict[str, Any]:
    """Node to generate the final dialectical synthesis response."""
    ep_result = state.get("epistemic_result")
    
    if not ep_result:
        return {"candidate_answer": "Unable to generate response due to missing epistemic analysis."}

    # HARD-STOP ABSTENTION: Enforce pignistic_belief < 0.10 threshold
    pignistic_belief = getattr(ep_result, 'belief', None) or 0.5
    if pignistic_belief < 0.10:
        logger.info(f"Hard-stop abstention triggered: pignistic_belief={pignistic_belief:.3f} < 0.10 threshold")
        return {
            "candidate_answer": "**Final Answer: ABSTAIN**",
            "final_reasoning": f"Pignistic belief {pignistic_belief:.3f} is below abstention threshold (0.10). System is prohibited from guessing A/B/C/D."
        }

    # Handle Abstention
    if ep_result.response_tier == ResponseTier.ABSTAIN:
        rationale = state.get("abstention_rationale", "Insufficient or conflicting evidence.")
        return {
            "candidate_answer": f"## ABSTAINED\n\n**Rationale**: {rationale}\n\nEvidence was either too limited or too contradictory to support a clinical conclusion."
        }

    # 1. Build Structured Evidence Summary
    from src.models.schemas import EvidenceItem, EvidenceStance
    raw_pool = state.get("evidence_pool", [])
    # Ensure items are EvidenceItem objects for attribute access
    evidence_pool = [EvidenceItem(**item) if isinstance(item, dict) else item for item in raw_pool]
    
    support = [item for item in evidence_pool if item.stance == EvidenceStance.SUPPORT]
    oppose = [item for item in evidence_pool if item.stance == EvidenceStance.OPPOSE]
    
    def format_item(item):
        w = item.reproducibility_score * item.applicability_score
        return f"PMID {item.pmid} (Weight: {w:.2f}): {item.claim}"

    support_str = "\n".join([format_item(i) for i in support]) or "None identified."
    oppose_str = "\n".join([format_item(i) for i in oppose]) or "None identified."
    
    # 2. Verbalize Belief & State
    import yaml
    from pathlib import Path
    config_path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    state_labels = config.get("state_labels", {})
    state_val = str(ep_result.state.value if hasattr(ep_result.state, 'value') else ep_result.state)
    state_label = state_labels.get(state_val, state_val)
    
    belief = ep_result.belief
    if ep_result.state == EpistemicState.FALSIFIED or belief < 0.3:
        verbal_belief = "Evidence does not support"
    elif belief > 0.8:
        verbal_belief = "Strong evidence indicates"
    elif belief > 0.6:
        verbal_belief = "Moderate evidence suggests"
    else:
        verbal_belief = "Evidence is uncertain"

    # 3. Generate Synthesis
    llm = ModelRegistry.get_flash_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
    
    try:
        response = await safe_ainvoke(prompt | llm, {
            "question": state["original_question"],
            "mcq_options": state.get("mcq_options") or "Not provided.",
            "epistemic_result": ep_result.model_dump_json(),
            "support_summary": support_str,
            "oppose_summary": oppose_str,
            "verbal_belief": verbal_belief,
            "state_label": state_label
        })
        return {
            "candidate_answer": response.content,
            "final_reasoning": "Dialectical synthesis completed based on epistemic analysis."
        }
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return {"candidate_answer": "Error generating clinical synthesis."}
