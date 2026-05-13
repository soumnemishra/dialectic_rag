# Epistemic Reasoning Pipeline: Architecture Guide

## Overview

This document describes the simplified epistemic reasoning pipeline that powers the clinical RAG system. The system is designed to model uncertainty, disagreement, and evidence reliability to generate safer and more transparent clinical recommendations.

## Core Research Contribution

> We present an epistemically aware clinical RAG framework that explicitly models temporal conflict, reproducibility, applicability, contradictory evidence, and uncertainty to generate safer and more transparent clinical recommendations.

## Architectural Principles

1. **Modular Design**: Each module focuses on a single epistemic reasoning task
2. **Interpretability**: All decisions are traceable through epistemic modules  
3. **Determinism**: No stochastic routing or multi-agent debate; single-pass pipeline
4. **Simplicity**: No planning, decomposition, or nested execution graphs
5. **Scientific Focus**: Every module directly supports the epistemic reasoning claim

## Pipeline Stages

### 1. Clinical Intent Classification
**Module**: `clinical_intent_node`
**Input**: Raw medical question
**Output**: 
- Intent type (informational/diagnostic/therapeutic/mechanism)
- Risk level (low/medium/high)
- Disclaimer and guideline flags

**Purpose**: Classify the clinical context to inform downstream processing.

### 2. Retrieval
**Module**: `rag_direct_node`
**Input**: Original question + clinical intent
**Output**: 
- Retrieved documents
- Document IDs
- Evidence quality notes

**Purpose**: Single pass retrieval using optimized queries based on clinical intent.

### 3. Epistemic Scoring (Parallel)

All three scoring modules execute in parallel after retrieval:

#### 3a. Temporal Conflict Scoring
**Module**: `temporal_conflict_node`
**Input**: Retrieved documents, dates
**Output**:
- Temporal Conflict Score (TCS)
- Year-to-year shift patterns
- Overturned findings

**Purpose**: Detect temporal evolution and contradictory evidence across time periods.

#### 3b. Reproducibility Scoring
**Module**: `rps_scoring_node`
**Input**: Document metadata (study design, sample size, etc.)
**Output**:
- RPS scores per document
- Methodological reliability grades

**Purpose**: Assess methodological quality and reliability of evidence.

#### 3c. Applicability Scoring
**Module**: `applicability_scoring_node`
**Input**: Document cohorts, clinical context
**Output**:
- Applicability score
- Cohort compatibility assessment

**Purpose**: Evaluate external validity and cohort applicability to the clinical question.

### 4. Synchronization Point
**Module**: `epistemic_join_node`
**Purpose**: Merge parallel streams before evidence assessment.

### 5. Evidence Quality Assessment
**Module**: `evidence_polarity_node`
**Input**: Retrieved documents, scores
**Output**:
- Polarity (support/refute/insufficient)
- Confidence
- Reasoning

**Purpose**: Synthesize epistemic scores into evidence quality assessment.

### 6. Evidence Governance
**Module**: `evidence_governance_node`
**Input**: Evidence polarity, risk level
**Output**:
- Governance decision (accept/abstain)

**Purpose**: Safety gate—abstain if evidence is insufficient or contradictory given clinical risk.

### 7. Controversy Classification
**Module**: `controversy_classifier_node`
**Input**: Evidence polarity, temporal conflicts
**Output**:
- Epistemic status (settled/contested/evolving)
- Controversy reasoning

**Purpose**: Classify the epistemic state of the evidence base.

### 8. Dialectic Gate (Conditional)
**Module**: `dialectic_gate_node`
**Logic**: Routes to adversarial retrieval if:
- Controversy = CONTESTED or EVOLVING
- Temporal Conflict Score > threshold
- Evidence polarity = INSUFFICIENT
- Clinical risk = HIGH
- Applicability score LOW

Otherwise, skip to EUP.

**Purpose**: Determine whether to perform contrastive retrieval.

### 8a. Adversarial Retrieval (Optional)
**Module**: `adversarial_retrieval_node`
**Input**: Original question, adversarial prompts
**Output**: 
- Counterargument documents
- Opposing thesis evidence

**Purpose**: Retrieve documents supporting opposing viewpoints for rigorous synthesis.

### 8b. Dialectical Synthesis (Optional)
**Module**: `dialectical_synthesis_node`
**Input**: Pro and con evidence
**Output**:
- Thesis documents
- Antithesis documents
- Synthesis reasoning

**Purpose**: Synthesize opposing viewpoints into structured dialectical knowledge.

### 9. Epistemic Uncertainty Propagation
**Module**: `eup_node`
**Input**: All epistemic scores, RPS, applicability, polarity
**Output**:
- Belief intervals (Dempster–Shafer masses)
- Uncertainty per claim
- Epistemic uncertainty score

**Purpose**: Propagate uncertainty through Dempster–Shafer fusion to quantify overall epistemic confidence.

### 10. Decision Alignment
**Module**: `decision_alignment_node`
**Input**: Final synthesis, belief intervals, guidelines
**Output**:
- Aligned final answer
- Guideline reconciliation

**Purpose**: Align epistemic outputs with clinical guidelines while preserving uncertainty.

### 11. Safety Critic
**Module**: `safety_critic_node`
**Input**: Final answer, safety flags, risk assessment
**Output**:
- Validated answer
- Safety disclaimers
- Final safety flags

**Purpose**: Final safety gate to prevent unsafe recommendations.

## Data Flow: GraphState Fields

### Query Context
- `original_question`: User's medical question
- `mcq_options`: Multiple-choice options (if applicable)
- `chat_history`: Conversation history

### Epistemic Scores
- `tcs_score`: Temporal Conflict Score (0.0–1.0)
- `rps_scores`: List of reproducibility scores per document
- `applicability_score`: Cohort applicability (0.0–1.0)

### Evidence Assessment
- `evidence_polarity`: Dict with polarity, confidence, reasoning
- `controversy_label`: settled/contested/evolving
- `dialectical_metadata`: Thesis-antithesis details

### Uncertainty Quantification
- `belief_intervals`: Dempster–Shafer belief masses
- `eus_per_claim`: Epistemic uncertainty score per claim
- `eus_override`: Override for critical cases

### Final Decision
- `final_answer`: Structured answer text
- `predicted_letter`: Multiple-choice selection
- `safety_flags`: List of applicable safety flags

### Audit Trail
- `trace_id`: Unique execution ID
- `trace_events`: Structured causal trace
- `trace_created_at`: ISO timestamp

## Conditional Routing

### Evidence Governance → Next Stage
If `governance_decision` == "abstain":
→ Jump to `safety_critic` (skip complex synthesis)

Else:
→ Continue to `controversy_classifier`

### Dialectic Gate → Retrieval Strategy
If controversy/conflict/insufficient evidence/high risk:
→ Trigger adversarial_retrieval → dialectical_synthesis

Else:
→ Skip directly to eup

## Key Design Decisions

1. **Single Retrieval Call**: No replanning or decomposition. One retrieval, then epistemic analysis.

2. **Parallel Epistemic Scoring**: TCS, RPS, and applicability computed in parallel for efficiency.

3. **Conditional Contrastive Retrieval**: Adversarial retrieval triggered only by explicit epistemic markers, not heuristic routing.

4. **Dempster–Shafer Fusion**: Belief masses quantify uncertainty without over-committing to point estimates.

5. **Safety as Final Gate**: No answer is produced without explicit safety validation.

6. **Interpretable Trace**: Every decision is logged with reasons and epistemic context.

## Removed Components

The following components have been removed to focus on epistemic reasoning:

- `planner`: No decomposition needed for single-question RAG
- `executor`: No nested plan execution
- `router_agent`: No multi-modal routing; single epistemic pipeline
- `supplemental_retrieval`: Folded into adversarial_retrieval
- `belief_revision_aggregate`: Integrated into eup_node
- Retry loops: Deterministic single-pass execution
- Multi-agent abstractions: Replaced with epistemic modules

## Naming Convention

| Legacy Term         | New Term                    |
|-------------------- |-----------------------------|
| Agent               | Module (or node)            |
| Multi-agent system  | Epistemic reasoning pipeline|
| Debate/argumentation| Dialectical synthesis       |
| Planning            | (Removed—no decomposition)  |
| Execution           | Pipeline execution          |

## Validation Checklist

- [x] Graph compiles without errors
- [x] All imports resolve
- [x] Epistemic modules functional
- [x] Trace logging enabled
- [x] Safety critic as final gate
- [x] No planning/routing cruft
- [x] Documentation aligned with research contribution

## References

The pipeline directly implements the following:

1. **Temporal Belief Revision**: NLI-based detection of evidence shifts
2. **Methodological Assessment**: RPS scoring based on study design, sample size, preregistration, etc.
3. **Applicability Scoring**: Cohort matching and external validity
4. **Contrastive Retrieval**: Adversarial query generation for contested evidence
5. **Dialectical Synthesis**: Structured thesis–antithesis integration
6. **Epistemic Uncertainty**: Dempster–Shafer mass function propagation
7. **Clinical Safety**: Multi-layer validation and abstention

Each module contributes directly to the core research claim.
