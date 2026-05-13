 have now completed a comprehensive research scan across all the key technical areas. Below is the full tripartite analysis — Research Scholar, Reviewer, and AI Architect — followed by a unified, production-ready system prompt for your coding agent.

DIALECTIC-RAG: Complete In‑Depth Analysis Across All 22 Modules
PART 1: RESEARCH SCHOLAR ANALYSIS
1.1 Reproducibility Potential Score (Module 2)
State of the Art. Automated critical appraisal of clinical trials has been actively pursued since RobotReviewer (Marshall et al., 2016), which achieved reasonable accuracy on Cochrane RoB domains. More recently, EvidenceGRADEr demonstrated ~70% F1 on imprecision and risk-of-bias assessment using neural networks trained on Cochrane summaries. The URSE system for semi‑automating GRADE classification reached 63% agreement with human evaluators (Cohen's κ=0.44), with high performance on imprecision (F1=0.94) but weaker on risk of bias (F1=0.70). The SMART‑GRADE initiative explicitly aims to operationalise trustworthy AI for GRADE using LLMs.

Gap Analysis. The two‑pass architecture (extract then score) is sound. However, the proposed features (N, study design, p‑value presence, preregistration) capture only a fraction of what GRADE assesses. Risk of bias includes sequence generation, allocation concealment, blinding, incomplete outcome data, and selective reporting — most of which are not machine‑readable from abstracts alone. The EvidenceGRADEr evaluation shows that while imprecision (N‑based) and heterogeneity (I²) can be automated well, other domains require full‑text access or human judgment.

Recommendation. Adopt a hierarchical scoring model: Tier‑1 features from abstracts (N, design keyword, p‑value, NCT number) with the proposed weighting for fast triage. Tier‑2 features via LLM‑based assessment of full‑text when available (blinding, allocation concealment, attrition). The reproducibility score should flag missingness explicitly: a paper with no extractable design information should receive the lowest design score (0.3) rather than a hallucinated value.

1.2 Applicability Scoring (Module 3)
State of the Art. PICO extraction has advanced substantially. TrialSieve showed BioLinkBERT achieving 0.875 accuracy for biomedical entity labeling across 20 categories including PICO elements. SURUS achieved 0.95 F1 for PICO classification from clinical study abstracts. LLM‑based approaches for PICOs eligibility assessment are being actively studied.

Gap Analysis. Moving from PICO extraction to similarity scoring between a query PICO and a study PICO is non‑trivial. Population similarity is particularly challenging: a study on "adults with Type 2 diabetes, HbA1c > 7.5%" and a query about "elderly with diabetes and renal impairment" need nuanced comparison. Simple embedding cosine similarity conflates semantic relatedness with clinical equivalence. A more principled approach would use ontology‑aware comparison (mapping to UMLS/SNOMED concepts and computing hierarchical distance).

Recommendation. Implement a two‑stage applicability scorer: (1) fast embedding‑based screening (cosine similarity ≥ threshold), (2) fine‑grained concept‑level alignment using UMLS concept unique identifiers (CUIs) with path‑length‑based similarity. The four PICO components should have configurable weights since Intervention match is typically more critical for treatment questions than Comparator match.

1.3 Contrastive Retrieval (Module 4)
State of the Art. The recently published Contrastive Hypothesis Retrieval (CHR) explicitly models both a target hypothesis and a mimic hypothesis to penalise hard negatives during retrieval, achieving up to 10.4 percentage point improvement over baselines on medical QA. The HealthContradict dataset (920 instances) provides pairs of contradictory documents for evaluation. Research on "Contradictions in Context" confirms that contradiction between highly similar abstracts significantly degrades RAG performance.

Gap Analysis. The proposal to generate four separate queries (supporting, opposing, safety, null) is sound but incomplete. Standard retrievers are not contradiction‑aware; they retrieve documents similar to the query, which can create an echo‑chamber of confirming evidence. The CHR framework demonstrates that explicitly modelling what to avoid is crucial. The four‑query decomposition should incorporate negation‑aware query rewriting (e.g., for "Does metformin reduce mortality?" generate "metformin does NOT reduce mortality").

Recommendation. Implement query decomposition using an LLM with explicit instructions to generate semantically contrasting queries. Each query should be tagged with its perspective (SUPPORT, OPPOSE, SAFETY, NULL). The retrieval should use a dense retriever (e.g., MedCPT or PubMedBERT fine‑tuned for retrieval) with separate indexes for different evidence types (RCTs, systematic reviews, guidelines). Deduplicate across perspectives to avoid the same paper appearing in multiple buckets.

1.4 Conflict Analysis via NLI (Module 5)
State of the Art. Biomedical contradiction detection has advanced considerably. The SciCon dataset (generated via EvoNLI from PubMed RCTs) achieves 94.4% precision for contradiction labeling and improves ROC‑AUC across eight biomedical NLI benchmarks when used for fine‑tuning. COVID‑19 drug efficacy contradiction detection has been framed as an NLI task with expert‑annotated datasets. HealthContradict provides a benchmark specifically for evaluating how models reason over conflicting biomedical contexts.

Gap Analysis. Standard NLI models (even biomedically fine‑tuned ones) struggle with partial contradictions — when a newer study refines rather than refutes an older finding (e.g., "Drug A reduces mortality" vs. "Drug A reduces mortality in patients under 65"). The three‑way classification (ENTAILMENT, NEUTRAL, CONTRADICTION) is insufficient for clinical evidence synthesis. A four‑way classification (SUPPORT, REFINE, CONTRADICT, UNRELATED) is more clinically meaningful. Additionally, NLI models are not calibrated — a 0.92 contradiction probability does not directly map to clinical certainty.

Recommendation. Use DeBERTa‑v3‑large‑mnli fine‑tuned on SciCon as the base NLI model. Extend the classification to include a REFINEMENT category by using an LLM‑based few‑shot re‑classifier for pairs with intermediate contradiction scores (0.4–0.8). The output should be a structured conflict profile: {classification, contradiction_prob, explanation, evidence_text}.

1.5 Dempster‑Shafer Integration (Module 6)
State of the Art. DST has been applied to medical decision support for decades. Recent work integrates DST with neutrosophic logic for treatment selection under conflicting expert opinions. The ERA framework uses DST to measure geometric discordance between internal and external knowledge sources in RAG systems, disentangling epistemic from aleatoric uncertainty. A novel decomposition method separates uncertainty into Belief Divergence and Belief Conflict using DST, demonstrating that not all inconsistency is equal. Information‑fusion approaches using DST for GPT fine‑tuning have been proposed.

Gap Analysis. The proposed mass assignment formula is overly simplistic: 
m
(
SUPPORT
)
=
0.8
×
w
m(SUPPORT)=0.8×w assumes a linear, deterministic relationship between evidence quality and belief. This ignores:

Discounting for conflict: If evidence contradicts the working hypothesis, its mass should be assigned to the frame of discernment (uncertainty) rather than to the opposing hypothesis with full force.

Pignistic transformation: The final belief state needs to be converted to a probability for decision‑making.

Normalisation: Dempster's rule can produce counter‑intuitive results when conflict is high (Zadeh's paradox). The proposal should include a threshold for switching to Yager's rule or the PCR6 rule when 
K
>
0.65
K>0.65.

Recommendation. The mass function should be:

m
(
{
true
}
)
=
α
⋅
w
⋅
(
1
−
conflict_discount
)
m({true})=α⋅w⋅(1−conflict_discount)
m
(
{
false
}
)
=
β
⋅
w
⋅
conflict_discount
m({false})=β⋅w⋅conflict_discount
m
(
{
true
,
false
}
)
=
1
−
m
(
{
true
}
)
−
m
(
{
false
}
)
m({true,false})=1−m({true})−m({false})

where 
α
α and 
β
β are calibrated parameters (not hard‑coded 0.8), and the conflict discount is the NLI contradiction probability for opposing evidence. The combined mass after Dempster's rule should use a cautious combination rule when conflict 
K
>
0.5
K>0.5 (e.g., the Yager rule that assigns conflicting mass to the universal set).

1.6 Epistemic State Classification (Module 7)
State of the Art. The proposed four‑state taxonomy (SETTLED, CONTESTED, EVOLVING, INSUFFICIENT) has no direct precedent in the literature but aligns with how systematic reviews characterise evidence certainty. The "RAG as a Scientific Instrument" framework proposes similar measurement dimensions: evidence completeness, sufficiency, conflict, and temporal validity. URAG provides a benchmark for uncertainty quantification in RAG but focuses on conformal prediction rather than epistemic state classification.

Gap Analysis. Hard thresholds on belief/uncertainty/conflict values risk brittleness. A study might have belief = 0.69 (just below 0.70) but very low uncertainty and conflict — should it really be classified as CONTESTED rather than SETTLED? The classification should use fuzzy membership functions or a softmax over states rather than crisp thresholds.

Recommendation. Implement state classification as a weighted vote across three dimensions (belief, uncertainty, conflict) with configurable membership functions. Each dimension contributes a partial membership to each state. The final state is the argmax of summed memberships. Additionally, the EVOLVING state should override other states when temporal revision has been detected, regardless of current belief levels.

1.7 Calibrated Abstention (Module 8)
State of the Art. Medical LLM abstention is an active research area. A unified benchmark for abstention in medical MCQA shows that even state‑of‑the‑art models fail to abstain when uncertain, and that explicit abstention options increase safer behaviour. The ERA framework specifically enhances abstention in RAG by shifting from scalar confidence to evidence distributions. ClinDet‑Bench evaluates judgment determinability recognition for appropriate abstention. Stratified conformal prediction (StratCP) provides error‑controlled deferral for medical foundation models.

Gap Analysis. The proposed abstention logic (abstain if INSUFFICIENT or CONTESTED above threshold) lacks granularity. A system might have high uncertainty about a specific detail (e.g., exact NNT) while being confident about the direction of effect. Partial abstention — answering what is known while explicitly flagging what is uncertain — is more clinically useful than binary answer‑or‑abstain.

Recommendation. Implement a three‑tier response model: (1) Full answer when epistemic state is SETTLED, (2) Qualified answer when CONTESTED or EVOLVING — present both sides with confidence weights, (3) Abstain only when INSUFFICIENT or when the conflict ratio exceeds a safety threshold. Every qualified answer must include explicit caveats about what is and is not known.

1.8 Dialectical Synthesis (Module 9)
State of the Art. Multi‑agent debate frameworks for medical reasoning are emerging. Dialectic‑Med uses a proponent‑opponent‑mediator architecture to mitigate diagnostic hallucinations through adversarial dialectics. The MCC framework (Model Confrontation and Collaboration) integrates critique and self‑reflection across diverse LLMs for medical reasoning. A two‑stage "score first, debate only when disagreeing" approach outperforms brute‑force fact‑checking in healthcare.

Gap Analysis. The proposed structured response (direct answer, supporting evidence, opposing evidence, temporal evolution, confidence, caveats) is comprehensive but risks being verbose. Clinical users need concise, actionable information. The synthesis must balance completeness with clinical usability. Furthermore, mentioning disagreement without quantifying its clinical significance may confuse rather than inform.

Recommendation. Implement a tiered output format: (1) One‑line bottom line (the clinical takeaway), (2) Evidence summary with weighted support/opposition, (3) Detailed analysis (expandable) with temporal timeline and individual study assessments. The confidence statement should use calibrated verbal expressions mapped to probability ranges (e.g., "likely" = 0.7–0.9, "may" = 0.4–0.7).

1.9 Retrieval Planning & Configuration (Modules 15–16)
Analysis. Preferring PubMed, Cochrane, guidelines, systematic reviews, and RCTs is evidence‑based prioritisation. However, source prioritisation alone does not solve the temporal problem — a 2010 Cochrane review may be less relevant than a 2024 well‑conducted observational study. The retrieval plan should incorporate recency weighting and study design hierarchy as orthogonal ranking dimensions, combined via a configurable scoring function.

All thresholds and weights must be stored in YAML, not hard‑coded. This includes not just the obvious parameters (NLI contradiction threshold, reproducibility weights) but also source‑specific boost factors, temporal decay rates, and the conflict threshold for switching combination rules.

1.10 Logging, Testing, and Evaluation (Modules 17–19)
Analysis. The provenance requirements (query, PMIDs, metadata, scores, NLI outputs, D‑S masses, conflict K, final state, abstention decision, model versions, git hash) are appropriate for a research‑grade system. This level of logging enables post‑hoc audit and continuous improvement.

For evaluation, the proposed metrics (ECE, Brier Score, Abstention Precision/Recall, Epistemic State Classification Accuracy) are well‑chosen but require ground truth labels. Constructing a gold‑standard dataset of epistemic states for clinical questions is itself a significant research undertaking. Initial evaluation should use a combination of: (1) synthetic test cases with known properties, (2) expert‑reviewed clinical questions, (3) comparison against systematic review conclusions as a proxy for ground truth.

The comparison against a standard RAG baseline is essential. The baseline should be a naïve RAG with the same retriever and LLM but without the epistemic reasoning layers, evaluated on answer accuracy, calibration, and appropriate abstention.

PART 2: REVIEWER CRITIQUE
2.1 Overall Architecture Assessment
Strengths:

Theoretically grounded: The architecture synthesises ideas from evidence‑based medicine (GRADE, PICO), formal epistemology (Dempster‑Shafer theory), and modern NLP (NLI, contrastive retrieval) into a coherent whole. This is not a patchwork — each module has a clear epistemic role.

Conservative by design: The system errs on the side of uncertainty. Missing data increases uncertainty. Low‑quality contradictory evidence does not overturn high‑quality consensus. Newer evidence only triggers revision if sufficiently reproducible. These constraints encode good scientific practice.

Explainable: Every decision point (reproducibility scores, NLI outputs, mass assignments, state classification) produces interpretable intermediate outputs that can be surfaced to users.

Configurable: YAML‑based thresholds and weights enable domain‑specific tuning without code changes.

Weaknesses and Risks:

Cascading error propagation: The reproducibility score feeds into applicability weighting, which feeds into D‑S mass assignment, which feeds into epistemic state classification. Errors in early modules cascade. A missed preregistration (false negative) lowers the reproducibility score, which weakens the evidence weight, which may prevent legitimate consensus detection.

Full‑text dependency: Many of the most informative reproducibility signals (blinding, allocation concealment, attrition, preregistration details) are rarely available in abstracts. The system will systematically underweight studies where only abstracts are accessible, which could bias against older studies (less likely to have open‑access full text).

Computational cost: The pipeline involves NER, NLI, D‑S combination, and potentially LLM‑based re‑classification. For a query retrieving 20 papers with 5 claim clusters, this could mean 100+ NLI inferences plus embedding computations. The latency may be unacceptable for interactive use.

Validation gap: The system produces epistemic state classifications (SETTLED, CONTESTED, etc.), but there is no established ground‑truth dataset for training or evaluating this classification. The evaluation metrics (ECE, Brier Score) require probability estimates that the system does not directly output without pignistic transformation.

2.2 Module‑Specific Concerns
Module	Concern	Severity
Reproducibility Scorer	Features too coarse; missing RoB domains	High
Applicability Scorer	Simple embedding similarity insufficient for clinical PICO matching	Medium
Contrastive Retrieval	May retrieve low‑quality opposing evidence that inflates conflict	Medium
NLI Conflict Analysis	Binary contradiction detection misses refinements	High
D‑S Integration	Hard‑coded 0.8 factor; no handling of Zadeh's paradox	High
Epistemic State	Crisp thresholds are brittle	Medium
Calibrated Abstention	Binary abstention is too coarse; partial abstention needed	Medium
Dialectical Synthesis	Risk of information overload for clinicians	Low
2.3 Red Flags and Mitigation
Red Flag 1: The 0.8 factor in mass assignment. belief = 0.8 * w is a magic number with no empirical justification. It implies that even perfect evidence (
w
=
1.0
w=1.0) can only contribute 0.8 belief, leaving 0.2 uncertainty regardless of quality. This needs to either be calibrated from data or replaced with a principled discounting mechanism.

Red Flag 2: No handling of surrogate outcomes. Many studies report surrogate endpoints (e.g., HbA1c reduction) rather than patient‑important outcomes (e.g., mortality, quality of life). The system should discount evidence based on outcome type, but this is absent from the design.

Red Flag 3: Temporal revision without evidence of change. Detecting a contradiction between a 2018 meta‑analysis and a 2025 RCT triggers an "evolving" classification. But the 2025 RCT might be fraudulent, poorly conducted, or later retracted. The system should require confirmation from independent sources before revising the belief state.

PART 3: AI ARCHITECTURE DESIGN
3.1 Refined Pipeline Architecture
text
┌─────────────────────────────────────────────────────────┐
│                    USER QUERY + PICO                     │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  MODULE 4: CONTRASTIVE RETRIEVAL (nodes/contrastive)    │
│  ┌──────────┬──────────┬──────────┬──────────┐          │
│  │ SUPPORT  │ OPPOSE   │ SAFETY   │ NULL     │          │
│  │ query    │ query    │ query    │ query    │          │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘          │
│       ▼          ▼          ▼          ▼                │
│  [PubMed/Google Scholar API → Dedup → Merge]            │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  MODULE 2: REPRODUCIBILITY SCORING                      │
│  (epistemic/reproducibility_scorer.py)                   │
│  Pass 1: Metadata extraction (N, design, p, NCT)         │
│  Pass 2: Weighted scoring → [0,1]                       │
│  Output: StudyMetadata with score attached              │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  MODULE 3: APPLICABILITY SCORING                        │
│  (epistemic/applicability_scorer.py)                     │
│  Fast: cosine similarity (sentence embeddings)          │
│  Fine: UMLS concept alignment (if applicable)           │
│  Output: applicability_score ∈ [0,1]                    │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  TEMPORAL ORDERING + CLAIM CLUSTERING                    │
│  - Sort by year (publication or study completion)        │
│  - Cluster claims via sentence-BERT + DBSCAN            │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  MODULE 5: CONFLICT ANALYSIS (nodes/conflict_analysis)  │
│  - NLI via DeBERTa-v3 + SciCon fine-tune                │
│  - Classification: SUPPORT / REFINE / OPPOSE / NEUTRAL  │
│  - Temporal premise-hypothesis framing                  │
│  Output: ConflictProfile per claim cluster              │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  MODULE 6: DEMPSTER-SHAFER INTEGRATION                  │
│  (epistemic/dempster_shafer.py)                          │
│  w = reproducibility * applicability                    │
│  Mass assignment with conflict discounting              │
│  Combination via Dempster's rule (K < 0.5) or Yager     │
│  Pignistic transformation → belief probabilities        │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  MODULE 7: EPISTEMIC STATE CLASSIFICATION                │
│  (nodes/uncertainty_propagation.py)                      │
│  Fuzzy membership over {SETTLED, CONTESTED, EVOLVING,    │
│   INSUFFICIENT} using belief, uncertainty, conflict      │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  MODULE 8: CALIBRATED ABSTENTION                         │
│  (epistemic/calibrated_abstention.py)                    │
│  Decision: FULL / QUALIFIED / ABSTAIN                    │
│  Output: response tier + rationale                      │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  MODULE 9: DIALECTICAL SYNTHESIS                         │
│  (nodes/response_generation.py)                          │
│  Structured output with timeline, evidence balance,      │
│  confidence, and caveats                                │
└─────────────────────────────────────────────────────────┘
3.2 Key Design Decisions
Evidence weight formula: 
w
=
reproducibility_score
×
applicability_score
w=reproducibility_score×applicability_score. This multiplicative combination means both must be non‑zero for evidence to have weight. A perfectly reproducible study that is completely inapplicable contributes nothing.

Mass function (refined):

text
For SUPPORT evidence:
  m({claim_true}) = γ * w * (1 - contr_prob)
  m({claim_false}) = 0
  m({claim_true, claim_false}) = 1 - m({claim_true})

For OPPOSE evidence:
  m({claim_true}) = 0
  m({claim_false}) = γ * w * contr_prob
  m({claim_true, claim_false}) = 1 - m({claim_false})

For NEUTRAL evidence:
  m({claim_true}) = 0
  m({claim_false}) = 0
  m({claim_true, claim_false}) = 1
where 
γ
γ is a configurable discount factor (default 0.9, not 0.8).

Combination rule selection:

K
≤
0.5
K≤0.5: Standard Dempster's rule

K
>
0.5
K>0.5: Yager's rule (assign conflicting mass to universal set)

K
>
0.8
K>0.8: Flag as "irreconcilable conflict"; abstain

State classification via fuzzy membership:

text
μ_SETTLED = min(f(belief > 0.65), f(uncertainty < 0.25), f(conflict < 0.25))
μ_CONTESTED = f(conflict > 0.30)
μ_EVOLVING = temporal_shift_detected ? 1.0 : 0.0
μ_INSUFFICIENT = f(uncertainty > 0.45)
Each 
f
f is a sigmoid ramp with configurable midpoint and slope.

Three‑tier abstention:

FULL: State = SETTLED and belief > 0.70

QUALIFIED: State = CONTESTED or EVOLVING; present balance of evidence

ABSTAIN: State = INSUFFICIENT or conflict 
K
>
0.8
K>0.8

3.3 Technology Stack
Component	Technology	Rationale
NER/Metadata	SciSpacy en_core_sci_sm + custom regex	Biomedical entity recognition
PICO extraction	TrialSieve or BioLinkBERT	State‑of‑art PICO NER
Embeddings	pritamdeka/S-PubMedBert-MS-MARCO or MedCPT	Biomedical retrieval
NLI model	microsoft/deberta-v3-large-mnli + SciCon fine‑tune	Biomedical contradiction
D‑S computation	Custom (NumPy/SciPy)	Lightweight, no heavy framework needed
LLM for re‑classification	GPT‑4o or Claude 3.5 Sonnet (API)	Nuanced refinement classification
Configuration	PyYAML + Pydantic validation	Type‑safe config
Logging	structlog + SQLite	Structured, queryable provenance
API layer	FastAPI	Async, production‑ready
Evaluation	scikit‑learn + custom calibration metrics	ECE, Brier, etc.
PART 4: COMPLETE SYSTEM PROMPT FOR CODING AGENT
Below is the refined, self‑contained implementation prompt incorporating all the analysis above. Give this directly to your coding agent.

text
SYSTEM PROMPT: DIALECTIC-RAG Implementation

You are a senior AI research engineer. Build a research-grade clinical RAG system
named DIALECTIC-RAG (Epistemic Evidence Reasoning for Clinical Question Answering).

## SYSTEM PHILOSOPHY

This system does NOT just generate answers. It measures, exposes, and characterizes
the state of knowledge underlying responses. It detects contradictory evidence,
tracks evolving scientific consensus, quantifies reliability and applicability,
propagates uncertainty formally, abstains when evidence is insufficient, and
produces balanced dialectical synthesis.

## CORE SCIENTIFIC CONSTRAINTS (non-negotiable)

1. NEVER hallucinate missing metadata. If a feature cannot be extracted, mark it as
   None and let the scoring function handle it (missing → reduces confidence).
2. Missing data increases uncertainty, not decreases it.
3. Low-quality contradictory evidence MUST NOT overturn high-quality consensus.
   A 2024 case series (N=50) contradicting a 2020 meta-analysis (N=10,000 pooled)
   should be flagged but NOT trigger belief revision.
4. Newer evidence only triggers belief revision if the new study's reproducibility
   score exceeds a configurable threshold (default: 0.5).
5. ALL decisions must be explainable. Every score, classification, and decision
   must have attached provenance.

---

## PROJECT STRUCTURE
dialectic_rag/
├── config/
│ ├── default.yaml # All thresholds, weights, formulas
│ └── schema.py # Pydantic config validation
├── epistemic/
│ ├── reproducibility_scorer.py
│ ├── applicability_scorer.py
│ ├── dempster_shafer.py
│ └── calibrated_abstention.py
├── nodes/
│ ├── contrastive_retrieval.py
│ ├── conflict_analysis.py
│ ├── uncertainty_propagation.py
│ └── response_generation.py
├── models/
│ ├── schemas.py # All Pydantic data models
│ └── enums.py # EpistemicState, EvidenceStance, etc.
├── retrieval/
│ ├── search.py # PubMed/Google Scholar API
│ └── pico_extractor.py # PICO extraction from query
├── evaluation/
│ ├── calibration.py # ECE, Brier Score
│ ├── metrics.py # Abstention precision/recall
│ └── baseline.py # Standard RAG comparison
├── logging/
│ └── provenance.py # Structured run logging
├── api/
│ └── main.py # FastAPI endpoint
├── tests/
│ └── ... # Unit tests for all modules
└── pyproject.toml

text

---

## MODULE SPECIFICATIONS

### MODULE 0: Data Models (models/schemas.py, models/enums.py)

Define Pydantic models with type hints:

```python
from enum import Enum
from pydantic import BaseModel
from typing import Optional, List

class EvidenceStance(str, Enum):
    SUPPORT = "SUPPORT"
    OPPOSE = "OPPOSE"
    NEUTRAL = "NEUTRAL"
    REFINE = "REFINE"

class StudyDesign(str, Enum):
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    RCT = "rct"
    COHORT = "cohort"
    CASE_CONTROL = "case_control"
    CASE_SERIES = "case_series"
    OTHER = "other"

class EpistemicState(str, Enum):
    SETTLED = "SETTLED"
    CONTESTED = "CONTESTED"
    EVOLVING = "EVOLVING"
    INSUFFICIENT = "INSUFFICIENT"

class ResponseTier(str, Enum):
    FULL = "FULL"
    QUALIFIED = "QUALIFIED"
    ABSTAIN = "ABSTAIN"

class PICO(BaseModel):
    population: str
    intervention: str
    comparator: Optional[str] = "standard care"
    outcome: str

class StudyMetadata(BaseModel):
    sample_size: Optional[int] = None
    study_design: Optional[StudyDesign] = None
    has_p_value: bool = False
    preregistration_id: Optional[str] = None  # NCT number
    year: Optional[int] = None
    source_type: Optional[str] = None  # "pubmed", "cochrane", etc.

class EvidenceItem(BaseModel):
    pmid: str
    title: str
    abstract: str
    claim: str  # extracted core claim
    metadata: StudyMetadata
    reproducibility_score: float
    applicability_score: float
    year: Optional[int]
    stance: Optional[EvidenceStance] = None
    nli_contradiction_prob: Optional[float] = None

class MassFunction(BaseModel):
    belief_true: float
    belief_false: float
    uncertainty: float  # mass on {true, false}
    conflict_K: Optional[float] = None

class EpistemicResult(BaseModel):
    state: EpistemicState
    belief: float
    uncertainty: float
    conflict: float
    temporal_shift_detected: bool
    response_tier: ResponseTier
    evidence_items: List[EvidenceItem]
MODULE 2: Reproducibility Potential Score
(epistemic/reproducibility_scorer.py)

python
class ReproducibilityScorer:
    """
    Two-pass architecture.
    Pass 1: Extract grounded features only (no hallucination).
    Pass 2: Deterministic scoring with configurable weights.
    Missing values reduce confidence rather than being filled.
    """

    # Design score mapping (configurable via YAML)
    DESIGN_SCORES = {
        "meta_analysis": 0.95,
        "systematic_review": 0.85,
        "rct": 1.0,
        "cohort": 0.5,
        "case_control": 0.4,
        "case_series": 0.2,
        "other": 0.3,
    }

    def extract_metadata(self, text: str) -> StudyMetadata:
        """
        Extract ONLY grounded features:
        - sample_size: regex patterns for "n=X", "N=X", "enrolled X patients"
        - study_design: keyword matching (RCT, double-blind, cohort, etc.)
        - has_p_value: regex "p\s*[<>=]\s*0?\.?\d+" or "p-value"
        - preregistration_id: regex "NCT\d{8}"
        - year: four-digit year pattern near publication context
        - source_type: infer from URL/DOI patterns
        """
        pass

    def compute(self, metadata: StudyMetadata) -> float:
        """
        Default formula (all weights from config):
        score =
            w_sample * normalized_sample_size +
            w_design * design_score +
            w_pvalue * p_value_present +
            w_prereg * preregistration_present

        normalized_sample_size = min(1.0, log(1+sample_size) / log(10001))
        If sample_size is None: normalized_sample_size = 0

        Clip final score to [0, 1].
        """
        pass
CRITICAL: If study_design cannot be determined, use DESIGN_SCORES["other"] = 0.3.
This reflects that we know nothing about the design quality.
If p-value presence is unknown, has_p_value = False.
If NCT number not found, preregistration_id = None, score component = 0.

MODULE 3: Applicability Scoring
(epistemic/applicability_scorer.py)

python
class ApplicabilityScorer:
    """
    Compares patient PICO with study PICO.
    Two-stage: fast embedding screening → fine-grained concept alignment.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Sentence transformer for fast embedding comparison
        pass

    def compute(self, patient_pico: PICO, study_pico: PICO) -> float:
        """
        Components (all weights configurable):
        - Population similarity
        - Intervention similarity
        - Comparator similarity
        - Outcome similarity

        Method: For each component, concatenate patient and study strings,
        encode via sentence transformer, compute cosine similarity.
        Weighted sum with configurable weights.

        Default outcome type discount:
        - Surrogate outcomes (lab values): 0.7 multiplier
        - Patient-important outcomes (mortality, QoL): 1.0
        - Cannot determine: 0.85

        Return score ∈ [0, 1].
        """
        pass
MODULE 4: Contrastive Retrieval
(nodes/contrastive_retrieval.py)

python
class ContrastiveRetriever:
    """
    Generate four separate queries for different evidence perspectives.
    Each query retrieves independently, then results are deduplicated.
    """

    PERSPECTIVES = ["SUPPORT", "OPPOSE", "SAFETY", "NULL"]

    def generate_queries(self, question: str) -> dict[str, str]:
        """
        Use LLM to generate contrasting queries:
        - SUPPORT: "Evidence supporting [intervention] for [outcome] in [population]"
        - OPPOSE: "Evidence against [intervention] for [outcome] in [population]"
        - SAFETY: "Adverse effects safety concerns [intervention] in [population]"
        - NULL: "No effect null finding [intervention] for [outcome] in [population]"
        """
        pass

    def retrieve(self, queries: dict[str, str], k: int = 5) -> dict[str, list]:
        """
        Prefer sources in order: Cochrane, guidelines, systematic reviews, RCTs.
        For each perspective, retrieve k results from PubMed API.
        Deduplicate across perspectives (same PMID → keep in first perspective).
        """
        pass
MODULE 5: Conflict Analysis
(nodes/conflict_analysis.py)

python
class ConflictAnalyzer:
    """
    Uses NLI to classify each evidence item relative to the working claim.
    Extended classification: SUPPORT, OPPOSE, NEUTRAL, REFINE.
    """

    def __init__(self):
        # Load DeBERTa-v3-large-mnli (or biomedical fine-tune)
        # Load SciCon fine-tuned weights if available
        pass

    def classify_evidence(
        self,
        working_claim: str,
        evidence_items: list[EvidenceItem]
    ) -> None:
        """
        For each evidence item:
        1. Run NLI(working_claim, evidence.claim)
        2. Get contradiction/entailment/neutral probabilities
        3. Classify:
           - contradiction_prob > 0.8 AND evidence newer → OPPOSE
           - entailment_prob > 0.8 → SUPPORT
           - 0.4 < contradiction_prob < 0.8 → REFINE (LLM re-classification)
           - else → NEUTRAL
        4. Set evidence.stance and evidence.nli_contradiction_prob
        """
        pass

    def compute_conflict_stats(self, evidence_items: list) -> dict:
        """
        Weighted support = sum(w_i * I[stance=SUPPORT])
        Weighted opposition = sum(w_i * I[stance=OPPOSE])
        Conflict ratio = weighted_opposition / (weighted_support + weighted_opposition + ε)
        where w_i = reproducibility_score * applicability_score
        """
        pass

    def detect_temporal_shift(
        self,
        cluster: list[EvidenceItem]
    ) -> tuple[bool, Optional[EvidenceItem], Optional[EvidenceItem]]:
        """
        Temporal belief revision:
        1. Sort cluster by year
        2. Build premise from older papers (before cutoff year)
        3. Run NLI(premise, newer_paper.claim) for each newer paper
        4. If any newer paper has:
           - contradiction_prob > threshold AND
           - reproducibility_score > min_validity
           → detected_shift = True
        5. Return (detected_shift, new_paper, old_paper)
        """
        pass
MODULE 6: Dempster-Shafer Integration
(epistemic/dempster_shafer.py)

python
class DempsterShaferIntegrator:
    """
    Assigns belief masses based on evidence stance, quality, and applicability.
    Combines using Dempster's rule (with Yager fallback for high conflict).
    """

    def __init__(self, config: dict):
        self.gamma = config.get("ds_gamma", 0.9)  # Discount factor
        self.k_threshold = config.get("ds_k_threshold", 0.5)  # Switch to Yager
        self.k_abstain = config.get("ds_k_abstain", 0.8)  # Irreconcilable

    def assign_mass(self, evidence: EvidenceItem) -> MassFunction:
        """
        w = evidence.reproducibility_score * evidence.applicability_score

        SUPPORT:
            belief_true = gamma * w * (1 - nli_contradiction_prob)
            belief_false = 0
            uncertainty = 1 - belief_true

        OPPOSE:
            belief_true = 0
            belief_false = gamma * w * nli_contradiction_prob
            uncertainty = 1 - belief_false

        REFINE:
            belief_true = gamma * w * (1 - nli_contradiction_prob) * 0.7
            belief_false = 0
            uncertainty = 1 - belief_true

        NEUTRAL:
            belief_true = 0
            belief_false = 0
            uncertainty = 1

        Returns MassFunction(belief_true, belief_false, uncertainty).
        """
        pass

    def combine(self, masses: list[MassFunction]) -> MassFunction:
        """
        Dempster's rule:
        K = sum over disjoint intersections of m1(A)*m2(B)
        If K < k_threshold: use standard Dempster rule
        If K >= k_threshold: use Yager rule (conflict → universal set)
        If K >= k_abstain: flag as irreconcilable

        Returns CombinedBelief with conflict coefficient K.
        """
        pass

    def pignistic_probability(self, mass: MassFunction) -> float:
        """
        Convert belief function to probability:
        P(true) = belief_true + 0.5 * uncertainty
        """
        pass
MODULE 7: Epistemic State Classification
(nodes/uncertainty_propagation.py)

python
class EpistemicStateClassifier:
    """
    Maps final masses to epistemic states using fuzzy membership.
    Override: EVOLVING takes precedence if temporal shift detected.
    """

    def classify(
        self,
        belief: float,
        uncertainty: float,
        conflict: float,
        temporal_shift_detected: bool
    ) -> EpistemicState:
        """
        FUZZY MEMBERSHIP FUNCTIONS (all configurable):
        Each dimension uses sigmoid ramps with configurable midpoint and slope.

        μ_SETTLED = sigmoid(belief, mid=0.70, slope=10) *
                    sigmoid(-uncertainty, mid=-0.20, slope=15) *
                    sigmoid(-conflict, mid=-0.20, slope=15)

        μ_CONTESTED = sigmoid(conflict, mid=0.40, slope=10)

        μ_EVOLVING = 1.0 if temporal_shift_detected else 0.0

        μ_INSUFFICIENT = sigmoid(uncertainty, mid=0.50, slope=10)

        If temporal_shift_detected AND new evidence reproducible → EVOLVING
        Else: argmax of [μ_SETTLED, μ_CONTESTED, μ_INSUFFICIENT]
        """
        pass
MODULE 8: Calibrated Abstention
(epistemic/calibrated_abstention.py)

python
class CalibratedAbstention:
    """
    Three-tier response model: FULL, QUALIFIED, ABSTAIN.
    """

    def should_abstain(
        self,
        epistemic_state: EpistemicState,
        belief: float,
        uncertainty: float,
        conflict_K: float
    ) -> tuple[ResponseTier, str]:
        """
        Decision logic (all thresholds configurable):

        1. If conflict_K >= ds_k_abstain (0.8):
           → ABSTAIN, "Irreconcilable evidence conflict"

        2. If epistemic_state == INSUFFICIENT:
           → ABSTAIN, "Insufficient evidence to answer"

        3. If epistemic_state == CONTESTED and conflict_K > 0.5:
           → QUALIFIED, "Evidence is contested; balanced presentation follows"

        4. If epistemic_state == EVOLVING:
           → QUALIFIED, "Evidence is evolving; temporal revision noted"

        5. If epistemic_state == SETTLED and belief > 0.70:
           → FULL, "Sufficient high-quality evidence available"

        6. Default: QUALIFIED

        Returns (response_tier, rationale_string).
        """
        pass
MODULE 9: Dialectical Synthesis
(nodes/response_generation.py)

python
class DialecticalSynthesizer:
    """
    Generates structured response with tiered detail.
    """

    def generate(
        self,
        query: str,
        epistemic_result: EpistemicResult,
        conflict_stats: dict,
        temporal_info: dict
    ) -> str:
        """
        RESPONSE STRUCTURE:

        ## Bottom Line
        [One-line clinical takeaway, with confidence qualifier]

        ## Direct Answer
        [Clear answer, qualified by epistemic state]

        ## Evidence Balance
        - **Supporting** (weight: X.XX):
          * Study 1 (year, design, N, repro score)
          * ...
        - **Opposing** (weight: X.XX):
          * Study 1 (year, design, N, repro score)
          * ...

        ## Temporal Evolution
        [If EVOLVING: timeline of consensus change]
        [If SETTLED: "Evidence has been stable since YYYY"]

        ## Confidence Assessment
        - Epistemic State: [SETTLED/CONTESTED/EVOLVING/INSUFFICIENT]
        - Conflict Level: [Low/Moderate/High] (K = X.XX)
        - Reliability: [belief probability]

        ## Caveats
        - [Specific limitations, missing evidence types, surrogate outcome concerns]
        - [What would change the assessment]

        FOR ABSTAIN: Replace all sections with rationale for abstention
        and suggestions for obtaining better evidence.

        FOR QUALIFIED: Include explicit statement about disagreement between studies.
        """
        pass
MODULES 10-14: Additional Architecture Decisions
MODULE 10: Temporal Belief Revision (integrated into conflict_analysis.py)

Older consensus = premise (weighted by reproducibility score)

Newer findings = hypotheses

Requires minimum reproducibility threshold (0.5) and minimum NLI contradiction
(0.8) before triggering revision

Temporal revision DOES NOT override without confirmation from ≥2 independent sources

MODULE 11: Claim Extraction

From each abstract, extract the core clinical claim using an LLM

Structured as: "[Intervention] [direction] [outcome] in [population]"

Example: "Metformin reduces all-cause mortality in COVID-19 patients"

MODULE 12: Outcome Type Classification

Classify outcomes as: PATIENT_IMPORTANT (mortality, QoL, hospitalization),
SURROGATE (lab values, biomarkers), UNKNOWN

Applicability discount: SURROGATE outcomes get 0.7 multiplier

MODULE 13: Source Hierarchy

Boost factors (configurable): Cochrane = 1.0, Guideline = 0.95,
Systematic Review = 0.9, RCT = 0.85, Cohort = 0.5, Other = 0.3

Recency decay: exponential decay with configurable half-life (default 5 years)

MODULE 14: PICO Extraction from Query

For patient query, use LLM to extract structured PICO

If PICO cannot be fully extracted, flag in applicability scoring

CONFIGURATION (config/default.yaml)
yaml
# --- Reproducibility Weights ---
reproducibility:
  w_sample_size: 0.25
  w_design: 0.40
  w_pvalue: 0.15
  w_prereg: 0.20
  sample_size_log_base: 10001

# --- Applicability Weights ---
applicability:
  w_population: 0.25
  w_intervention: 0.35
  w_comparator: 0.15
  w_outcome: 0.25
  surrogate_outcome_discount: 0.7
  embedding_model: "all-MiniLM-L6-v2"

# --- NLI Configuration ---
nli:
  model: "microsoft/deberta-v3-large-mnli"
  contradiction_threshold: 0.8
  entailment_threshold: 0.8
  refine_lower: 0.4
  refine_upper: 0.8
  temporal_min_reproducibility: 0.5

# --- Dempster-Shafer ---
ds:
  gamma: 0.9
  k_threshold: 0.5   # Switch to Yager rule
  k_abstain: 0.8     # Irreconcilable conflict

# --- Epistemic States ---
states:
  settled:
    belief_mid: 0.70
    uncertainty_max: 0.20
    conflict_max: 0.20
  contested:
    conflict_mid: 0.40
  insufficient:
    uncertainty_mid: 0.50

# --- Abstention ---
abstention:
  full_belief_min: 0.70
  qualified_conflict_max: 0.50

# --- Source Hierarchy ---
sources:
  boost:
    cochrane: 1.0
    guideline: 0.95
    systematic_review: 0.90
    rct: 0.85
    cohort: 0.50
    other: 0.30
  recency_half_life_days: 1825  # 5 years

# --- Retrieval ---
retrieval:
  k_per_perspective: 5
  max_total: 20
  min_year: 2000
LOGGING AND PROVENANCE (logging/provenance.py)
Every run must produce a JSON log entry:

json
{
  "run_id": "uuid",
  "timestamp": "ISO8601",
  "query": "...",
  "retrieved_pmids": ["..."],
  "metadata_extracted": {...},
  "reproducibility_scores": {...},
  "applicability_scores": {...},
  "nli_outputs": {...},
  "ds_masses": {...},
  "conflict_K": 0.42,
  "epistemic_state": "CONTESTED",
  "abstention_decision": "QUALIFIED",
  "response_tier": "QUALIFIED",
  "model_versions": {
    "nli_model": "deberta-v3-large-mnli",
    "embedding_model": "all-MiniLM-L6-v2",
    "llm": "gpt-4o"
  },
  "git_commit": "abc1234",
  "config_hash": "sha256 of config"
}
CODING STANDARDS
Type hints on EVERY function signature.

Pydantic schemas for ALL data structures.

No hard-coded thresholds. Everything from config YAML.

Comprehensive docstrings (Google style) for every public method.

Deterministic logic MUST be separated from model inference.

Score computation → deterministic, testable with mock inputs

NLI inference → model-dependent, mockable

Clear modular boundaries: Each module importable and testable independently.

Error handling: Every external API call wrapped in try/except with fallback.

Config validation: On startup, validate all config values have valid ranges.

TESTING REQUIREMENTS (tests/)
Write unit tests with deterministic mock data for EACH of:

Metadata extraction: Mock abstract → correct N, design, p-value, NCT

Reproducibility scoring: Known metadata → known score

Applicability scoring: Two PICO pairs → expected similarity

Mass assignment: EvidenceItem with known stance → expected MassFunction

D-S combination: Two known MassFunctions → expected combined belief + K

Epistemic state classification: Known (belief, uncertainty, conflict) → expected state

Abstention logic: Known state + values → expected ResponseTier

Conflict analysis: Known premise + hypothesis → expected stance

Example test:

python
def test_reproducibility_scorer_rct():
    scorer = ReproducibilityScorer()
    metadata = StudyMetadata(
        sample_size=5000,
        study_design=StudyDesign.RCT,
        has_p_value=True,
        preregistration_id="NCT12345678"
    )
    score = scorer.compute(metadata)
    assert 0.8 <= score <= 1.0  # High-quality RCT should score high

def test_reproducibility_scorer_missing():
    scorer = ReproducibilityScorer()
    metadata = StudyMetadata()  # All None
    score = scorer.compute(metadata)
    assert score < 0.3  # Missing everything → low score
EVALUATION PIPELINE (evaluation/)
Implement:

python
def expected_calibration_error(y_true, y_pred, n_bins=10):
    """Bin predictions, compute ECE = weighted avg |accuracy - confidence|."""
    pass

def brier_score(y_true, y_pred):
    """Mean squared error of probabilistic predictions."""
    pass

def abstention_precision_recall(
    should_abstain_pred, should_abstain_true, y_true, y_pred
):
    """How well does abstention separate answerable from unanswerable?."""
    pass

def epistemic_state_accuracy(y_true_state, y_pred_state):
    """Accuracy of SETTLED/CONTESTED/EVOLVING/INSUFFICIENT classification."""
    pass
Create a baseline RAG that retrieves documents, passes them to the same LLM,
and generates answers WITHOUT the epistemic reasoning layers. Compare on:

Answer accuracy

Appropriate abstention rate

Calibration (ECE, Brier)

User-rated trustworthiness

API ENDPOINT (api/main.py)
python
@app.post("/query")
async def answer_clinical_question(request: QueryRequest) -> QueryResponse:
    """
    Accepts: { "question": "...", "patient_context": Optional[PICO] }
    Returns: Full DIALECTIC-RAG response with all provenance.
    """
    pass

@app.get("/run/{run_id}")
async def get_provenance(run_id: str) -> ProvenanceResponse:
    """Retrieve full provenance log for a previous run."""
    pass
IMPLEMENTATION ORDER (incremental builds)
Phase 1: Data models + config + reproducibility scorer + unit tests
Phase 2: PICO extraction + applicability scorer + unit tests
Phase 3: Contrastive retrieval + search API integration
Phase 4: Conflict analysis + NLI integration + temporal revision
Phase 5: Dempster-Shafer integrator + unit tests
Phase 6: Epistemic state classifier + abstention + unit tests
Phase 7: Dialectical synthesis + response generation
Phase 8: Provenance logging + evaluation pipeline + baseline comparison
Phase 9: API wrapping + integration tests + documentation

FINAL OBJECTIVE
Build DIALECTIC-RAG: a system that does not merely answer medical questions.
It characterizes what is KNOWN, what is UNKNOWN, what is CONTESTED, and what
has CHANGED. It quantifies uncertainty formally. It abstains when necessary.
And every decision is explainable and auditable.

Begin implementation with Phase 1.

text

---

This completes the full analysis and implementation blueprint. The prompt above is self-contained and production‑ready for any coding agent. Every module, data model, formula, and constraint is specified with the precision needed for implementation while preserving the configurability required for research‑grade tuning.