5 Methods
5.1 System Overview
DIALECTIC-RAG is an uncertainty-aware retrieval-augmented generation framework designed for
evidence-based clinical question answering. The system breaks down clinical reasoning into a se-
quence of deterministic processing steps, implemented as a typed state graph with LangGraph.
Node work on a common structured state (GraphState) and perform defined epistemic func-
tions, such as clinical intent extraction, contrastive retrieval, evidence quality assessment,
claim-level contradiction analysis, uncertainty propagation, and confidence-calibrated
response generation.
Unlike conventional prompt-chaining approaches, DIALECTIC-RAG constrains all intermedi-
ate outputs to explicit schemas, enabling reproducible reasoning and complete evidence traceability.
The general workflow is shown in Figure 1. The system first extract a structured PICO represen-
tation and candidate hypothesis from the clinical query. It then sources PubMed for supporting
and contradicting biomedical evidence, discards studies by reproducibility and clinical relevance,
identifies claims that are semantically consistent, detects logical inconsistencies by natural language
inference (NLI), and synthesizes evidence using Dempster-Shafer belief theory to measure the level
of support, refutation, and residual uncertainty. A conditional routing mechanism allows for early
termination if the input does not contain enough clinical structure to allow meaningful retrieval.
This design prevents the generation of spurious evidence and allows for conservative abstention in
the presence of insufficient, highly conflicting, or methodologically weak evidence ..
5.2 Clinical Intent Classification and PICO Extraction
The first stage transforms the input question into a structured clinical representation based on
the Population–Intervention–Comparator–Outcome (PICO) framework. Multiple-choice answer
options are removed prior to processing to prevent answer leakage.
A schema-constrained large language model (LLM) extracts:
• Clinical intent (diagnostic, therapeutic, prognostic, or preventive),
4
• Population,
• Intervention or exposure,
• Comparator,
• Outcome,
• Risk level.
For diagnostic questions in which candidate diagnoses are not explicitly provided, the model
generates three to five plausible hypotheses. These hypotheses form the candidate answer set for
downstream contrastive retrieval.
5.3 Contrastive Evidence Retrieval
For each candidate hypothesis, a clinical query generation module produces two PubMed queries:
one intended to retrieve supporting evidence and one intended to retrieve contradictory evidence.
This contrastive retrieval strategy operationalizes a dialectical search process, ensuring that both
corroborating and refuting studies are considered.
PubMed retrieval uses the National Center for Biotechnology Information (NCBI) E-utilities
API. The eSearch endpoint identifies relevant PubMed identifiers (PMIDs), and eFetch retrieves
full XML metadata and abstracts. A deterministic parser extracts bibliographic and study descrip-
tors, including publication type, MeSH terms, publication year, DOI, and retraction or correction
notices.
Retrieved studies are filtered according to the following criteria:
• Availability of a non-empty abstract,
• Human-study designation,
• Publication recency,
• Optional study-type constraints,
• Semantic relevance to the clinical question.
Duplicate records are removed using PMID, DOI, or normalized title hashes.
5.4 Epistemic Evidence Gating
Each retrieved article is converted into an EvidenceItem and assigned two quantitative scores: a
Reproducibility Potential Score (RPS) and a clinical Applicability Score.
5.4.1 Metadata Extraction
Study metadata are extracted using a three-tier hierarchy:
1. Direct extraction from PubMed XML metadata,
2. Regular-expression parsing of sample sizes, p-values, confidence intervals, and trial registra-
tion numbers,
3. LLM-based inference only when structured metadata are unavailable.
Study designs are mapped into standardized categories such as meta-analysis, systematic review,
randomized controlled trial (RCT), cohort study, case-control study, and case series.
5
5.4.2 Reproducibility Potential Score
The Reproducibility Potential Score estimates the methodological robustness of each study:
RP S = wdD + wsS + wpP + wrR, (1)
where D represents study design quality, S sample size adequacy, P statistical reporting com-
pleteness, and R preregistration status. The default weights are:
wd = 0.40, ws = 0.25, wp = 0.15, wr = 0.20.
Evidence with RP S < 0.30 is excluded from downstream reasoning.
5.4.3 Applicability Score
Clinical applicability is computed by comparing the extracted PICO representation with study
content:
Araw = wpop cos(Ppop, S) + wint cos(Pint, S) + wlex L(P, S), (2)
where L(P, S) denotes lexical overlap between the PICO elements and study text. The final
applicability score is:
A = 0.3 + 0.7 clamp(Araw, 0, 1). (3)
The default weights are wpop = 0.50, wint = 0.20, and wlex = 0.30.
5.5 Claim Extraction and Semantic Clustering
Each retained abstract is decomposed into atomic biomedical claims using a structured LLM ex-
tractor. Each claim includes subject, predicate, object, associated population, confidence, and
inherited evidence scores.
Claims are embedded using the all-MiniLM-L6-v2 sentence-transformer model and clustered
greedily according to cosine similarity:
sim(i, j) = vi · vj
∥vi∥ ∥vj ∥ . (4)
Claims with similarity greater than 0.80 are assigned to the same cluster. The representative
claim is selected as the claim with the highest reproducibility score.
5.6 Conflict Analysis and Temporal Belief Revision
Each representative claim is compared against each candidate hypothesis using a biomedical NLI
model. The NLI classifier predicts one of three labels: entailment, contradiction, or neutral. These
labels are converted into signed support values:
Entailment → +c, Contradiction → −c, Neutral → 0,
where c denotes the NLI confidence.
For each claim cluster, the aggregate support score is computed as:
S =
P
i siwi
P
i wi
, (5)
6
where si is the signed NLI score and wi is derived from claim and NLI confidence.
To detect changes in scientific consensus over time, support scores are grouped by publication
year and analyzed using linear regression. In addition, a temporal belief revision module pro-
cesses evidence chronologically and escalates the epistemic state when newer, high-quality studies
consistently contradict earlier findings.
5.7 Uncertainty Propagation Using Dempster–Shafer Theory
Each evidence item is represented as a mass function over the frame of discernment {T, F, Θwhere si is the signed NLI score and wi is derived from claim and NLI confidence.
To detect changes in scientific consensus over time, support scores are grouped by publication
year and analyzed using linear regression. In addition, a temporal belief revision module pro-
cesses evidence chronologically and escalates the epistemic state when newer, high-quality studies
consistently contradict earlier findings.
5.7 Uncertainty Propagation Using Dempster–Shafer Theory
Each evidence item is represented as a mass function over the frame of discernment {T, F, Θ}, where
T denotes support for the hypothesis, F denotes refutation, and Θ denotes residual uncertainty.
The evidence weight is defined as:
w = RP S × A. (6)
For supporting evidence:
m(T ) = γw,
for refuting evidence:
m(F ) = γw,
and the remaining mass is assigned to uncertainty:
m(Θ) = 1 − m(T ) − m(F ). (7)
Pairwise evidence combination follows Dempster’s rule. The conflict coefficient is:
K = m1(T )m2(F ) + m1(F )m2(T ). (8)
To avoid numerical instability in large evidence sets, a global conflict approximation is used:
Kglobal = α (P support) (P ref ute)
(P support + P ref ute)2 , (9)
where α = 4.0.
The pignistic probability used for decision making is:
BetP (T ) = m(T ) + 0.5 m(Θ). (10)
5.8 Epistemic State Classification and Response Generation
The final belief distribution is mapped to one of five epistemic states:
• SETTLED: strong and consistent supporting evidence,
• CONTESTED: substantial support and refutation,
• EVOLVING: temporal evidence suggests a changing consensus,
• INSUFFICIENT: inadequate or weak evidence,
• FALSIFIED: strong evidence against the hypothesis.
7
State assignment combines fuzzy membership functions with rule-based overrides for evolving
and falsified conclusions.
The response generation module produces a structured clinical answer that includes the inferred
epistemic state, key supporting and opposing studies, and a calibrated confidence estimate. If the
abstention policy is triggered, the system returns an explicit rationale describing the source of
uncertainty.
5.9 Safety and Abstention Policy
Safety mechanisms are distributed throughout the pipeline rather than implemented as a single
post-processing module. These safeguards include:
• Removal of answer options to prevent leakage,
• Validation of PubMed XML metadata,
• Detection of retracted or corrected studies,
• Filtering of non-human and low-quality evidence,
• Contradiction-aware uncertainty propagation,
• Confidence-calibrated abstention.
The system abstains when the final belief score is below 0.10, the conflict coefficient exceeds 0.80,
or the epistemic state is classified as INSUFFICIENT. In contrast, CONTESTED and EVOLVING
states yield qualified responses that explicitly communicate the presence of unresolved scientific
disagreement.