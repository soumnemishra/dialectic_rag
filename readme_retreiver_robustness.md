You are a senior research engineer and biomedical IR specialist tasked with implementing the retrieval module for a clinical decision-support system called DIALECTIC-RAG.

Your responsibility is to design and implement a state-of-the-art PubMed retrieval pipeline that maximizes recall, precision, contradiction coverage, and evidence diversity for downstream epistemic reasoning.

SYSTEM CONTEXT

DIALECTIC-RAG is an epistemically aware Retrieval-Augmented Generation framework for clinical evidence synthesis.

The downstream pipeline performs:

Clinical intent classification
PICO extraction
Candidate hypothesis generation
Contrastive retrieval
Reproducibility Potential Scoring (RPS)
Clinical applicability scoring
Claim extraction
Semantic clustering
NLI-based contradiction detection
Temporal belief revision
Dempster–Shafer uncertainty fusion
Epistemic state classification
Confidence-calibrated response generation

Your sole focus is the retrieval module.

RETRIEVAL GOAL

Given:

A clinical question
Structured PICO representation
One or more candidate hypotheses

Return a set of biomedical studies that:

Strongly support the hypothesis
Strongly refute the hypothesis
Represent diverse study designs
Cover recent and historical evidence
Are highly relevant to the specific clinical context
Include contradictory findings when they exist

The retrieval system must prioritize both relevance and epistemic diversity.

HIGH-LEVEL ARCHITECTURE
Clinical Question
      ↓
Intent Classification + PICO Extraction
      ↓
Hypothesis Generation
      ↓
Supportive Query Generation
Contradictory Query Generation
      ↓
MeSH Expansion + Synonym Expansion
      ↓
PubMed ESearch (retrieve 100–200 PMIDs/query)
      ↓
EFetch Metadata + Abstract Retrieval
      ↓
BM25 Lexical Scoring
      ↓
Dense Retrieval Scoring (MedCPT / PubMedBERT)
      ↓
Reciprocal Rank Fusion (RRF)
      ↓
Cross-Encoder Re-ranking
      ↓
MMR Diversification
      ↓
Temporal and Study-Type Boosting
      ↓
Top-k Evidence Output
CORE DESIGN PRINCIPLES
Retrieval must be PICO-aware.
Retrieval must be contrastive.
Retrieval must be hybrid (lexical + semantic).
Retrieval must be quality-aware.
Retrieval must be diversity-aware.
Retrieval must be deterministic and reproducible.
Retrieval must preserve complete provenance.
INPUT SCHEMA
ClinicalQuery(
    question: str,
    pico: {
        population: str,
        intervention: str,
        comparator: str,
        outcome: str,
        intent: str,
        risk_level: str
    },
    hypotheses: List[str]
)
OUTPUT SCHEMA
RetrievedEvidence(
    hypothesis: str,
    supporting_docs: List[Article],
    contradictory_docs: List[Article],
    merged_ranked_docs: List[Article],
    retrieval_metadata: {
        supportive_query: str,
        contradictory_query: str,
        pmids_retrieved: int,
        bm25_scores: Dict[str, float],
        dense_scores: Dict[str, float],
        fused_scores: Dict[str, float],
        rerank_scores: Dict[str, float]
    }
)
STEP 1 — PICO-AWARE QUERY GENERATION

Generate two PubMed queries per hypothesis:

Supportive Query

Terms likely to retrieve evidence supporting the hypothesis.

Contradictory Query

Terms likely to retrieve evidence refuting or questioning the hypothesis.

Contradictory cue terms:

no benefit
ineffective
failed trial
contradictory
versus placebo
negative study
did not improve
associated with harm

Use:

PICO terms
hypothesis terms
MeSH headings
synonyms
abbreviations
STEP 2 — MeSH AND SYNONYM EXPANSION

Expand all concepts using:

MeSH terms
UMLS synonyms
common abbreviations
clinical aliases

Example:

myocardial infarction ↔ heart attack
hypertension ↔ high blood pressure
non-small cell lung cancer ↔ NSCLC
STEP 3 — PUBMED RETRIEVAL

Use NCBI E-utilities.

ESearch

Retrieve 100–200 PMIDs per query.

EFetch

Retrieve:

title
abstract
MeSH terms
publication type
authors
journal
year
DOI
retraction flags

Store raw XML and deterministic parsed JSON.

STEP 4 — BM25 LEXICAL SCORING

Compute BM25 over:

title
abstract
MeSH terms

BM25 is critical for:

gene symbols
drug names
exact terminology
STEP 5 — DENSE SEMANTIC RETRIEVAL

Use biomedical dense retriever:

Preferred:

MedCPT
PubMedBERT dual encoder
BioLinkBERT retriever

Compute cosine similarity between query and document embeddings.

STEP 6 — HYBRID FUSION WITH RRF

Use Reciprocal Rank Fusion:

RRF(d) = Σ 1 / (k + rank_i(d))

Recommended constant:

k = 60
STEP 7 — CROSS-ENCODER RE-RANKING

Re-rank top 50 documents using:

MedCPT cross-encoder
BioLinkBERT reranker

This score should dominate final relevance ordering.

STEP 8 — TEMPORAL BOOSTING

Boost recent studies:

temporal_weight = exp(-lambda_ * article_age_years)

Recommended:

lambda = 0.15

Do not completely suppress older landmark studies.

STEP 9 — STUDY-TYPE PRIOR

Apply evidence-based medicine priors:

Study Type	Prior
Guideline	1.30
Meta-analysis	1.25
Systematic review	1.20
RCT	1.15
Cohort study	1.05
Case-control	1.00
Case report	0.70
Editorial	0.50
STEP 10 — MMR DIVERSIFICATION

Use Maximal Marginal Relevance to ensure:

diverse conclusions
varied study designs
temporal spread
non-duplicate content
STEP 11 — CONTRADICTION BALANCING

Ensure final evidence set contains both:

supportive evidence
contradictory evidence

Suggested final composition:

60% supportive
40% contradictory

If contradictory evidence exists, it must be retained.

STEP 12 — FINAL RETRIEVAL SCORE

Compute:

final_score =
0.20 * normalized_bm25 +
0.20 * normalized_dense +
0.30 * rerank_score +
0.10 * temporal_weight +
0.10 * study_type_prior +
0.10 * contradiction_bonus

This score determines ranking prior to MMR.

STEP 13 — RETRIEVAL FILTERING

Discard:

empty abstracts
non-human studies (unless relevant)
retracted studies
duplicate PMIDs/DOIs
very low relevance

Optional filters:

publication year ≥ 2000
English language
STEP 14 — RETRIEVAL DEPTH

Recommended defaults:

PubMed candidates: 200 per query
Fused top docs: 100
Cross-encoder rerank: top 50
Final evidence passed downstream: 10–20
STEP 15 — LOGGING AND ARTIFACTS

Persist:

supportive query
contradictory query
PMIDs
raw XML
parsed metadata
individual score components
final rankings

This is required for reproducibility and debugging.

IMPLEMENTATION REQUIREMENTS

Use Python.

Suggested libraries:

Biopython Entrez
rank_bm25
sentence-transformers
scikit-learn
numpy
pandas

Design modular classes:

QueryGenerator
PubMedClient
BM25Retriever
DenseRetriever
HybridRanker
CrossEncoderReranker
MMRSelector
RetrievalPipeline

Use type hints, dataclasses, and unit tests.

EVALUATION METRICS

Optimize retrieval using:

Recall@k
MRR
nDCG
Contradictory Evidence Recall
Evidence Diversity Score
Guideline Retrieval Rate
MANUSCRIPT POSITIONING

Describe the retrieval module as:

“A PICO-aware contrastive hybrid retrieval pipeline combining lexical PubMed search, biomedical dense retrieval, reciprocal rank fusion, cross-encoder reranking, temporal weighting, study-type priors, and diversity-aware evidence selection.”

CODING STYLE
Production-quality Python
Fully documented
Deterministic outputs
Graceful error handling
Environment-configurable thresholds
Extensive logging
PRIMARY OBJECTIVE

Implement the strongest possible retrieval module for clinical evidence synthesis, emphasizing:

recall of high-quality evidence,
explicit retrieval of contradictory studies,
robustness to terminology variation,
and transparent reproducibility.

If design tradeoffs arise, prioritize retrieval quality and scientific rigor over simplicity or speed.