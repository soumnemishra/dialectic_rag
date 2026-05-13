# DIALECTIC-RAG: Truth Matrix & Codebase Reality Check

This `README.md` provides a comprehensive and highly critical analysis of the DIALECTIC-RAG codebase. It cuts through the theoretical architecture to evaluate what is *actually* implemented, exposing technical debt, architectural flaws, and fragile logic.

## 1. Project Overview & Codebase Structure

DIALECTIC-RAG is designed as an epistemically-aware clinical Retrieval-Augmented Generation (RAG) framework. It aims to transcend standard RAG by retrieving both supporting and opposing clinical evidence, evaluating methodological rigor (Reproducibility Potential Score - RPS) and clinical applicability, and fusing these using Dempster-Shafer theory to identify the current scientific consensus (Settled, Contested, Evolving, or Falsified).

In reality, the framework is a complex LangGraph pipeline that glues together large language models (LLMs), local sentence transformers, and PubMed E-utilities. While the theoretical math for uncertainty propagation is soundly implemented, the data extraction and routing layers suffer from fragile string parsing, inefficient model loading, and heavy reliance on LLM fallbacks that compromise determinism and speed.

### Codebase Tree & Annotations

```text
d:\dialetic_rag\
├── config/                     # YAML configuration files (epistemic.yaml, thresholds.yaml, etc.)
├── src/                        # Core application code
│   ├── agent.py                # High-level API (MedicalAgent) wrapping the LangGraph execution
│   ├── config.py               # Settings and environment variable management
│   ├── pubmed_client.py        # PubMed E-utilities client (eSearch, eFetch, XML parsing)
│   ├── query_builder.py        # Legacy/alternative query builder logic
│   ├── core/                   # Shared utilities, model registry (LLMs, SentenceTransformers)
│   ├── epistemic/              # Core scoring & mathematical fusion logic
│   │   ├── applicability_scorer.py   # Embedding-based PICO overlap scoring
│   │   ├── calibrated_abstention.py  # Logic for determining when the agent should abstain
│   │   ├── dempster_shafer.py        # Belief mass assignment and D-S combination rule
│   │   ├── epistemic_state_classifier.py # Fuzzy logic mapping beliefs to epistemic states
│   │   ├── metadata_extractor.py     # LLM-based extraction of N, Study Design, P-values
│   │   ├── nli_engine.py             # Cross-encoder inference (Entailment/Contradiction)
│   │   └── reproducibility_scorer.py # Deterministic weighting of extracted metadata for RPS
│   ├── graph/                  # LangGraph orchestrator
│   │   └── workflow.py         # Defines the StateGraph nodes and edges
│   ├── nodes/                  # LangGraph Node Implementations
│   │   ├── claim_clustering.py         # Extracts claims and clusters via cosine similarity
│   │   ├── conflict_analysis.py        # Runs NLI on clusters and fits temporal trends
│   │   ├── contrastive_retrieval.py    # Generates support/oppose queries and calls PubMed
│   │   ├── epistemic_scoring.py        # Pass 1 & 2 scoring (RPS + Applicability)
│   │   ├── pico_extraction.py          # Separates vignette from MCQs
│   │   ├── response_generation.py      # LLM synthesis of final response
│   │   └── uncertainty_propagation.py  # Fuses evidence masses and triggers abstention
│   ├── retrieval/              # Retrieval helpers
│   │   └── pico_extractor.py   # LLM prompt for PICO extraction
│   └── models/                 # Pydantic schemas and graph state definitions
├── test_single_question.py     # Entry point for testing a single clinical vignette
└── run_evaluation.py           # Batch evaluation script for MedQA/PubMedQA
```

## 2. The LangGraph Pipeline Mapping

The theoretical nodes map directly to LangGraph nodes defined in `src/graph/workflow.py`. Here is the exact tracing:

1. **Clinical Intent Classification & PICO Extraction:** Handled by `src/nodes/pico_extraction.py` calling `src/retrieval/pico_extractor.py`.
2. **Contrastive Retrieval:** Handled by `src/nodes/contrastive_retrieval.py` calling `ContrastiveRetriever` and `PubMedClient`.
3. **Epistemic Scoring (RPS & Applicability):** Handled by `src/nodes/epistemic_scoring.py` coordinating `MetadataExtractor`, `ReproducibilityScorer`, and `ApplicabilityScorer`.
4. **Claim Clustering:** Handled by `src/nodes/claim_clustering.py` using `all-MiniLM-L6-v2` embeddings and a custom `greedy_cluster` function.
5. **Conflict Analysis (NLI):** Handled by `src/nodes/conflict_analysis.py` calling `src/epistemic/nli_engine.py`.
6. **Uncertainty Propagation (Dempster-Shafer theory):** Handled by `src/nodes/uncertainty_propagation.py` calling `src/epistemic/dempster_shafer.py`.
7. **Response Generation & Calibrated Abstention:** Handled by `src/nodes/response_generation.py` referencing the output of `src/epistemic/calibrated_abstention.py`.

## 3. Implementation Status (Truth Matrix)

| Module | Status | Reality Check |
| :--- | :--- | :--- |
| **Clinical Intent Classification & PICO Extraction** | **Partially Implemented** | Uses a highly fragile regex (`re.split(r"(?:\n\s*Options:\s*\n\|\n\s*A:\|\n\s*A\))")`) to separate vignettes from MCQs. If regex fails, falls back to "unknown" instead of throwing a validation error. Doesn't actually classify clinical *intent* (e.g., rejecting non-medical prompts). |
| **Contrastive Retrieval** | **Fully Implemented** | Successfully generates supportive/challenging queries and searches PubMed. *Flaw:* If no candidate answers are extracted, it blindly searches for "intervention", which will pull garbage data. |
| **Epistemic Scoring (RPS & Applicability)** | **Fully Implemented** | Math is sound. Extracts metadata via LLM and scores reproducibility. *Flaw:* Applicability defaults silently to `0.5` if the embedding model fails or if inputs are malformed, silently diluting evidence. |
| **Claim Clustering** | **Fully Implemented** | Extracts atomic claims and clusters using cosine similarity. *Flaw:* Instantiates the SentenceTransformer model *per request* (`ModelRegistry.get_sentence_transformer`) inside the node, creating massive memory overhead for parallel runs. |
| **Conflict Analysis (NLI)** | **Partially Implemented** | NLI engine attempts to use a local cross-encoder but explicitly includes logic to fall back to an LLM JSON parser if the local model fails. LLMs are notoriously poor and slow at strict NLI tasks compared to dedicated encoders. Temporal trend uses basic `scipy.stats.linregress` which fails gracefully but is statistically weak for sparse publication years. |
| **Uncertainty Propagation (Dempster-Shafer)** | **Fully Implemented** | Dempster's rule of combination and Yager's fallback are accurately coded. Belief masses are assigned correctly based on stance and RPS weights. |
| **Response Generation & Calibrated Abstention** | **Fully Implemented** | Works as intended. Respects abstention tiers (FULL, QUALIFIED, ABSTAIN) and synthesizes clinical rationales effectively. |

## 4. Architectural Flaws & Technical Debt

A deep scan of the codebase reveals several critical flaws that will break the system in a production or high-throughput environment:

### 1. Hardcoded Configuration Paths that DO NOT Exist
Files like `src/nodes/epistemic_scoring.py` and `src/epistemic/applicability_scorer.py` attempt to load configuration using:
`Path(__file__).resolve().parents[1] / "config" / "default.yaml"`
**Reality:** There is no `default.yaml` in the `config/` directory (only `thresholds.yaml`, `epistemic.yaml`, etc.). This means the system is currently swallowing `FileNotFoundError` exceptions and falling back to hardcoded dictionaries scattered throughout the codebase.

### 2. Missing Input Validation & Guardrails
If a user inputs "What is the recipe for chocolate cake?", the system does not reject it. It will attempt to extract PICO, query PubMed for "chocolate cake", and attempt to run NLI on culinary literature. There is no intent classifier to halt the graph at `START`.

### 3. Inefficient Model Loading (Memory Leaks)
`src/nodes/claim_clustering.py` calls `ModelRegistry.get_sentence_transformer("all-MiniLM-L6-v2")` inside the node execution. `ApplicabilityScorer` does the same. If the registry isn't heavily caching these singletons, this will blow up GPU/RAM under concurrent load. 

### 4. Fragile Regex & Silent Failures
- The MCQ stripping logic in `pico_extraction.py` assumes perfectly formatted datasets (e.g., `A)` or `A:`). Real-world doctor inputs will break this.
- Across the pipeline, `try-except` blocks catch `Exception` and return fallback values (e.g., `0.5` applicability, `"unknown"` PICO, `NEUTRAL` NLI) instead of raising errors. This makes debugging nearly impossible because the pipeline succeeds but hallucinates a "Neutral" consensus based on default values.

### 5. PubMed Client Bottlenecks
`src/pubmed_client.py` uses `BeautifulSoup(..., "xml")` for parsing eFetch responses. For hundreds of articles, BeautifulSoup is notoriously slow and memory-intensive compared to `lxml`'s native tree parsing. 

## 5. Setup & Execution Instructions

### Prerequisites & Dependencies
Assuming standard Python dependencies (require installing via `pip`):
```bash
pip install langgraph langchain-core pydantic aiohttp beautifulsoup4 numpy scipy scikit-learn sentence-transformers pyyaml
```

### Environment Variables
You MUST configure the following keys in your `.env` file or environment:
- `OPENAI_API_KEY` or equivalent provider key (required for LLM extraction and synthesis).
- `NCBI_API_KEY` (Highly recommended to increase PubMed rate limits from 3/sec to 10/sec).
- `MRAGE_SEMANTIC_MODEL` (Optional: defaults to `all-MiniLM-L6-v2`).

### Execution Commands

**To run a single clinical vignette test (best for debugging):**
```bash
python test_single_question.py
```
*Note: This will output a detailed trace JSON file into the `results/` directory.*

**To run batch evaluation on MedQA:**
```bash
python run_evaluation.py --dataset data/benchmark.json --limit 10
```
