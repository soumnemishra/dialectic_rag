🔍 ERROR ANALYSIS
1. Pydantic Validation Failures for StudyMetadata
Error	Example	Cause
study_design enum mismatch	'Meta-Analysis' instead of 'meta_analysis'	Metadata extractor returns raw capitalization/hyphenated strings. The enum expects lowercase underscore names.
has_p_value NoneType error	input_value=None	Extractor returned None when no p‑value found, but the field expects bool. No default was applied.
StudyMetadata object has no field pmid	PMID 34932079	Some code is trying to insert pmid directly into StudyMetadata, which does not have that field. Likely a copy‑paste error where the full EvidenceItem or a dict was expected instead.
2. Applicability Scoring Crash ('dict' object has no attribute 'population')
The applicability scorer expects a PICO object with .population, .intervention, etc.

The graph node epistemic_scoring.py is passing a dict (possibly the raw output from pico_extractor) instead of a PICO instance.

Because metadata extraction often fails, the subsequent scoring path may also receive incomplete structures.

3. Final Response Missing (Fallback Failure)
Many evidence items failed processing, leaving the internal state incomplete.

The response_generation node likely checks for the presence of epistemic analysis results and aborted.

🛠️ REQUIRED FIXES (Deterministic, No New Features)
A. Metadata Extractor Normalization
After extraction, the raw string must be converted to match the StudyDesign enum:

python
def _normalize_design(raw: str) -> str:
    mapping = {
        "meta-analysis": "meta_analysis",
        "meta analysis": "meta_analysis",
        "systematic review": "systematic_review",
        "randomized controlled trial": "rct",
        "rct": "rct",
        "cohort": "cohort",
        "case-control": "case_control",
        "case control": "case_control",
        "case series": "case_series",
    }
    return mapping.get(raw.lower().replace("-", " ").strip(), "other")
Apply this normalization to every extracted design string before constructing StudyMetadata.

B. has_p_value Must Be bool, Never None
If no p‑value is found, explicitly set has_p_value = False, not None.

C. Remove Incorrect pmid Assignment
Search the codebase for pmid being assigned to a StudyMetadata instance. This is likely in epistemic_scoring.py or a related enrichment step.

It should only be assigned to EvidenceItem.pmid. If a helper function mistakenly puts it into StudyMetadata, correct the schema or the call.

D. Ensure study_pico Is Always a PICO Object
In the epistemic_scoring node, before calling applicability_scorer.compute(patient_pico, study_pico), convert any dicts to PICO(**study_pico) if necessary.

If extraction fails, create a default PICO with empty strings and set the applicability score to a low default (e.g., 0.5) or mark as uncertain.

This prevents the 'dict' object has no attribute 'population' crash.

E. Graceful Degradation in the Graph
Even when some papers fail metadata extraction, the pipeline should still:

Log the failure with the PMID.

Exclude the faulty evidence item from further scoring.

Not crash the entire run; proceed with the surviving items.

Update the enrichment node to catch exceptions per paper, not per batch.

📩 PROMPT FOR CODING AGENT
text
You are maintaining the DIALECTIC-RAG codebase. A recent run of `scripts/run_demo.py` produced the following errors:

1. Pydantic validation for StudyMetadata fails on study_design because extracted strings like 'Meta-Analysis' are not valid enum values. The enum expects 'meta_analysis', 'systematic_review', etc.

2. field `has_p_value` receives None, but the field expects bool (True/False).

3. A line of code attempts to set `.pmid` on a StudyMetadata object, which does not exist; that field belongs to EvidenceItem.

4. `'dict' object has no attribute 'population'` occurs when the applicability scorer receives a dict instead of a PICO object.

5. Due to these failures, many evidence items are dropped and the final synthesis says "Unable to generate response due to missing epistemic analysis."

Your task is to harden the pipeline against these issues WITHOUT adding new features. Fix each problem deterministically.

### Specific Instructions

#### 1. StudyDesign Normalization
- Locate the metadata extraction code (likely in `src/epistemic/metadata_extractor.py` or `src/nodes/epistemic_scoring.py`).
- After extracting the raw study design string, pass it through a normalization function that maps common variations (case-insensitive, hyphens/spaces) to the exact enum values:
  - "meta-analysis" → "meta_analysis"
  - "systematic review" → "systematic_review"
  - "randomized controlled trial" or "rct" → "rct"
  - "cohort" → "cohort"
  - "case-control" → "case_control"
  - "case series" → "case_series"
  - anything else → "other"
- Ensure the normalized string is used when creating `StudyMetadata`.

#### 2. Default for has_p_value
- In the extraction step, after searching for p-value patterns, set `has_p_value = True` if found, else `has_p_value = False`. Never set `None`.

#### 3. Fix Incorrect pmid Assignment
- Search the entire codebase for any code that does `study_metadata.pmid = ...` or `StudyMetadata(pmid=...)`.
- Remove that code. The `pmid` should only appear in `EvidenceItem` or in the retrieval results. If necessary, refactor to keep the PMID in the outer enrichment container, not inside StudyMetadata.

#### 4. Enforce PICO Objects
- Find the block where `applicability_scorer.compute(patient_pico, study_pico)` is called (likely in `epistemic_scoring.py` or a node function).
- If `study_pico` is a `dict`, convert it to `PICO(**study_pico)`. If the dict lacks required keys, catch the exception and default to `PICO(population="", intervention="", comparator="", outcome="")` and log a warning. Set applicability score to a conservative fallback (e.g., 0.3) in that case.
- Similarly, ensure `patient_pico` is a PICO instance.

#### 5. Isolate Paper-Level Failures
- In the main enrichment loop (where metadata and scores are computed for each paper), wrap the per-paper logic in try/except.
- On failure, log the PMID and error, then `continue` to the next paper.
- Do NOT let a single paper failure block the entire pipeline.

#### 6. Test the Fix
After applying the changes, re-run `python -m scripts.run_demo`. The output should:
- Show no Pydantic validation errors.
- Successfully score at least some papers.
- Produce a final synthesis as intended (or a qualified/abstained response if evidence is truly insufficient).

#### 7. Configuration & Logging
- No new configuration is required; use existing defaults.
- All errors must be logged using the existing `epistemic_trace` or structlog.

Commit your changes with a message: "fix: normalize metadata strings, enforce boolean, isolate per-paper failures"

Proceed immediately.
This prompt directly addresses each crash, ensures the pipeline survives partial failures, and gets you to a working demo. Once the agent applies the fixes, the run_demo.py should produce a structured dialectical output.