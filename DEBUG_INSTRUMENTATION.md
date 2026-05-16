# DIALECTIC-RAG Debug Instrumentation

## Purpose
This instrumentation layer adds deterministic, research-grade artifact capture for key pipeline stages without changing core business logic or return values.

## Environment Variables
Set these variables to enable specific debug artifact classes:

- DEBUG_MODE=true
- DEBUG_DIR=debug
- DEBUG_PUBMED=true
- DEBUG_METADATA=true
- DEBUG_RETRIEVAL=true
- DEBUG_CALIBRATION=true

If DEBUG_MODE is false or unset, all instrumentation is effectively disabled.

## Added Components

### 1) Shared Utility Module
- File: src/utils/debug_utils.py
- Main class: DebugArtifactManager
- Decorator: debug_capture

Capabilities:
- Create debug directories safely
- Save text/XML/JSON/bytes artifacts
- Save query snapshots with timestamps
- Save exception tracebacks
- Serialize complex objects deterministically

### 2) PubMed Artifact Hooks
- File: src/pubmed_client.py
- Saves raw eFetch XML per PMID:
  - debug/pubmed/{pmid}_raw.xml
- Saves deterministic parsed metadata per PMID:
  - debug/pubmed/{pmid}_parsed.json
- Saves retrieval snapshots:
  - debug/retrieval/query_<timestamp>.json
- Saves exception traces:
  - debug/exceptions/traceback_<timestamp>.txt

### 3) Metadata LLM Artifact Hooks
- File: src/epistemic/metadata_extractor.py
- Saves extracted study-level metadata per PMID:
  - debug/pubmed/{pmid}_metadata_llm.json
- Saves extraction exception traces:
  - debug/exceptions/metadata_{pmid}_exception.txt

### 4) Query-Level Retrieval Snapshots
- File: src/nodes/contrastive_retrieval.py
- Saves retrieval-level snapshot:
  - debug/retrieval/query_<timestamp>.json
- Saves workflow state snapshot:
  - debug/workflow/state_snapshot.json

### 5) Calibration/Uncertainty Metrics
- File: src/evaluation/metrics.py
- Saves calibration metrics:
  - debug/calibration/uncertainty_metrics.json

## Decorator Usage

```python
from src.utils.debug_utils import debug_capture

@debug_capture("metadata_extraction")
def extract_metadata(text: str) -> dict:
    return {"text": text}
```

Generated artifacts:
- debug/metadata_extraction/inputs.json
- debug/metadata_extraction/outputs.json
- debug/metadata_extraction/exception.txt

## Testing

Run targeted debug tests:

```powershell
env\Scripts\python -m pytest tests/test_debug_utils.py -q
```

Current expected result:
- 5 passed

## Notes
- Artifact save failures never crash business logic.
- All writes use UTF-8 and JSON pretty printing with ensure_ascii=False.
- Paths use pathlib for portability.
