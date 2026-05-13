# FILE: src/config.py
"""
Configuration management for the Medical RAG Chatbot.

Loads environment variables and provides centralized configuration access.
All secrets and API keys are read from environment variables.

Example Usage:
    from src.config import settings
    
    print(settings.GEMINI_MODEL_HEAVY)
    print(settings.GEMINI_MODEL_LIGHT)
    print(settings.PUBMED_BASE_URL)
"""
'''ths is the centralized configuration of the entire project file '''


import os
import logging
from datetime import datetime
from uuid import uuid4
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings  # this comes from pydantic settings system
'''it cutomatically loads value from 
1. Enviroment variables
2..env file
3.default values'''

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Attributes:
        GOOGLE_CLOUD_PROJECT: GCP project ID for Vertex AI.
        GOOGLE_CLOUD_LOCATION: GCP region (default: us-central1).
        GEMINI_MODEL_HEAVY: Gemini heavy model name to use.
        GEMINI_MODEL_LIGHT: Gemini light model name to use for fast tasks.
        PUBMED_BASE_URL: PubMed E-utilities base URL.
        PUBMED_API_KEY: Optional NCBI API key for higher rate limits.
        MAX_SEARCH_RESULTS: Maximum number of PubMed results to fetch.
        RETRY_MAX_ATTEMPTS: Max retry attempts for transient errors.
        RETRY_WAIT_SECONDS: Initial wait time for exponential backoff.
    """

    # NLI Model Selection
    NLI_MODEL_NAME: str = Field(
        default="cross-encoder/nli-deberta-v3-base",
        description="HuggingFace model name for biomedical NLI (e.g., MedNLI, SciNLI, or default cross-encoder)"
    )
    
    # Google API Configuration
    GOOGLE_API_KEY: Optional[str] = Field(
        default=None,
        description="Google AI Studio API key"
    )
    
    # GCP Configuration (for Vertex AI)
    GOOGLE_CLOUD_PROJECT: str = Field(
        default="your-gcp-project-id",
        description="GCP project ID"
    )
    GOOGLE_CLOUD_LOCATION: str = Field(
        default="us-central1",
        description="GCP region for Vertex AI"
    )
    
    # Gemini Configuration
    GEMINI_MODEL_HEAVY: str = Field(
        default="gemini-2.5-flash",
        description="Gemini heavy model to use"
    )

    GEMINI_MODEL_LIGHT: str = Field(
        default="gemini-3.1-flash-lite-preview",
        description="Gemini light model to use for fast tasks"
    )

    # Ollama settings are deprecated. This deployment uses Gemini-only routing
    # with distinct heavy and light models. Legacy OLLAMA_* env vars are ignored.
    LLM_POLICY: str = Field(
        default="heavy_only",
        description="Model policy: hybrid | heavy_only | cost_saver"
    )
    USE_GEMINI_HEAVY: bool = Field(
        default=True,
        description="Force heavy LLM to Gemini when available"
    )
    USE_GEMINI_LIGHT: bool = Field(
        default=True,
        description="Use Gemini light model for fast/light tasks"
    )
    
    # Colab Configuration (for hybrid architecture)
    USE_COLAB: bool = Field(
        default=False,
        description="Use Google Colab for heavy compute tasks"
    )
    COLAB_API_URL: str = Field(
        default="",
        description="ngrok URL from Colab server (e.g., https://xxxx.ngrok-free.app)"
    )
    USE_HYBRID: bool = Field(
        default=False,
        description="Use Hybrid (Gemini for Heavy, Ollama for Light) agents"
    )
    
    # Retrieval Configuration
    USE_MEDCPT: bool = Field(
        default=False,
        description="Use MedCPT for biomedical retrieval (requires GPU, ~4GB VRAM)"
    )
    USE_HYBRID_RETRIEVAL: bool = Field(
        default=True,
        description="Use hybrid retrieval (BM25 + Dense with RRF)"
    )
    
    # PubMed Configuration
    PUBMED_BASE_URL: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        description="PubMed E-utilities base URL"
    )
    PUBMED_API_KEY: Optional[str] = Field(
        default=None,
        description="NCBI API key for higher rate limits"
    )
    MAX_SEARCH_RESULTS: int = Field(
        default=10,
        description="Maximum PubMed results per query"
    )
    PUBMED_ESEARCH_DELAY_SECONDS: float = Field(
        default=0.0,
        description="Minimum delay between PubMed eSearch calls (seconds)"
    )
    
    # Retry Configuration
    RETRY_MAX_ATTEMPTS: int = Field(
        default=10,
        description="Maximum retry attempts"
    )
    RETRY_WAIT_SECONDS: float = Field(
        default=4.0,
        description="Initial wait time for backoff"
    )

    # Experimental / Epistemic Flags
    ENABLE_NLI: bool = Field(
        default=True,
        description="Enable NLI-based contradiction detection"
    )
    ENABLE_DIALECTICAL_RETRIEVAL: bool = Field(
        default=True,
        description="Enable dialectical (contrastive) retrieval path"
    )
    ENABLE_DS_FUSION: bool = Field(
        default=True,
        description="Enable external Dempster-Shafer fusion module integration"
    )
    FAST_EPISTEMIC: bool = Field(
        default=False,
        description="Fast epistemic mode: reduce retrieval/rerank/extraction budgets for quicker runs"
    )
    RPS_USE_LLM: bool = Field(
        default=True,
        description="Use LLM for RPS extraction; if false, use rule-based heuristics"
    )
    TCS_USE_LLM: bool = Field(
        default=True,
        description="Use LLM for Temporal Conflict Scoring; if false, use heuristics"
    )

    # Epistemic Mode
    EPISTEMIC_MODE: bool = Field(
        default=False,
        description="Enable epistemic guardrails: EUS thresholding, controversy gating, "
                    "NLI contradiction maps, and Dempster-Shafer without calibration overrides"
    )

    # Logging
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    LOG_TO_FILE: bool = Field(
        default=True,
        description="Write logs to file"
    )
    LOG_DIR: str = Field(
        default="logs",
        description="Directory to store log files"
    )
    LOG_FILE_PREFIX: str = Field(
        default="ma_rag",
        description="Log file name prefix"
    )
    LOG_NODE_TIMINGS: bool = Field(
        default=True,
        description="Log start/end timing for graph nodes"
    )
    # Runtime tuning (MRAGE_ env vars)
    MRAGE_EXTRACTION_CONCURRENCY: int = Field(
        default=5,
        description="Max concurrent chunk extraction calls (MRAGE_EXTRACTION_CONCURRENCY)"
    )
    MRAGE_FAST_EPISTEMIC_CAP: int = Field(
        default=2,
        description="Cap for extraction concurrency when FAST_EPISTEMIC is enabled (MRAGE_FAST_EPISTEMIC_CAP)"
    )
    MRAGE_LLM_MAX_CONCURRENCY: int = Field(
        default=5,
        description="Global LLM concurrency semaphore size (MRAGE_LLM_MAX_CONCURRENCY)"
    )
    MRAGE_LLM_RPM: int = Field(
        default=60,
        description="LLM requests-per-minute rate limiter (MRAGE_LLM_RPM)"
    )
    MRAGE_LLM_TIMEOUT: float = Field(
        default=120.0,
        description="Per-LLM call timeout in seconds (MRAGE_LLM_TIMEOUT)"
    )
    MRAGE_CHUNK_EXTRACTION_TIMEOUT: float = Field(
        default=90.0,
        description="Per-chunk extraction timeout in seconds (MRAGE_CHUNK_EXTRACTION_TIMEOUT)"
    )
    MRAGE_EXTRACTION_TIMEOUT: float = Field(
        default=180.0,
        description="Backward-compatible overall extraction timeout (MRAGE_EXTRACTION_TIMEOUT)"
    )
    MRAGE_TCS_MAX_PAIRS: int = Field(
        default=10,
        description="Maximum temporal pair comparisons per question (MRAGE_TCS_MAX_PAIRS)"
    )
    MRAGE_FAST_RETRIEVAL_MIN: int = Field(
        default=5,
        description="Minimum number of retrieved documents when FAST_EPISTEMIC is enabled (MRAGE_FAST_RETRIEVAL_MIN)"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """
    Get application settings singleton.
    
    Returns:
        Settings: Validated settings instance.
    """
    # Bug Fix #9: Return singleton instead of creating new instance
    return settings


# Global settings instance (singleton)
settings = Settings()


def configure_logging() -> None:
    """Configure structured logging for the application."""
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    log_file = None
    run_id = os.getenv("RUN_ID", "") or uuid4().hex[:8]

    if settings.LOG_TO_FILE:
        os.makedirs(settings.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(
            settings.LOG_DIR,
            f"{settings.LOG_FILE_PREFIX}_{timestamp}.log",
        )
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | run=%(run_id)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        if not hasattr(record, "run_id"):
            record.run_id = run_id
        return record

    logging.setLogRecordFactory(record_factory)

    logger.info(
        "Logging configured | level=%s file=%s run_id=%s",
        settings.LOG_LEVEL,
        log_file,
        run_id,
    )


def is_evaluation_mode() -> bool:
    """Return True only when MA_RAG_EVAL_MODE is explicitly set to 'true'."""
    return os.getenv("MA_RAG_EVAL_MODE", "").strip().lower() == "true"


def get_past_exp_limit() -> int:
    """Optional cap for past_exp entries; 0 means no trimming."""
    raw_value = os.getenv("MA_RAG_PAST_EXP_LIMIT", "0").strip()
    try:
        limit = int(raw_value)
    except ValueError:
        return 0
    return max(limit, 0)
