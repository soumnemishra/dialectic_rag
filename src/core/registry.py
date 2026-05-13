# this is the model abstraction layer or model registry pattern 

# this file contains the information which descide which model/llm to 
# use based on enviroment 

#this place manages our model registry 


import asyncio
import logging
import os
from collections import deque
from threading import Lock
from typing import Any, Dict, Tuple

from src.config import settings

logger = logging.getLogger(__name__)


class AsyncLimiter:
    """Async sliding-window limiter for LLM requests."""

    def __init__(self, max_calls: int, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._timestamps = deque()
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            while self._timestamps and now - self._timestamps[0] >= self.period_seconds:
                self._timestamps.popleft()

            if len(self._timestamps) >= self.max_calls:
                sleep_time = self.period_seconds - (now - self._timestamps[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                now = asyncio.get_running_loop().time()
                while self._timestamps and now - self._timestamps[0] >= self.period_seconds:
                    self._timestamps.popleft()

            self._timestamps.append(now)


_LLM_RPM = int(getattr(settings, "MRAGE_LLM_RPM", 60))
_LLM_RATE_LIMITER = AsyncLimiter(_LLM_RPM, 60.0)
_CONCURRENCY_SEMAPHORE = asyncio.Semaphore(int(getattr(settings, "MRAGE_LLM_MAX_CONCURRENCY", 5)))
_DEFAULT_LLM_TIMEOUT = float(getattr(settings, "MRAGE_LLM_TIMEOUT", 120.0))


async def safe_ainvoke(chain_or_llm, payload: Dict[str, Any], *, timeout: float | None = None):
    """Rate-limited, concurrency-bounded LLM call with timeout guard.

    Args:
        timeout: Per-call timeout in seconds. Defaults to MRAGE_LLM_TIMEOUT (120s).
                 Set to 0 or None to disable timeout (not recommended).
    """
    effective_timeout = timeout if timeout is not None else _DEFAULT_LLM_TIMEOUT
    async with _LLM_RATE_LIMITER:
        async with _CONCURRENCY_SEMAPHORE:
            if effective_timeout and effective_timeout > 0:
                return await asyncio.wait_for(
                    chain_or_llm.ainvoke(payload),
                    timeout=effective_timeout,
                )
            return await chain_or_llm.ainvoke(payload)

class ModelRegistry:
    # """
    # Central registry for LLM instances.
    # Supports: Colab (remote GPU) and Gemini (Vertex/API).
    # """

    _smart_llm_cache: Dict[Tuple[Any, ...], Any] = {}
    _smart_llm_lock = Lock()
    _sentence_transformer_cache: Dict[Tuple[str, str], Any] = {}
    _sentence_transformer_lock = Lock()
    _cross_encoder_cache: Dict[Tuple[str, str], Any] = {}
    _cross_encoder_lock = Lock()
    _embedding_device: str | None = None
    _embedding_device_lock = Lock()
    
    @staticmethod
    def get_llm(temperature: float = 0.0, json_mode: bool = False):
        """Standard/Legacy accessor - proxies to Smart LLM."""
        return ModelRegistry.get_smart_llm(temperature, json_mode)

    @staticmethod
    def _best_device() -> str:
        """Select the fastest available local device for embedding models."""
        try:
            import torch
        except Exception:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def get_embedding_device() -> str:
        """Return a cached embedding device selection (prefers CUDA when available)."""
        with ModelRegistry._embedding_device_lock:
            if ModelRegistry._embedding_device is not None:
                return ModelRegistry._embedding_device

            try:
                import torch
                cuda_available = torch.cuda.is_available()
            except Exception:
                cuda_available = False

            device = ModelRegistry._best_device()
            logger.info(
                "Embedding device selected | device=%s cuda_available=%s",
                device,
                cuda_available,
            )
            ModelRegistry._embedding_device = device
            return device

    @staticmethod
    def get_sentence_transformer(model_name: str, device: str | None = None):
        """Return a cached SentenceTransformer instance, loading it once."""
        resolved_device = device or ModelRegistry.get_embedding_device()
        cache_key = (model_name, resolved_device)

        with ModelRegistry._sentence_transformer_lock:
            cached = ModelRegistry._sentence_transformer_cache.get(cache_key)
            if cached is not None:
                return cached

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. Embedding model '%s' unavailable.",
                    model_name,
                )
                return None

            logger.info(
                "Embedding model selected | backend=sentence-transformers model=%s device=%s",
                model_name,
                resolved_device,
            )
            model = SentenceTransformer(model_name, device=resolved_device)
            ModelRegistry._sentence_transformer_cache[cache_key] = model
            return model

    @staticmethod
    def get_cross_encoder(model_name: str, device: str | None = None):
        """Return a cached CrossEncoder instance, loading it once."""
        resolved_device = device or ModelRegistry.get_embedding_device()
        cache_key = (model_name, resolved_device)

        with ModelRegistry._cross_encoder_lock:
            cached = ModelRegistry._cross_encoder_cache.get(cache_key)
            if cached is not None:
                return cached

            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. Cross-encoder model '%s' unavailable.",
                    model_name,
                )
                return None

            logger.info(
                "Cross-encoder selected | backend=sentence-transformers model=%s device=%s",
                model_name,
                resolved_device,
            )
            model = CrossEncoder(model_name, device=resolved_device)
            ModelRegistry._cross_encoder_cache[cache_key] = model
            return model
# if we use collab parrt this part of the code supports that 
    @staticmethod
    def _get_colab_llm(temperature: float = 0.0, json_mode: bool = False):
        """Get LLM from Colab server (OpenAI-compatible API)."""
        from langchain_openai import ChatOpenAI

        logger.info(
            "LLM selected | backend=colab model=%s temp=%.2f json=%s base_url=%s",
            settings.GEMINI_MODEL_HEAVY,
            temperature,
            json_mode,
            settings.COLAB_API_URL,
        )
        
        extra_kwargs = {}
        if json_mode:
            extra_kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

        return ChatOpenAI(
            base_url=f"{settings.COLAB_API_URL}/v1",
            api_key="not-needed",  # Colab server doesn't require auth
            model=settings.GEMINI_MODEL_HEAVY,
            temperature=temperature,
            max_tokens=512,
            **extra_kwargs,
        )

    @staticmethod
    def get_smart_llm(temperature: float = 0.0, json_mode: bool = False):
        """
        Get 'Smart' LLM for complex tasks (Planner, Extractor, QA).
        
        Priority: Colab > Vertex (if forced) > Gemini
        """
        cache_key = (
            settings.LLM_POLICY,
            settings.USE_COLAB,
            settings.COLAB_API_URL,
            settings.USE_GEMINI_HEAVY,
            settings.USE_GEMINI_LIGHT,
            settings.GOOGLE_CLOUD_PROJECT,
            settings.GOOGLE_CLOUD_LOCATION,
            settings.GEMINI_MODEL_HEAVY,
            settings.GEMINI_MODEL_LIGHT,
            settings.GOOGLE_API_KEY,
            float(temperature),
            bool(json_mode),
        )

        with ModelRegistry._smart_llm_lock:
            cached = ModelRegistry._smart_llm_cache.get(cache_key)
            if cached is not None:
                return cached

        # Option 1: Use Colab (remote GPU)
        if settings.USE_COLAB and settings.COLAB_API_URL:
            llm = ModelRegistry._get_colab_llm(temperature, json_mode)
            with ModelRegistry._smart_llm_lock:
                ModelRegistry._smart_llm_cache[cache_key] = llm
            return llm

        # Option 2: Force Vertex AI for heavy model when available
        if settings.USE_GEMINI_HEAVY and settings.GOOGLE_CLOUD_PROJECT:
            from langchain_google_vertexai import ChatVertexAI
            logger.info(
                "LLM selected | backend=vertex role=smart model=%s temp=%.2f json=%s project=%s location=%s",
                settings.GEMINI_MODEL_HEAVY,
                temperature,
                json_mode,
                settings.GOOGLE_CLOUD_PROJECT,
                settings.GOOGLE_CLOUD_LOCATION,
            )
            extra_kwargs = {}
            if json_mode:
                extra_kwargs["model_kwargs"] = {
                    "generation_config": {
                        "response_mime_type": "application/json",
                    }
                }
            return ChatVertexAI(
                model_name=settings.GEMINI_MODEL_HEAVY,
                project=settings.GOOGLE_CLOUD_PROJECT,
                location=settings.GOOGLE_CLOUD_LOCATION,
                temperature=temperature,
                **extra_kwargs,
            )
            with ModelRegistry._smart_llm_lock:
                ModelRegistry._smart_llm_cache[cache_key] = llm
            return llm
        
        # Option 3: Use Gemini (API)
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info(
            "LLM selected | backend=gemini role=smart model=%s temp=%.2f json=%s",
            settings.GEMINI_MODEL_HEAVY,
            temperature,
            json_mode,
        )
        extra_kwargs = {}
        if json_mode:
            extra_kwargs["model_kwargs"] = {
                "response_mime_type": "application/json",
            }
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL_HEAVY,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
            convert_system_message_to_human=True,
            max_retries=settings.RETRY_MAX_ATTEMPTS,
            **extra_kwargs,
        )
        with ModelRegistry._smart_llm_lock:
            ModelRegistry._smart_llm_cache[cache_key] = llm
        return llm

    @staticmethod
    def get_fast_llm(temperature: float = 0.0, json_mode: bool = False):
        """
        Get 'Fast' LLM for simple tasks (Step Definer).

        Uses Gemini light model for fast/light tasks when enabled.
        """
        policy = (settings.LLM_POLICY or "hybrid").lower()
        use_light = getattr(settings, "USE_GEMINI_LIGHT", True) and policy in {"hybrid", "cost_saver"}

        if use_light:
                # Prefer Vertex AI when a GCP project is configured (uses Vertex auth).
                if getattr(settings, "GOOGLE_CLOUD_PROJECT", None):
                    from langchain_google_vertexai import ChatVertexAI
                    logger.info(
                        "LLM selected | backend=vertex role=fast model=%s temp=%.2f json=%s",
                        settings.GEMINI_MODEL_LIGHT,
                        temperature,
                        json_mode,
                    )
                    extra_kwargs = {}
                    if json_mode:
                        extra_kwargs["model_kwargs"] = {"generation_config": {"response_mime_type": "application/json"}}
                    return ChatVertexAI(
                        model_name=settings.GEMINI_MODEL_LIGHT,
                        project=settings.GOOGLE_CLOUD_PROJECT,
                        location=settings.GOOGLE_CLOUD_LOCATION,
                        temperature=temperature,
                        **extra_kwargs,
                    )

                # Fallback to Gemini Developer API (requires GOOGLE_API_KEY).
                from langchain_google_genai import ChatGoogleGenerativeAI
                logger.info(
                    "LLM selected | backend=gemini role=fast model=%s temp=%.2f json=%s",
                    settings.GEMINI_MODEL_LIGHT,
                    temperature,
                    json_mode,
                )
                extra_kwargs = {}
                if json_mode:
                    extra_kwargs["model_kwargs"] = {"response_mime_type": "application/json"}
                return ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL_LIGHT,
                    google_api_key=settings.GOOGLE_API_KEY,
                    temperature=temperature,
                    convert_system_message_to_human=True,
                    max_retries=settings.RETRY_MAX_ATTEMPTS,
                    **extra_kwargs,
                )

        return ModelRegistry.get_smart_llm(temperature, json_mode)

    @staticmethod
    def get_light_llm(temperature: float = 0.0, json_mode: bool = False):
        """Alias for get_fast_llm."""
        return ModelRegistry.get_fast_llm(temperature, json_mode)

    @staticmethod
    def get_heavy_llm(temperature: float = 0.0, json_mode: bool = False):
        """Alias for get_smart_llm."""
        return ModelRegistry.get_smart_llm(temperature, json_mode)

    @staticmethod
    def get_flash_llm(temperature: float = 0.0, json_mode: bool = False):
        """Alias for get_fast_llm."""
        return ModelRegistry.get_fast_llm(temperature, json_mode)

