from __future__ import annotations

import json
import inspect
import logging
import os
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, cast

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def serialize_for_json(value: Any) -> Any:
    """Convert common Python/Pydantic objects into JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): serialize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_for_json(v) for v in value]
    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        try:
            return serialize_for_json(value.model_dump())
        except Exception:
            return str(value)
    if is_dataclass(value):
        try:
            return serialize_for_json(asdict(value))
        except Exception:
            return str(value)
    if hasattr(value, "__dict__"):
        try:
            return serialize_for_json(vars(value))
        except Exception:
            return str(value)
    return str(value)


class DebugArtifactManager:
    """Persist deterministic debugging artifacts without affecting business logic."""

    def __init__(self, enabled: bool = False, base_dir: str = "debug") -> None:
        env_enabled = _env_bool("DEBUG_MODE", enabled)
        env_dir = os.getenv("DEBUG_DIR", base_dir)
        self._enabled = env_enabled
        self.base_dir = Path(env_dir)
        if self._enabled:
            self._safe_mkdir(self.base_dir)

    def is_enabled(self) -> bool:
        return self._enabled

    def _safe_mkdir(self, path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Failed to create debug directory %s: %s", path, exc)

    def _resolve(self, relative_path: str) -> Optional[Path]:
        if not self._enabled:
            return None
        path = self.base_dir / relative_path
        self._safe_mkdir(path.parent)
        return path

    def _write_text(self, relative_path: str, content: str) -> Optional[Path]:
        path = self._resolve(relative_path)
        if path is None:
            return None
        try:
            path.write_text(content, encoding="utf-8")
            return path
        except Exception as exc:
            logger.warning("Failed to save debug artifact %s: %s", path, exc)
            return None

    def save_text(self, relative_path: str, content: str) -> Optional[Path]:
        return self._write_text(relative_path, content)

    def save_json(self, relative_path: str, data: Any) -> Optional[Path]:
        serialized = serialize_for_json(data)
        payload = json.dumps(serialized, indent=2, ensure_ascii=False, default=str)
        return self._write_text(relative_path, payload)

    def save_xml(self, relative_path: str, xml_text: str) -> Optional[Path]:
        return self._write_text(relative_path, xml_text)

    def save_bytes(self, relative_path: str, content: bytes) -> Optional[Path]:
        path = self._resolve(relative_path)
        if path is None:
            return None
        try:
            path.write_bytes(content)
            return path
        except Exception as exc:
            logger.warning("Failed to save debug bytes artifact %s: %s", path, exc)
            return None

    def save_query_snapshot(self, query: str, payload: dict[str, Any]) -> Optional[Path]:
        timestamp = datetime.utcnow().isoformat(timespec="seconds").replace(":", "-")
        relative_path = f"retrieval/query_{timestamp}.json"
        snapshot = {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            **payload,
        }
        return self.save_json(relative_path, snapshot)

    def save_exception(self, relative_path: str, exception: Exception) -> Optional[Path]:
        trace = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        return self.save_text(relative_path, trace)


def get_debug_manager(
    enabled: Optional[bool] = None,
    base_dir: Optional[str] = None,
) -> DebugArtifactManager:
    resolved_enabled = _env_bool("DEBUG_MODE", False) if enabled is None else enabled
    resolved_base_dir = os.getenv("DEBUG_DIR", "debug") if base_dir is None else base_dir
    return DebugArtifactManager(enabled=resolved_enabled, base_dir=resolved_base_dir)


def debug_capture(name: str, manager: Optional[DebugArtifactManager] = None) -> Callable[[F], F]:
    """
    Decorator to persist deterministic inputs/outputs/exceptions for functions.

    Supports both sync and async callables.
    """

    def decorator(func: F) -> F:
        local_manager = manager or get_debug_manager()
        folder = name.strip().replace(" ", "_")

        async def _capture_async(*args: Any, **kwargs: Any) -> Any:
            if local_manager.is_enabled():
                local_manager.save_json(f"{folder}/inputs.json", {"args": args, "kwargs": kwargs})
            try:
                result = await cast(Callable[..., Any], func)(*args, **kwargs)
                if local_manager.is_enabled():
                    local_manager.save_json(f"{folder}/outputs.json", result)
                return result
            except Exception as exc:
                if local_manager.is_enabled():
                    local_manager.save_exception(f"{folder}/exception.txt", exc)
                raise

        def _capture_sync(*args: Any, **kwargs: Any) -> Any:
            if local_manager.is_enabled():
                local_manager.save_json(f"{folder}/inputs.json", {"args": args, "kwargs": kwargs})
            try:
                result = cast(Callable[..., Any], func)(*args, **kwargs)
                if local_manager.is_enabled():
                    local_manager.save_json(f"{folder}/outputs.json", result)
                return result
            except Exception as exc:
                if local_manager.is_enabled():
                    local_manager.save_exception(f"{folder}/exception.txt", exc)
                raise

        if inspect.iscoroutinefunction(func):
            wrapped = wraps(func)(_capture_async)
            return cast(F, wrapped)

        wrapped = wraps(func)(_capture_sync)
        return cast(F, wrapped)

    return decorator