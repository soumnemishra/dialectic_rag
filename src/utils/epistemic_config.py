import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "epistemic.yaml"


def _coerce_value(raw: str) -> Any:
    value = raw.strip().strip("\"").strip("'")
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """
    Parse a minimal YAML subset used by config/epistemic.yaml.

    Supports:
      - Nested dicts via 2-space indentation
      - Scalar values (bool, int, float, string)
      - Comments beginning with '#'
    """
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(0, root)]

    for line in text.splitlines():
        raw = line.rstrip()
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Drop trailing inline comments (best-effort for simple config).
        if "#" in raw:
            raw = raw.split("#", 1)[0].rstrip()
            stripped = raw.strip()
            if not stripped:
                continue

        indent = len(raw) - len(raw.lstrip(" "))
        if ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent < stack[-1][0]:
            stack.pop()

        current = stack[-1][1] if stack else root
        if value == "":
            new_block: Dict[str, Any] = {}
            current[key] = new_block
            stack.append((indent + 2, new_block))
        else:
            current[key] = _coerce_value(value)

    return root


@lru_cache(maxsize=1)
def load_epistemic_config(path: str | None = None) -> Dict[str, Any]:
    """
    Load epistemic configuration from YAML (or return empty config).

    If PyYAML is available it will be used, otherwise a minimal parser
    handles the limited config structure we need.
    """
    config_path = Path(path or os.getenv("MRAGE_EPISTEMIC_CONFIG", _DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        logger.info("Epistemic config not found at %s; using defaults", config_path)
        return {}

    try:
        text = config_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to read epistemic config: %s", exc)
        return {}

    try:
        import yaml  # type: ignore
        parsed = yaml.safe_load(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return _parse_simple_yaml(text)


def get_epistemic_setting(key_path: str, default: Any, env_var: str | None = None) -> Any:
    """
    Resolve a setting from environment variables or config/epistemic.yaml.

    Args:
        key_path: dot-path for nested config values (e.g. "extraction.max_chunks_complex")
        default: fallback value if config is missing
        env_var: optional environment override name
    """
    if env_var:
        env_value = os.getenv(env_var)
        if env_value is not None:
            return _coerce_value(env_value)

    config = load_epistemic_config()
    node: Any = config
    for key in key_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]

    return default if node is None else node
