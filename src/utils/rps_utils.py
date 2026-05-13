
from typing import Any, Dict, Optional, Tuple
import math
import logging
import re
import yaml
from src.utils.epistemic_config import get_epistemic_setting
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Centralized config loader ---
_THRESHOLDS_PATH = Path(__file__).resolve().parents[2] / "config" / "thresholds.yaml"
_RPS_CONFIG = None

def _load_rps_config():
    global _RPS_CONFIG
    if _RPS_CONFIG is not None:
        return _RPS_CONFIG
    try:
        with open(_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        _RPS_CONFIG = config.get("rps", {})
    except Exception as e:
        logger.warning(f"Failed to load RPS config from {str(_THRESHOLDS_PATH)}: {e}")
        _RPS_CONFIG = {}
    return _RPS_CONFIG

def get_rps_weights():
    cfg = _load_rps_config()
    defaults = {
        "study_design": 0.35,
        "sample_size": 0.25,
        "preregistration": 0.15,
        "transparency": 0.10,
        "journal_quality": 0.10,
        "risk_of_bias": 0.05,
        "statistical_precision": 0.07,
        "multicenter": 0.06,
        "funding_transparency": 0.03,
        "population_applicability": 0.04,
    }
    configured = cfg.get("weights", {})
    if isinstance(configured, dict):
        defaults.update(configured)
    return defaults

def get_rps_norm():
    cfg = _load_rps_config()
    return cfg.get("normalization", {
        "sample_size_max": 10000,
        "sample_size_log_base": 10,
        "sample_size_log_min": 1,
        "sample_size_log_max": 10000,
        "clamp_min": 0.0,
        "clamp_max": 1.0,
    })

def get_rps_transparency_features():
    cfg = _load_rps_config()
    return cfg.get("transparency_features", {
        "open_data": True,
        "open_code": True,
        "preregistration": True,
    })

# --- End config loader ---


# Remove old hard-coded weights and bases

# For edge handling, define SKIP_SAMPLE_SIZE_STUDY_TYPES and STUDY_TYPE_BASE if referenced
SKIP_SAMPLE_SIZE_STUDY_TYPES = {
    "expert opinion",
    "editorial",
    "guideline",
    "practice guideline",
    "review",
}

STUDY_TYPE_BASE = {
    "systematic review": 0.45,
    "review": 0.40,
    "meta-analysis": 0.50,
    "cohort": 0.40,
    "randomized controlled trial": 0.48,
    "rct": 0.48,
    "case-control": 0.35,
    "cross-sectional": 0.32,
    "case report": 0.20,
    "case series": 0.25,
    "in-vitro": 0.28,
    "other": 0.30,
    "unknown": 0.30,
}

TRACKED_RPS_FEATURES: tuple[str, ...] = (
    "study_design",
    "sample_size",
    "preregistration",
    "transparency",
    "journal_quality",
    "risk_of_bias",
    "statistical_precision",
    "multicenter",
    "funding_transparency",
    "population_applicability",
)

MISSING_VALUE_STRINGS = {
    "",
    "unknown",
    "unspecified",
    "pending analysis",
    "none",
    "null",
    "n/a",
    "na",
}

TOP_TIER_JOURNAL_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"\bnew england journal of medicine\b|\bnejm\b", 1.0),
    (r"\bthe lancet\b|\blancet\b", 0.95),
    (r"\bjama\b|\bjournal of the american medical association\b", 0.95),
    (r"\bbmj\b|\bbritish medical journal\b", 0.9),
    (r"\bannals of internal medicine\b", 0.85),
    (r"\bnature medicine\b", 0.85),
)


STUDY_TYPE_KEYWORDS: list[tuple[str, str]] = [
    (r"meta[- ]analy", "meta-analysis"),
    (r"systematic review", "systematic review"),
    (r"randomized|randomised|\brct\b", "rct"),
    (r"prospective cohort", "cohort"),
    (r"retrospective cohort|cohort study", "cohort"),
    (r"case report", "case report"),
    (r"case series", "case series"),
    (r"case[- ]control", "case-control"),
    (r"cross[- ]sectional", "cross-sectional"),
    (r"review", "review"),
]


def _should_skip_sample_size_extraction(item: Dict[str, Any], study_type: str) -> bool:
    if item.get("skip_sample_size_extraction"):
        return True
    st = str(study_type or "").strip().lower()
    return st in SKIP_SAMPLE_SIZE_STUDY_TYPES

def _parse_sample_size(raw_sample: Any) -> int:
    """Robustly parse common sample-size notations.

    Prioritizes patterns like 'n=1200' (case-insensitive). Falls back
    to the first integer found in the string if a targeted pattern is
    not present. Returns 0 when parsing fails.
    """
    if isinstance(raw_sample, str):
        s = raw_sample.replace(",", "")
        # Targeted patterns for metadata fields
        for pattern in [
            r"\bn\s*[=:\-]\s*(\d+)\b",
            r"\bN\s*[=:\-]\s*(\d+)\b",
            r"\bsample size\s*[=:\-]\s*(\d+)\b",
            r"(\d+)\s*(?:patients|participants|subjects|individuals)"
        ]:
            m = re.search(pattern, s, flags=re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
        # Fallback: first integer in the string
        match = re.search(r"\d+", s)
        return int(match.group()) if match else 0
    try:
        return int(raw_sample) if raw_sample else 0
    except Exception:
        return 0


def _extract_sample_size_via_regex(text: str) -> int:
    """Dedicated lightweight regex parser for abstract text."""
    if not text:
        return 0
    
    # Clean text for parsing
    s = text.replace(",", "")
    
    # Priority patterns from Task 3
    patterns = [
        r"\bN\s*=\s*([0-9]+)\b",
        r"\bn\s*=\s*([0-9]+)\b",
        r"([0-9]+)\s*(?:patients|participants|subjects|cases|individuals|women|men)\s*(?:were enrolled|were included|participated|were studied|was)",
            r"sample size of\s*([0-9]+)\b",
            r"sample size[:\s]*([0-9]+)\b",
        r"total of\s*([0-9]+)\s*(?:patients|participants|subjects|cases|individuals)",
        r"enrolled\s*([0-9]+)\s*(?:patients|participants|subjects|cases|individuals)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, s, flags=re.IGNORECASE)
        if match:
            try:
                val = int(match.group(1))
                if 0 < val < 1000000: # Sanity check
                    return val
            except Exception:
                continue
    return 0


def get_sample_size_from_abstract(abstract_text: str, pmid: str = "unknown") -> int:
    """Extract sample size from abstract text using regex-only parsing."""
    return _extract_sample_size_via_regex(abstract_text or "")


def _infer_study_type(item: Dict[str, Any], study_type: str) -> str:
    normalized = str(study_type or "").strip().lower()
    if normalized and normalized not in {"unknown", "unspecified", "pending analysis"}:
        return normalized

    search_space = " ".join(
        str(part or "")
        for part in (
            item.get("study_type"),
            item.get("publication_types"),
            item.get("title"),
            item.get("abstract"),
        )
    ).lower()

    for pattern, mapped in STUDY_TYPE_KEYWORDS:
        if re.search(pattern, search_space, flags=re.IGNORECASE):
            return mapped

    return normalized or "unknown"


def _compute_rps_rule_based(study_type: str, sample: int, raw_sample: Any) -> float:
    if "rct" in study_type or "randomized" in study_type:
        return 0.8 if sample >= 50 else 0.7
    if "meta-analysis" in study_type or "systematic review" in study_type:
        return 0.9
    if "cohort" in study_type:
        return 0.6 if sample >= 100 else 0.5
    if "case-control" in study_type:
        return 0.5
    if "case report" in study_type or "case series" in study_type:
        return 0.2
    return 0.4


def _compute_rps_continuous(study_type: str, sample: int, include_bonus: bool) -> float:
    base = STUDY_TYPE_BASE.get(study_type, 0.40)

    bonus = 0.0
    if include_bonus and sample > 0:
        # Reviews don't get sample size bonus as they represent aggregated data
        if study_type not in ("review", "systematic review", "meta-analysis"):
            bonus = min(0.2, 0.05 * (sample ** 0.3))

    return min(0.95, base + bonus)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in MISSING_VALUE_STRINGS
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _clamp01(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        score = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, score))


def _combined_text(item: Dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key, "") or "")
        for key in (
            "title",
            "abstract",
            "fact",
            "publication_types",
            "journal",
            "journal_title",
            "source",
        )
    )


def _read_bool(item: Dict[str, Any], keys: tuple[str, ...]) -> tuple[Optional[bool], bool]:
    for key in keys:
        if key not in item or not _has_value(item.get(key)):
            continue
        value = item.get(key)
        if isinstance(value, bool):
            return value, True
        text = str(value).strip().lower()
        if text in {"true", "yes", "1", "y"}:
            return True, True
        if text in {"false", "no", "0", "n"}:
            return False, True
    return None, False


def _score_study_design(study_type: str) -> tuple[Optional[float], Any]:
    st = str(study_type or "").strip().lower()
    if not st or st in {"unknown", "unspecified", "pending analysis"}:
        return None, None
    if "meta-analysis" in st or "systematic review" in st:
        return 1.0, study_type
    if "rct" in st or "randomized" in st or "randomised" in st:
        return 0.9, study_type
    if "prospective cohort" in st:
        return 0.75, study_type
    if "cohort" in st:
        return 0.65, study_type
    if "case-control" in st:
        return 0.60, study_type
    if "cross-sectional" in st:
        return 0.50, study_type
    if "case series" in st:
        return 0.30, study_type
    if "case report" in st:
        return 0.20, study_type
    if "expert" in st or "editorial" in st or "guideline" in st:
        return 0.10, study_type
    if st == "review" or st.endswith(" review"):
        return 0.40, study_type
    if st == "other":
        return 0.40, study_type
    return 0.40, study_type


def _score_sample_size(item: Dict[str, Any], study_type: str, sample: int) -> tuple[Optional[float], Any]:
    if sample <= 0 and not _should_skip_sample_size_extraction(item, study_type):
        sample = _extract_sample_size_via_regex(_combined_text(item))
    if sample <= 0:
        return None, None

    norm = get_rps_norm()
    n_max = max(1.0, float(norm.get("sample_size_log_max", 10000)))
    score = math.log(1 + sample) / math.log(1 + n_max)
    return min(1.0, max(0.0, score)), sample


def _score_preregistration(item: Dict[str, Any], text: str) -> tuple[Optional[float], Any]:
    value, present = _read_bool(
        item,
        ("pre_registered", "preregistration", "registered", "trial_registered"),
    )
    if present:
        return (1.0 if value else 0.0), bool(value)

    if re.search(r"\b(no|not)\s+(?:pre[- ]?)?registered\b|\bno preregistration\b", text, flags=re.IGNORECASE):
        return 0.0, False
    if re.search(
        r"\bNCT\d{8}\b|clinicaltrials\.gov|prospectively registered|trial registration|pre[- ]registered",
        text,
        flags=re.IGNORECASE,
    ):
        return 1.0, True
    return None, None


def _score_transparency(item: Dict[str, Any], text: str) -> tuple[Optional[float], Any]:
    indicators: list[tuple[str, tuple[str, ...], str]] = [
        ("open_data", ("open_data", "data_available", "data_sharing"), r"open data|data (?:are|is) available|data sharing"),
        ("open_code", ("open_code", "code_available"), r"open code|code (?:is|are) available|source code"),
        ("protocol", ("protocol_published", "protocol"), r"published protocol|study protocol|protocol available"),
        ("supplementary", ("supplementary_appendix", "supplementary"), r"supplementary appendix|supplementary material"),
    ]
    values: Dict[str, bool] = {}
    for name, keys, pattern in indicators:
        value, present = _read_bool(item, keys)
        if present:
            values[name] = bool(value)
        elif re.search(pattern, text, flags=re.IGNORECASE):
            values[name] = True

    if not values:
        return None, None
    return sum(1.0 for value in values.values() if value) / len(values), values


def _score_journal_quality(item: Dict[str, Any]) -> tuple[Optional[float], Any]:
    if "journal_quality" in item and _has_value(item.get("journal_quality")):
        score = _clamp01(item.get("journal_quality"))
        if score is not None:
            return score, score

    journal_text = " ".join(
        str(item.get(key, "") or "")
        for key in ("journal", "journal_title", "source", "publication_journal")
    ).strip()
    if not journal_text:
        return None, None

    for pattern, score in TOP_TIER_JOURNAL_PATTERNS:
        if re.search(pattern, journal_text, flags=re.IGNORECASE):
            return score, journal_text
    return 0.5, journal_text


def _score_risk_of_bias(item: Dict[str, Any], text: str) -> tuple[Optional[float], Any]:
    for key in ("risk_of_bias", "risk_of_bias_score", "bias_quality"):
        if key not in item or not _has_value(item.get(key)):
            continue
        value = item.get(key)
        if isinstance(value, bool):
            return (0.2 if value else 0.8), value
        score = _clamp01(value)
        if score is not None:
            return score, value
        value_text = str(value).strip().lower()
        if "low" in value_text:
            return 0.8, value
        if "moderate" in value_text:
            return 0.5, value
        if "high" in value_text:
            return 0.2, value

    positive = [
        r"double[- ]blind",
        r"blinded",
        r"placebo[- ]controlled",
        r"intention[- ]to[- ]treat",
        r"randomized",
        r"randomised",
    ]
    negative = [
        r"small pilot",
        r"pilot study",
        r"open[- ]label",
        r"uncontrolled",
        r"retrospective only",
        r"non[- ]randomized",
        r"non[- ]randomised",
    ]
    pos_hits = sum(1 for pattern in positive if re.search(pattern, text, flags=re.IGNORECASE))
    neg_hits = sum(1 for pattern in negative if re.search(pattern, text, flags=re.IGNORECASE))
    if not pos_hits and not neg_hits:
        return None, None
    score = min(1.0, max(0.0, 0.5 + (0.12 * pos_hits) - (0.15 * neg_hits)))
    return score, {"positive": pos_hits, "negative": neg_hits}


def _score_statistical_precision(item: Dict[str, Any], text: str) -> tuple[Optional[float], Any]:
    value, present = _read_bool(
        item,
        ("p_value_reported", "confidence_interval_reported", "statistical_precision_reported"),
    )
    if present:
        return (1.0 if value else 0.0), bool(value)
    if re.search(
        r"95%\s*ci|confidence interval|\bp\s*[<=>]\s*0?\.\d+|hazard ratio|odds ratio|relative risk|risk ratio",
        text,
        flags=re.IGNORECASE,
    ):
        return 1.0, True
    return None, None


def _score_multicenter(item: Dict[str, Any], text: str) -> tuple[Optional[float], Any]:
    value, present = _read_bool(item, ("multi_center", "multicenter", "multi_site"))
    if present:
        return (1.0 if value else 0.0), bool(value)
    if re.search(r"multi[- ]center|multicenter|multi[- ]site|multiple institutions|across \d+ centers", text, flags=re.IGNORECASE):
        return 1.0, True
    if re.search(r"single[- ]center|single center", text, flags=re.IGNORECASE):
        return 0.0, False
    return None, None


def _score_funding_transparency(item: Dict[str, Any], text: str) -> tuple[Optional[float], Any]:
    funding_disclosed, present = _read_bool(item, ("funding_disclosed", "funding_statement"))
    if present:
        return (1.0 if funding_disclosed else 0.0), bool(funding_disclosed)

    industry_funded, industry_present = _read_bool(item, ("industry_funded", "industry_sponsored"))
    if industry_present:
        return (0.6 if industry_funded else 0.8), {"industry_funded": bool(industry_funded)}

    for key in ("funding", "sponsor", "sponsorship"):
        if key not in item or not _has_value(item.get(key)):
            continue
        value = str(item.get(key))
        if re.search(r"industry|pharma|biotech|sponsor", value, flags=re.IGNORECASE):
            return 0.6, value
        if re.search(r"none|no funding", value, flags=re.IGNORECASE):
            return 0.8, value
        return 1.0, value

    if re.search(r"funded by|supported by|grant|sponsored by|no funding", text, flags=re.IGNORECASE):
        score = 0.6 if re.search(r"industry|pharma|biotech|sponsored by", text, flags=re.IGNORECASE) else 1.0
        return score, True
    return None, None


def _score_population_applicability(item: Dict[str, Any], text: str) -> tuple[Optional[float], Any]:
    for key in ("population_applicability", "applicability_score", "cohort_match_score"):
        if key in item and _has_value(item.get(key)):
            score = _clamp01(item.get(key))
            if score is not None:
                return score, item.get(key)

    for key in ("population", "target_population", "study_population"):
        if key in item and _has_value(item.get(key)):
            return 0.7, item.get(key)

    if re.search(
        r"\b(?:adults|children|elderly|pregnant women|men|women|patients|participants) with\b",
        text,
        flags=re.IGNORECASE,
    ):
        return 0.7, True
    return None, None


def _prepare_rps_inputs(item: Dict[str, Any]) -> tuple[str, int]:
    study_type = _infer_study_type(item, str(item.get("study_type", "")).lower())
    raw_sample = item.get("sample_size")
    sample = _parse_sample_size(raw_sample)
    if sample <= 0 and not _should_skip_sample_size_extraction(item, study_type):
        sample = _extract_sample_size_via_regex(_combined_text(item))

    norm = get_rps_norm()
    max_allowed_sample = int(norm.get("sample_size_max", 10000))
    if sample > max_allowed_sample:
        sample = max_allowed_sample
    return study_type, sample


def _compute_rps_weighted(
    item: Dict[str, Any], study_type: str, sample: int
) -> Tuple[float, float, float, Dict[str, Any], list[str]]:
    """Compute RPS from available features, then shrink by feature coverage."""
    weights = get_rps_weights()
    norm = get_rps_norm()
    scores: Dict[str, float] = {}
    available_features: Dict[str, Any] = {}
    text = _combined_text(item)

    feature_scores = {
        "study_design": _score_study_design(study_type),
        "sample_size": _score_sample_size(item, study_type, sample),
        "preregistration": _score_preregistration(item, text),
        "transparency": _score_transparency(item, text),
        "journal_quality": _score_journal_quality(item),
        "risk_of_bias": _score_risk_of_bias(item, text),
        "statistical_precision": _score_statistical_precision(item, text),
        "multicenter": _score_multicenter(item, text),
        "funding_transparency": _score_funding_transparency(item, text),
        "population_applicability": _score_population_applicability(item, text),
    }

    for feature in TRACKED_RPS_FEATURES:
        score, value = feature_scores[feature]
        if score is None:
            continue
        scores[feature] = min(1.0, max(0.0, float(score)))
        available_features[feature] = value

    missing_features = [feature for feature in TRACKED_RPS_FEATURES if feature not in scores]
    clamp_min = float(norm.get("clamp_min", 0.0))
    clamp_max = float(norm.get("clamp_max", 1.0))
    if not scores:
        return 0.5, 0.5, 0.0, available_features, missing_features

    available_weight = sum(float(weights.get(feature, 0.0)) for feature in scores)
    if available_weight > 0:
        raw_rps = sum(float(weights.get(feature, 0.0)) * score for feature, score in scores.items()) / available_weight
    else:
        raw_rps = sum(scores.values()) / len(scores)

    coverage = len(scores) / len(TRACKED_RPS_FEATURES)
    final_rps = coverage * raw_rps + (1.0 - coverage) * 0.5

    raw_rps = min(clamp_max, max(clamp_min, raw_rps))
    final_rps = min(clamp_max, max(clamp_min, final_rps))
    return round(final_rps, 3), round(raw_rps, 3), round(coverage, 3), available_features, missing_features


def compute_rps(item: Dict[str, Any]) -> float:
    """Compute coverage-aware RPS from whatever quality features are available.

    Missing features are omitted from the raw weighted average. The final score
    is shrunk toward neutral according to feature coverage.
    """
    study_type, sample = _prepare_rps_inputs(item)
    final_rps, raw_rps, coverage, available_features, missing_features = _compute_rps_weighted(item, study_type, sample)
    try:
        logger.info(
            "compute_rps: pmid=%s study_type=%s sample_size=%s -> rps=%.3f (raw=%.3f, coverage=%.2f)",
            str(item.get("target_pmid", item.get("pmid", "-"))),
            study_type,
            str(item.get("sample_size")),
            final_rps,
            raw_rps,
            coverage,
        )
    except Exception:
        pass
    return final_rps

def compute_rps_verbose(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a dict with rps, rps_raw, rps_feature_coverage, available_features, missing_features.
    """
    study_type, sample = _prepare_rps_inputs(item)
    final_rps, raw_rps, coverage, available_features, missing_features = _compute_rps_weighted(item, study_type, sample)
    return {
        "rps": final_rps,
        "rps_raw": raw_rps,
        "rps_feature_coverage": coverage,
        "available_features": available_features,
        "missing_features": missing_features,
    }

def grade_from_rps(rps: float) -> str:
    if rps >= 0.70: return "A"
    if rps >= 0.40: return "B"
    return "C"
