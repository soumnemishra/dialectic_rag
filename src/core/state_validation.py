"""
GraphState validation and guard utilities.

This module provides runtime validation and guards for critical GraphState fields
that can be empty or malformed, preventing silent failures downstream.

Design:
  - Validators check fields at key insertion points (after major nodes)
  - Guards provide safe defaults for optional/critical fields
  - Errors are logged but non-fatal (safety-first design)
  - Can be enabled/disabled via settings
"""

import logging
from typing import Dict, Any, List, Optional
from src.state.state import GraphState

logger = logging.getLogger(__name__)


class GraphStateValidator:
    """Validates and guards critical GraphState fields."""
    
    # Fields that must exist (may be empty but should be defined)
    REQUIRED_FIELDS = {
        "original_question": str,
        "plan": list,
        "step_output": list,
        "step_docs_ids": list,
        "step_notes": list,
        "evidence_polarity": dict,
        "belief_intervals": dict,
        "trace_events": list,
    }
    
    # Fields with acceptable empty values
    OPTIONAL_FIELDS = {
        "final_answer": (str, ""),
        "candidate_answer": (str, ""),
        "temporal_conflicts": (list, []),
        "rps_scores": (list, []),
        "thesis_docs": (list, []),
        "antithesis_docs": (list, []),
        "safety_flags": (list, []),
    }
    
    # Numeric fields with valid ranges
    NUMERIC_FIELDS = {
        "retry_count": (int, 0, 10),
        "applicability_score": (float, 0.0, 1.0),
        "tcs_score": (float, 0.0, 1.0),
    }
    
    @staticmethod
    def validate_field(field_name: str, value: Any, field_type: type) -> bool:
        """
        Validate a single field.
        
        Args:
            field_name: Name of the field
            value: Value to validate
            field_type: Expected type
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(value, field_type):
            logger.warning(
                f"Invalid type for {field_name}: expected {field_type.__name__}, "
                f"got {type(value).__name__}"
            )
            return False
        return True
    
    @staticmethod
    def validate_numeric_field(field_name: str, value: Any, vtype: type, 
                               min_val: float, max_val: float) -> bool:
        """
        Validate numeric field with range.
        
        Args:
            field_name: Name of the field
            value: Value to validate
            vtype: Expected type (int or float)
            min_val: Minimum acceptable value
            max_val: Maximum acceptable value
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(value, vtype):
            logger.warning(
                f"Invalid type for {field_name}: expected {vtype.__name__}, "
                f"got {type(value).__name__}"
            )
            return False
        
        if not (min_val <= value <= max_val):
            logger.warning(
                f"Value out of range for {field_name}: {value} not in [{min_val}, {max_val}]"
            )
            return False
        
        return True
    
    @staticmethod
    def validate_belief_intervals(belief_intervals: Dict[str, Any]) -> bool:
        """
        Validate belief_intervals structure.
        
        Expected structure:
            {
                "global": {
                    "belief": float [0, 1],
                    "plausibility": float [0, 1],
                    "conflict": float [0, 1],
                    "uncertainty": float [0, 1]
                }
            }
        
        Args:
            belief_intervals: Belief intervals dict
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(belief_intervals, dict):
            logger.warning("belief_intervals is not a dict")
            return False
        
        # Must have at least 'global' key
        if "global" not in belief_intervals:
            logger.warning("belief_intervals missing 'global' key")
            return False
        
        global_belief = belief_intervals.get("global", {})
        if not isinstance(global_belief, dict):
            logger.warning("belief_intervals['global'] is not a dict")
            return False
        
        # Check required numeric fields
        required_keys = ["belief", "plausibility", "conflict", "uncertainty"]
        for key in required_keys:
            if key not in global_belief:
                logger.warning(f"belief_intervals['global'] missing '{key}'")
                return False
            
            val = global_belief[key]
            if not isinstance(val, (int, float)):
                logger.warning(f"belief_intervals['global']['{key}'] is not numeric: {type(val)}")
                return False
            
            if not (0.0 <= val <= 1.0):
                logger.warning(f"belief_intervals['global']['{key}'] out of range: {val}")
                return False
        
        return True
    
    @staticmethod
    def validate_evidence_polarity(evidence_polarity: Dict[str, Any]) -> bool:
        """
        Validate evidence_polarity structure.
        
        Expected structure:
            {
                "polarity": "strong_support" | "weak_support" | "refute" | "insufficient",
                "confidence": float [0, 1],
                "reasoning": str
            }
        
        Args:
            evidence_polarity: Evidence polarity dict
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(evidence_polarity, dict):
            logger.warning("evidence_polarity is not a dict")
            return False
        
        valid_polarities = {"strong_support", "weak_support", "refute", "insufficient"}
        polarity = evidence_polarity.get("polarity", "").lower()
        
        if polarity not in valid_polarities:
            logger.warning(f"Invalid polarity: {polarity}")
            return False
        
        confidence = evidence_polarity.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            logger.warning(f"evidence_polarity confidence not numeric: {type(confidence)}")
            return False
        
        if not (0.0 <= confidence <= 1.0):
            logger.warning(f"evidence_polarity confidence out of range: {confidence}")
            return False
        
        return True
    
    @classmethod
    def validate_state(cls, state: GraphState, 
                      level: str = "warning") -> Dict[str, List[str]]:
        """
        Validate an entire GraphState.
        
        Args:
            state: GraphState to validate
            level: "warning" (log & continue) | "strict" (raise on errors)
            
        Returns:
            Dict with keys "errors" and "warnings" listing issues
            
        Raises:
            ValueError: If level="strict" and errors found
        """
        errors = []
        warnings = []
        
        # Check required fields exist
        for field_name, expected_type in cls.REQUIRED_FIELDS.items():
            if field_name not in state:
                msg = f"Missing required field: {field_name}"
                if level == "strict":
                    errors.append(msg)
                else:
                    warnings.append(msg)
            elif field_name in ["belief_intervals", "evidence_polarity"]:
                # Special validation for complex types
                value = state[field_name]
                if field_name == "belief_intervals":
                    if not cls.validate_belief_intervals(value):
                        msg = f"Invalid belief_intervals structure"
                        errors.append(msg)
                elif field_name == "evidence_polarity":
                    if not cls.validate_evidence_polarity(value):
                        msg = f"Invalid evidence_polarity structure"
                        errors.append(msg)
            elif not cls.validate_field(field_name, state[field_name], expected_type):
                msg = f"Type mismatch for {field_name}"
                errors.append(msg)
        
        # Check numeric fields
        for field_name, (vtype, min_val, max_val) in cls.NUMERIC_FIELDS.items():
            if field_name in state:
                value = state[field_name]
                if not cls.validate_numeric_field(field_name, value, vtype, min_val, max_val):
                    msg = f"Invalid numeric field: {field_name}"
                    errors.append(msg)
        
        if level == "strict" and errors:
            raise ValueError(f"GraphState validation errors: {errors}")
        
        return {"errors": errors, "warnings": warnings}


class GraphStateGuards:
    """Provides safe defaults and guards for critical GraphState fields."""
    
    # Safe default structures
    EMPTY_BELIEF_INTERVALS = {
        "global": {
            "belief": 0.5,
            "plausibility": 0.5,
            "conflict": 0.0,
            "uncertainty": 0.0
        }
    }
    
    EMPTY_EVIDENCE_POLARITY = {
        "polarity": "insufficient",
        "confidence": 0.0,
        "reasoning": "No evidence evaluated yet"
    }
    
    EMPTY_ROUTER_OUTPUT = {
        "execution_mode": "direct_qa",
        "requires_planning": False,
        "requires_extraction": False,
        "requires_evidence_grading": False,
        "answer_policy": {},
        "execution_budget": None
    }
    
    @staticmethod
    def ensure_field(state: GraphState, field_name: str, 
                    default_value: Any, field_type: type) -> Any:
        """
        Ensure a field exists in state with a safe default.
        
        Args:
            state: GraphState
            field_name: Name of field
            default_value: Value to use if field missing or invalid
            field_type: Expected type
            
        Returns:
            Safe value for the field
        """
        if field_name not in state:
            logger.debug(f"Field {field_name} missing, using default")
            return default_value
        
        value = state.get(field_name)
        if not isinstance(value, field_type):
            logger.warning(
                f"Field {field_name} has wrong type ({type(value).__name__}), using default"
            )
            return default_value
        
        return value
    
    @staticmethod
    def guard_belief_intervals(state: GraphState) -> Dict[str, Any]:
        """
        Guard belief_intervals field with safe defaults.
        
        Args:
            state: GraphState
            
        Returns:
            Safe belief_intervals dict
        """
        value = state.get("belief_intervals")
        
        if not isinstance(value, dict):
            logger.warning("belief_intervals is not a dict, using defaults")
            return GraphStateGuards.EMPTY_BELIEF_INTERVALS.copy()
        
        # Validate structure
        if not GraphStateValidator.validate_belief_intervals(value):
            logger.warning("belief_intervals structure invalid, using defaults")
            return GraphStateGuards.EMPTY_BELIEF_INTERVALS.copy()
        
        return value
    
    @staticmethod
    def guard_evidence_polarity(state: GraphState) -> Dict[str, Any]:
        """
        Guard evidence_polarity field with safe defaults.
        
        Args:
            state: GraphState
            
        Returns:
            Safe evidence_polarity dict
        """
        value = state.get("evidence_polarity", {})
        
        if not isinstance(value, dict):
            logger.warning("evidence_polarity is not a dict, using defaults")
            return GraphStateGuards.EMPTY_EVIDENCE_POLARITY.copy()
        
        # Validate structure
        if not GraphStateValidator.validate_evidence_polarity(value):
            logger.warning("evidence_polarity structure invalid, using defaults")
            return GraphStateGuards.EMPTY_EVIDENCE_POLARITY.copy()
        
        return value
    
    @staticmethod
    def guard_router_output(state: GraphState) -> Dict[str, Any]:
        """
        Guard router_output field with safe defaults.
        
        Args:
            state: GraphState
            
        Returns:
            Safe router_output dict
        """
        value = state.get("router_output")
        
        if not isinstance(value, dict):
            logger.warning("router_output is not a dict, using defaults")
            return GraphStateGuards.EMPTY_ROUTER_OUTPUT.copy()
        
        # Validate required keys
        required_keys = {
            "execution_mode",
            "requires_planning",
            "requires_extraction",
            "requires_evidence_grading",
            "answer_policy"
        }
        
        if not all(k in value for k in required_keys):
            logger.warning("router_output missing required keys, using defaults")
            return GraphStateGuards.EMPTY_ROUTER_OUTPUT.copy()
        
        return value
    
    @staticmethod
    def safe_get_numeric(state: GraphState, field_name: str, 
                        default: float, min_val: float = 0.0, 
                        max_val: float = 1.0) -> float:
        """
        Safely get and validate a numeric field.
        
        Args:
            state: GraphState
            field_name: Name of numeric field
            default: Default value if invalid
            min_val: Minimum acceptable value
            max_val: Maximum acceptable value
            
        Returns:
            Safe numeric value
        """
        value = state.get(field_name, default)
        
        try:
            value = float(value)
            if not (min_val <= value <= max_val):
                logger.warning(
                    f"{field_name} out of range ({value}), using default ({default})"
                )
                return default
            return value
        except (TypeError, ValueError):
            logger.warning(
                f"{field_name} not numeric, using default ({default})"
            )
            return default


def ensure_graphstate_ready(state: GraphState) -> GraphState:
    """
    Ensure GraphState is ready for processing by guarding critical fields.
    
    This is a convenience function to be called at major decision points
    (before critical agents) to prevent downstream failures.
    
    Args:
        state: GraphState to guard
        
    Returns:
        Modified state with guarded fields (in-place)
    """
    # Guard critical complex fields
    if "belief_intervals" not in state or not isinstance(state.get("belief_intervals"), dict):
        state["belief_intervals"] = GraphStateGuards.guard_belief_intervals(state)
    
    if "evidence_polarity" not in state or not isinstance(state.get("evidence_polarity"), dict):
        state["evidence_polarity"] = GraphStateGuards.guard_evidence_polarity(state)
    
    if "router_output" not in state or not isinstance(state.get("router_output"), dict):
        state["router_output"] = GraphStateGuards.guard_router_output(state)
    
    # Guard critical lists
    for field in ["step_output", "step_docs_ids", "step_notes", "trace_events", "safety_flags"]:
        if field not in state or not isinstance(state.get(field), list):
            state[field] = []
    
    # Guard numeric fields
    try:
        state["retry_count"] = max(0, int(state.get("retry_count") or 0))
    except (TypeError, ValueError):
        state["retry_count"] = 0
    state["applicability_score"] = GraphStateGuards.safe_get_numeric(
        state, "applicability_score", 1.0, 0.0, 1.0
    )
    state["tcs_score"] = GraphStateGuards.safe_get_numeric(
        state, "tcs_score", 0.0, 0.0, 1.0
    )
    
    return state
