import json
import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class TraceReporter:
    """
    Main entry point for assembling and reporting structured causal traces.
    """
    
    @staticmethod
    def assemble(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assembles scattered trace events into a hierarchical format.
        """
        events = state.get("trace_events", [])
        if not events:
            return {"error": "No trace events found in state"}

        # 1. Group events by section
        sections = {}
        for event in events:
            sec = event.get("section", "unknown")
            if sec not in sections:
                sections[sec] = []
            sections[sec].append(event)

        # 2. Extract final decision state
        final_answer = state.get("candidate_answer") or state.get("final_answer")
        predicted_letter = state.get("predicted_letter")
        
        # Extract EUS from EpistemicResult if present
        ep_res = state.get("epistemic_result")
        eus = ep_res.uncertainty if ep_res else state.get("eus_value")
        
        # 3. Build hierarchical structure
        structured = {
            "trace_id": state.get("trace_id"),
            "question_id": state.get("question_id", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
            "final_outcome": {
                "answer": final_answer,
                "letter": predicted_letter,
                "source": state.get("answer_source"),
                "eus": eus
            },
            "pipeline_stages": {
                "retrieval": sections.get("retrieval", []),
                "evidence_analysis": sections.get("node", []),
                "ds_fusion": sections.get("ds_fusion", []),
                "dialectic_gate": sections.get("dialectic_gate", []),
                "decision_governance": sections.get("decision_governance", [])
            }
        }

        # 4. Perform Causal Analysis
        analyzer = CausalAnalyzer(state, events)
        structured["causal_analysis"] = analyzer.analyze()
        
        return structured

    @staticmethod
    def save_trace(trace: Dict[str, Any], output_dir: str = "results/traces"):
        """
        Saves the structured trace to a JSON file.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        trace_id = trace.get("trace_id", "unknown")
        qid = trace.get("question_id", "unknown")
        filename = f"trace_{qid}_{trace_id[:8]}.json"
        path = os.path.join(output_dir, filename)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
            
        logger.info("Trace saved to %s", path)

    @staticmethod
    def print_summary(trace: Dict[str, Any]):
        """
        Prints a human-readable summary of the causal analysis.
        """
        if "error" in trace:
            logger.warning("Cannot print summary: %s", trace["error"])
            return

        causal = trace.get("causal_analysis", {})
        summary = causal.get("summary", {})
        outcome = trace.get("final_outcome", {})
        
        print("\n" + "="*60)
        print(f" EPISTEMIC TRACE SUMMARY | Q: {trace.get('question_id')}")
        print("="*60)
        print(f"Outcome: {outcome.get('letter', 'UNKNOWN')} ({outcome.get('source', 'unknown')})")
        print(f"EUS: {outcome.get('eus', 'N/A')}")
        print("-" * 60)
        print("Causal Influence:")
        for sev in ["DECISION_CRITICAL", "STRONG", "MODERATE", "WEAK"]:
            mods = summary.get(sev, [])
            if mods:
                print(f"  [{sev}]: {', '.join(mods)}")
        
        dead_zones = causal.get("dead_zones", [])
        if dead_zones:
            print(f"  [DEAD-ZONES]: {', '.join(dead_zones)}")
        print("="*60 + "\n")


class CausalAnalyzer:
    """
    Analyzes trace events to determine causal influence and governance impact.
    """
    
    def __init__(self, state: Dict[str, Any], events: List[Dict[str, Any]]):
        self.state = state
        self.events = events
        self.epsilon = 0.01

    def analyze(self) -> Dict[str, Any]:
        """
        Runs the full analysis suite.
        """
        module_influence = self._compute_module_influence()
        dead_zones = self._detect_dead_zones(module_influence)
        summary = self._build_summary(module_influence)
        
        return {
            "summary": summary,
            "module_details": module_influence,
            "dead_zones": dead_zones,
            "governance_lineage": self._trace_governance_lineage()
        }

    def _compute_module_influence(self) -> Dict[str, Any]:
        influence_map = {}
        # Collect all end events with snapshots
        for event in self.events:
            if event.get("event") == "end" and event.get("node"):
                node = event["node"]
                diff = event.get("data", {}).get("snapshot_diff", {})
                
                inf = {
                    "severity": self._calculate_severity(node, diff),
                    "fields_changed": diff,
                    "governance_active": False # Default
                }
                influence_map[node] = inf
        return influence_map

    def _calculate_severity(self, node: str, diff: Dict[str, Any]) -> str:
        if not diff:
            return "NONE"
        
        # Decision critical fields
        critical_fields = {"predicted_letter", "final_answer", "evidence_decision"}
        if any(f in diff for f in critical_fields):
            return "DECISION_CRITICAL"
            
        # Governance routing fields
        strong_fields = {"router_output", "plan_error", "controversy_label"}
        if any(f in diff for f in strong_fields):
            return "STRONG"
            
        # Continuous fields (Epistemic signals)
        if diff:
            return "MODERATE" if len(diff) > 1 else "WEAK"
            
        return "NONE"

    def _detect_dead_zones(self, module_influence: Dict[str, Any]) -> List[str]:
        dead_zones = []
        # A module is in a dead zone if it has influence but its fields aren't consumed
        # We'll use a heuristic for now: if severity is WEAK/MODERATE and it's not in the 
        # governance policy consumed list for active path.
        # This will be refined in Phase 3.
        for node, inf in module_influence.items():
            if inf["severity"] in ["WEAK", "MODERATE"]:
                # Check if any field in diff was used by a later node
                # For now, if it's an epistemic scorer and EUP didn't change belief much, it's a dead zone
                if node in ["rps_scoring", "applicability_scoring", "temporal_conflict"]:
                    eup_diff = module_influence.get("eup", {}).get("fields_changed", {})
                    if not eup_diff:
                        dead_zones.append(node)
        return dead_zones

    def _build_summary(self, module_influence: Dict[str, Any]) -> Dict[str, List[str]]:
        summary = {
            "DECISION_CRITICAL": [],
            "STRONG": [],
            "MODERATE": [],
            "WEAK": [],
            "NONE": []
        }
        for node, inf in module_influence.items():
            summary[inf["severity"]].append(node)
        return summary

    def _trace_governance_lineage(self) -> List[Dict[str, Any]]:
        # This will be fully implemented in Phase 3
        return []
