import pytest

from src.models.state import GraphState


class FakeGraph:
    def __init__(self):
        self.initial_state = None

    async def ainvoke(self, initial_state):
        self.initial_state = initial_state
        return {
            **initial_state,
            "candidate_answer": "Schema-aligned answer",
            "trace_events": [
                {
                    "node": "response_generation",
                    "section": "response_generation",
                    "output": {"status": "ok"},
                }
            ],
        }


@pytest.mark.asyncio
async def test_agent_initial_state_matches_graph_state(monkeypatch):
    fake_graph = FakeGraph()
    monkeypatch.setattr("src.agent.build_workflow", lambda: fake_graph)

    from src.agent import MedicalAgent

    agent = MedicalAgent()
    answer, query_log, reasoning_steps, risk_metadata = await agent.chat(
        "Does treatment X help?",
        history=[{"role": "user", "content": "prior"}],
    )

    expected_keys = set(GraphState.__annotations__)
    actual_keys = set(fake_graph.initial_state)

    assert actual_keys == expected_keys
    assert "predicted_letter" not in actual_keys
    assert "past_exp" not in actual_keys
    assert "tcs_score" not in actual_keys
    assert answer == "Schema-aligned answer"
    assert query_log == []
    assert reasoning_steps
    assert risk_metadata["epistemic_state"] == "UNKNOWN"
