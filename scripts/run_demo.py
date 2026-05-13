import asyncio
import json
from langgraph.checkpoint.memory import MemorySaver
from src.graph.workflow import build_workflow

async def run_demo():
    """Runs a sample query through the full DIALECTIC-RAG pipeline."""
    memory = MemorySaver()
    app = build_workflow(checkpointer=memory)
    
    question = "Does vitamin D supplementation reduce all-cause mortality?"
    print(f"--- RUNNING DIALECTIC-RAG ---\nQuestion: {question}\n")
    
    # Run the graph
    config = {"configurable": {"thread_id": "demo_1"}}
    state = {"original_question": question}
    
    async for event in app.astream(state, config=config):
        for node_name, node_state in event.items():
            print(f"Node completed: {node_name}")
            # Optional: print specific state updates
            if node_name == "uncertainty_propagation":
                res = node_state.get("epistemic_result")
                if res:
                    print(f"  Epistemic State: {res.state}")
                    print(f"  Belief: {res.belief}")
                    print(f"  Conflict K: {res.conflict}")

    # Final Result
    final_state = await app.aget_state(config)
    print("\n--- FINAL SYNTHESIS ---\n")
    print(final_state.values.get("candidate_answer", "No answer generated."))
    
if __name__ == "__main__":
    asyncio.run(run_demo())
