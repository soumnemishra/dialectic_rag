import sys
import os

# Add src to path
sys.path.append(os.getcwd())

import logging
logging.basicConfig(level=logging.INFO)

try:
    from src.orchestrator.graph import build_graph
    print("Attempting to compile graph...")
    app = build_graph()
    print("Successfully compiled the Epistemic Reasoning Pipeline graph!")
except Exception as e:
    print(f"FAILED to compile graph: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
