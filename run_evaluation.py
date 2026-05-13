import asyncio
import logging
import sys
import os
import argparse
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterator

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.adapter import MaRagAdapter
from src.evaluation.evaluator import PubMedQAEvaluator
from src.config import settings, configure_logging

# Configure logging
configure_logging()

# Ensure logger is defined after logging is configured
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add logging to the evaluation script
logger.info("Starting evaluation script...")

@dataclass
class MedQATestCase:
    question_id: str
    question: str
    options: Dict[str, str]
    correct_answer_text: str

    def to_prompt(self) -> str:
        valid_options = {k: v for k, v in self.options.items() if v}
        opts_str = "\n".join([f"{k}: {v}" for k, v in valid_options.items()])
        valid_keys = "/".join(valid_options.keys())
        return (
            f"{self.question}\n\n"
            f"Options:\n{opts_str}\n\n"
            "Please analyze the evidence and conclude with "
            f"'**Final Answer: [{valid_keys}]**'."
        )


class MedQADataset:
    def __init__(self, questions: List[MedQATestCase]):
        self.questions = questions

    def __iter__(self) -> Iterator[MedQATestCase]:
        return iter(self.questions)

    def sample(self, k: int, seed: Optional[int] = None) -> List[MedQATestCase]:
        if seed is not None:
            random.seed(seed)
        return random.sample(self.questions, min(k, len(self.questions)))


def _load_medqa_cases(path: str, subset: str = "all") -> List[MedQATestCase]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items: List[tuple[str, Dict[str, Any]]] = []
    if subset == "all":
        if isinstance(data, dict):
            for key, raw in data.items():
                if isinstance(raw, dict):
                    items.extend([(f"{key}_{qid}", qdata) for qid, qdata in raw.items()])
                elif isinstance(raw, list):
                    items.extend([(f"{key}_{i}", item) for i, item in enumerate(raw)])
    else:
        raw = data.get(subset, {}) if isinstance(data, dict) else {}
        if isinstance(raw, dict):
            items = [(qid, qdata) for qid, qdata in raw.items()]
        elif isinstance(raw, list):
            items = [(f"{subset}_{i}", item) for i, item in enumerate(raw)]

    cases: List[MedQATestCase] = []
    for question_id, item in items:
        question = str(item.get("question", "")).strip()
        options = item.get("options", {}) or {}
        answer = str(item.get("answer", "")).strip()

        if not question:
            continue

        case = MedQATestCase(
            question_id=str(question_id),
            question=question,
            options={
                "A": str(options.get("A", "")),
                "B": str(options.get("B", "")),
                "C": str(options.get("C", "")),
                "D": str(options.get("D", "")),
            },
            correct_answer_text=answer,
        )
        cases.append(case)

    return cases


async def main(args) -> None:
    logger.info("Initializing evaluation components...")
    agent = MaRagAdapter()
    logger.info("MaRagAdapter initialized.")

    # NOTE: If MaRagAdapter formats prompts internally, update it to handle
    # MedQA multiple-choice prompts as constructed below.
    cases = _load_medqa_cases(args.dataset, args.subset)
    dataset = MedQADataset(cases)

    evaluator = PubMedQAEvaluator(
        agent=agent,
        dataset=dataset,
        output_dir=args.output_dir,
        delay_seconds=10.0,
        save_intermediate=True
    )
    logger.info("PubMedQAEvaluator initialized.")

    logger.info("Evaluation started with limit: %s", args.limit)
    if args.resume_from:
        logger.info("Resuming from previous results: %s", args.resume_from)
    results = await evaluator.evaluate(
        limit=args.limit,
        dataset_name=args.subset,
        resume_from=args.resume_from,
    )
    logger.info("Evaluation completed. Summary: %s", results.summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples")
    parser.add_argument("--dataset", type=str, default="data/benchmark.json", help="Path to MedQA JSON file")
    parser.add_argument("--subset", type=str, default="all", help="Which dataset subset to run (e.g., medqa, pubmedqa, all)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for evaluation results")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a previous results JSON file to resume from")
    args = parser.parse_args()

    asyncio.run(main(args))
