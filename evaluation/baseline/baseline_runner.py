"""CLI runner for the baseline RAG evaluation path."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.evaluation.evaluator import PubMedQAEvaluator, DatasetResults
from src.evaluation.pubmedqa_dataset import PubMedQADataset
from src.evaluation.baseline.baseline_adapter import BaselineAdapter


class BaselineEvaluator:
    """Evaluate the baseline RAG path on PubMedQA using the shared evaluator."""

    def __init__(
        self,
        dataset: PubMedQADataset | None = None,
        output_dir: str = "./results/baseline",
        delay_seconds: float = 1.0,
    ):
        self.dataset = dataset or PubMedQADataset()
        self.output_dir = output_dir
        self.delay_seconds = delay_seconds
        self.adapter = BaselineAdapter()
        self.evaluator = PubMedQAEvaluator(
            self.adapter,
            dataset=self.dataset,
            delay_seconds=self.delay_seconds,
            output_dir=self.output_dir,
        )

    async def evaluate(
        self,
        limit: int | None = None,
        sample_seed: int | None = 42,
        dataset_name: str = "pubmedqa_baseline",
    ) -> DatasetResults:
        return await self.evaluator.evaluate(
            limit=limit,
            sample_seed=sample_seed,
            dataset_name=dataset_name,
        )


async def _run(limit: int | None, seed: int, output_dir: str) -> DatasetResults:
    evaluator = BaselineEvaluator(output_dir=output_dir)
    return await evaluator.evaluate(limit=limit, sample_seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the baseline RAG evaluation.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of questions to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset sampling.")
    parser.add_argument("--output-dir", type=str, default="./results/baseline", help="Directory for baseline results.")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results = asyncio.run(_run(args.limit, args.seed, args.output_dir))
    print(results.summary())


if __name__ == "__main__":
    main()
