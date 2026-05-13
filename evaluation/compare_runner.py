"""Comparative evaluation runner for MA-RAG versus the baseline RAG system."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from src.evaluation.adapter import MaRagAdapter
from src.evaluation.evaluator import PubMedQAEvaluator
from src.evaluation.metrics import compute_comparative_metrics
from src.evaluation.pubmedqa_dataset import PubMedQADataset
from src.evaluation.baseline.baseline_runner import BaselineEvaluator


async def run_comparison(
    dataset_limit: int = 50,
    sample_seed: int = 42,
    output_dir: str = "results/comparison",
) -> dict:
    """Run both systems on the same sampled questions and save comparison metrics."""
    comparison_root = Path(output_dir)
    comparison_root.mkdir(parents=True, exist_ok=True)

    dataset = PubMedQADataset()
    marage_output = comparison_root / "marage"
    baseline_output = comparison_root / "baseline"

    marage_results = await PubMedQAEvaluator(
        MaRagAdapter(),
        dataset=dataset,
        output_dir=str(marage_output),
    ).evaluate(limit=dataset_limit, sample_seed=sample_seed, dataset_name="pubmedqa_marage")

    baseline_results = await BaselineEvaluator(
        dataset=dataset,
        output_dir=str(baseline_output),
    ).evaluate(limit=dataset_limit, sample_seed=sample_seed, dataset_name="pubmedqa_baseline")

    comparative = compute_comparative_metrics(marage_results.results, baseline_results.results)
    comparative["marage_calibration_metrics"] = marage_results.calibration_metrics
    comparative["baseline_calibration_metrics"] = baseline_results.calibration_metrics

    out_path = comparison_root / f"comparison_{int(time.time())}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(comparative, handle, indent=2)

    print(json.dumps(comparative["summary"], indent=2))
    print(f"Comparison saved to {out_path}")
    return comparative


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a comparative MA-RAG vs baseline evaluation.")
    parser.add_argument("--limit", type=int, default=50, help="Number of questions to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--output-dir", type=str, default="results/comparison", help="Directory for comparison outputs.")
    args = parser.parse_args()

    asyncio.run(run_comparison(dataset_limit=args.limit, sample_seed=args.seed, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
