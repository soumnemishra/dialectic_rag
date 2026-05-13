# FILE: src/evaluation/__init__.py
"""
MIRAGE Benchmark Evaluation Module.

Provides tools for evaluating the MedicalAgent against the MIRAGE benchmark,
specifically focused on PubMedQA dataset.

Example Usage:
    from src.evaluation import PubMedQADataset, PubMedQAEvaluator
    
    dataset = PubMedQADataset()
    evaluator = PubMedQAEvaluator(agent)
    results = await evaluator.evaluate(limit=50)
    print(results.summary())
"""
#the summary of the evaluation module for the medical agent
from src.evaluation.pubmedqa_dataset import (
    PubMedQADataset,
    PubMedQAQuestion,
)
from src.evaluation.evaluator import (
    PubMedQAEvaluator,
    EvaluationResult,
    DatasetResults,
)

__all__ = [
    "PubMedQADataset",
    "PubMedQAQuestion",
    "PubMedQAEvaluator",
    "EvaluationResult",
    "DatasetResults",
]
