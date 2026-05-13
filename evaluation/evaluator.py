# FILE: src/evaluation/evaluator.py
"""
PubMedQA Evaluator for MIRAGE Benchmark.

Runs the MedicalAgent against PubMedQA questions and calculates accuracy.
Handles answer extraction from free-form LLM responses.

Example Usage:
    >>> from src.agent import MedicalAgent
    >>> from src.evaluation import PubMedQADataset, PubMedQAEvaluator
    >>> 
    >>> agent = MedicalAgent()
    >>> dataset = PubMedQADataset()
    >>> evaluator = PubMedQAEvaluator(agent, dataset)
    >>> 
    >>> results = await evaluator.evaluate(limit=50)
    >>> print(results.summary())

ENV_VARS:
    EVAL_DELAY_SECONDS: Delay between questions to respect rate limits (default: 10.0)
    EVAL_SAVE_INTERMEDIATE: Save results after each question (default: true)
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.pubmedqa_dataset import PubMedQADataset, PubMedQAQuestion
from src.evaluation.metrics import compute_calibration_metrics, compute_evaluation_invariants
from src.exceptions import TransientError

try:
    from google.api_core.exceptions import ResourceExhausted
except Exception:  # pragma: no cover - fallback when google-api-core is unavailable
    class ResourceExhausted(Exception):
        pass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Result of evaluating a single question.
    
    Attributes:
        question_id: Unique identifier for the question.
        question: The question text.
        correct_answer: Ground truth answer (yes/no/maybe).
        predicted_answer: Model's predicted answer.
        is_correct: Whether prediction matches ground truth.
        raw_response: Full LLM response text.
        sources: List of PubMed sources used.
        latency_seconds: Time taken to answer.
        error: Error message if evaluation failed.
    """
    question_id: str
    question: str
    correct_answer: str
    predicted_answer: str
    is_correct: bool
    raw_response: str = ""
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question_id": self.question_id,
            "question": self.question[:200] + "..." if len(self.question) > 200 else self.question,
            "correct_answer": self.correct_answer,
            "predicted_answer": self.predicted_answer,
            "is_correct": self.is_correct,
            "raw_response": self.raw_response, # Added for debugging
            "sources": self.sources,
            "metadata": self.metadata,
            "latency_seconds": round(self.latency_seconds, 2),
            "error": self.error,
        }


@dataclass 
class DatasetResults:
    """
    Aggregated results for the entire dataset evaluation.
    
    Attributes:
        dataset_name: Name of the evaluated dataset.
        total_questions: Total number of questions evaluated.
        correct_count: Number of correct predictions.
        accuracy: Accuracy percentage.
        results: List of individual evaluation results.
        start_time: When evaluation started.
        end_time: When evaluation ended.
        answer_distribution: Distribution of predicted answers.
    """
    dataset_name: str = "pubmedqa"
    total_questions: int = 0
    correct_count: int = 0
    accuracy: float = 0.0
    results: List[EvaluationResult] = field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    answer_distribution: Dict[str, int] = field(default_factory=dict)
    ground_truth_distribution: Dict[str, int] = field(default_factory=dict)
    safety_audit_hits: int = 0
    safety_audit_total: int = 0
    calibration_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: EvaluationResult) -> None:
        """Add a single result and update statistics."""
        self.results.append(result)
        self.total_questions = len(self.results)
        
        if result.is_correct:
            self.correct_count += 1
        
        self.accuracy = (self.correct_count / self.total_questions * 100) if self.total_questions > 0 else 0.0
        
        # Track answer distribution
        pred = result.predicted_answer.lower()
        self.answer_distribution[pred] = self.answer_distribution.get(pred, 0) + 1
        
        truth = result.correct_answer.lower()
        self.ground_truth_distribution[truth] = self.ground_truth_distribution.get(truth, 0) + 1

        safety_intercepted = result.metadata.get("safety_intercepted")
        if safety_intercepted is not None:
            self.safety_audit_total += 1
            if safety_intercepted:
                self.safety_audit_hits += 1
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 50,
            f"MIRAGE PubMedQA Evaluation Results",
            "=" * 50,
            f"Total Questions: {self.total_questions}",
            f"Correct: {self.correct_count}",
            f"Accuracy: {self.accuracy:.1f}%",
            "",
            "Predicted Answer Distribution:",
        ]
        
        for answer, count in sorted(self.answer_distribution.items()):
            pct = count / self.total_questions * 100 if self.total_questions > 0 else 0
            lines.append(f"  {answer}: {count} ({pct:.1f}%)")
        
        lines.append("")
        lines.append("Ground Truth Distribution:")
        for answer, count in sorted(self.ground_truth_distribution.items()):
            pct = count / self.total_questions * 100 if self.total_questions > 0 else 0
            lines.append(f"  {answer}: {count} ({pct:.1f}%)")
        
        if self.start_time and self.end_time:
            lines.append("")
            lines.append(f"Started: {self.start_time}")
            lines.append(f"Ended: {self.end_time}")

        if self.safety_audit_total > 0:
            hit_rate = (self.safety_audit_hits / self.safety_audit_total) * 100
            lines.append("")
            lines.append(
                f"Safety Audit Hits: {self.safety_audit_hits}/{self.safety_audit_total} ({hit_rate:.1f}%)"
            )

        if self.calibration_metrics:
            lines.append("")
            lines.append("Calibration Metrics:")
            for key, value in sorted(self.calibration_metrics.items()):
                lines.append(f"  {key}: {value}")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_name": self.dataset_name,
            "total_questions": self.total_questions,
            "correct_count": self.correct_count,
            "accuracy": round(self.accuracy, 2),
            "answer_distribution": self.answer_distribution,
            "ground_truth_distribution": self.ground_truth_distribution,
            "safety_audit_hits": self.safety_audit_hits,
            "safety_audit_total": self.safety_audit_total,
            "calibration_metrics": self.calibration_metrics,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "results": [r.to_dict() for r in self.results],
        }
    
    def save(self, path: str) -> None:
        """Save results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to {path}")


class PubMedQAEvaluator:
    """
    Evaluator for running MedicalAgent against PubMedQA.
    
    Handles:
    - Running the agent on each question
    - Extracting yes/no/maybe answers from free-form responses
    - Calculating accuracy and statistics
    - Saving intermediate and final results
    """
    
    # Patterns for extracting answers from LLM responses
    ANSWER_PATTERNS = [
        # High-priority: **Final Answer: yes/no/maybe** format (added for PubMedQA)
        r"(?i)\\boxed\{([A-D])\}",
        r"(?i)\*\*final\s+answer:\s*\[?(abstain|unknown|[A-D]|yes|no|maybe)\]?\*\*",
        r"(?i)final\s+answer:\s*\[?(abstain|unknown|[A-D]|yes|no|maybe)\]?",
        # Explicit final answer patterns
        r"(?:final\s+)?answer\s*(?:is|:)\s*[\"']?\b(abstain|unknown|yes|no|maybe|[A-D])\b[\"']?",
        r"(?:my\s+)?(?:conclusion|verdict)\s*(?:is|:)\s*[\"']?\b(abstain|unknown|yes|no|maybe|[A-D])\b[\"']?",
        r"(?:the\s+)?(?:correct\s+)?answer\s*(?:is|would be|should be)\s*[\"']?\b(abstain|unknown|yes|no|maybe|[A-D])\b[\"']?",
        # Option format - improved patterns
        r"(?:i\s+)?(?:choose|select|pick)\s*(?:option\s*)?[\"']?\b([abcd])\b[\"']?",
        r"(?:the\s+)?(?:correct\s+)?option\s+(?:is\s+)?([abcd])\b",
        r"option\s+([abcd])\s*(?:is\s+correct|is\s+the\s+answer)?",
        # Standalone at end of response
        r"(?:^|\n)\s*\**\s*(yes|no|maybe|abstain|unknown|[A-D])\s*\**\s*(?:$|\n)",
        # Bracketed answers
        r"\[(yes|no|maybe|abstain|unknown|[A-D])\]",
        r"\*\*(yes|no|maybe|abstain|unknown|[A-D])\*\*",
        r"\b(?:abstain|unknown)\b",
    ]
    
    def __init__(
        self,
        agent: Any,  # MedicalAgent type, using Any to avoid circular import
        dataset: Optional[PubMedQADataset] = None,
        delay_seconds: float = 10.0,
        save_intermediate: bool = True,
        output_dir: str = "./results",
    ) -> None:
        """
        Initialize the evaluator.
        
        Args:
            agent: MedicalAgent instance to evaluate.
            dataset: PubMedQADataset instance. If None, will load automatically.
            delay_seconds: Delay between questions to respect rate limits.
            save_intermediate: Whether to save results after each question.
            output_dir: Directory to save results.
        """
        self.agent = agent
        self.dataset = dataset or PubMedQADataset()
        self.delay_seconds = float(os.getenv("EVAL_DELAY_SECONDS", delay_seconds))
        self.save_intermediate = os.getenv("EVAL_SAVE_INTERMEDIATE", str(save_intermediate)).lower() == "true"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "PubMedQA Evaluator initialized",
            extra={
                "delay_seconds": self.delay_seconds,
                "save_intermediate": self.save_intermediate,
            }
        )
    
    def extract_answer(self, response: str) -> str:
        """
        Extract answer from LLM response.
        Uses regex to find the answer.
        """
        response_lower = response.lower()

        stripped = response.strip()
        simplified = stripped.rstrip(" .,:;!?)]}\"'")
        if simplified.lower() in {"yes", "no", "maybe", "abstain", "unknown", "a", "b", "c", "d"}:
            return simplified.lower()

        prefix_match = re.match(
            r"(?is)^\s*(yes|no|maybe|abstain|unknown|[A-D])\s*(?:$|[\[\(\{].*|[:\-].*)",
            stripped,
        )
        if prefix_match:
            return prefix_match.group(1).lower()

        # Try each pattern in order of specificity
        for pattern in self.ANSWER_PATTERNS:
            matches = re.findall(pattern, response_lower, re.IGNORECASE | re.MULTILINE)
            if matches:
                match = matches[-1]  # Take the last match (likely the final answer)
                return match.lower().strip()

        # Fallback: count occurrences and take the most frequent in last 200 chars
        last_part = response_lower[-200:]
        counts = {
            "yes": len(re.findall(r"\byes\b", last_part)),
            "no": len(re.findall(r"\bno\b", last_part)),
            "maybe": len(re.findall(r"\bmaybe\b", last_part)),
            "abstain": len(re.findall(r"\babstain\b", last_part)),
            "unknown": len(re.findall(r"\bunknown\b", last_part)),
        }

        if max(counts.values()) > 0:
            return max(counts, key=counts.get)

        logger.warning(
            "Could not extract answer from response",
            extra={"response_preview": response[:200]}
        )
        return "unknown"

    async def evaluate_question(
        self,
        question: PubMedQAQuestion,
    ) -> EvaluationResult:
        start_time = time.time()

        try:
            # Format prompt and run agent
            prompt = question.to_prompt()
            try:
                result = await self.agent.answer_query(prompt, question_id=question.question_id)
            except TypeError:
                # Backwards-compatible call for agents that accept only the
                # prompt string (tests use a simple MockMedicalAgent).
                result = await self.agent.answer_query(prompt)


            latency = time.time() - start_time

            # Extract answer from response
            predicted = self.extract_answer(result.answer)
            correct = question.correct_answer_text.lower().strip()

            # Clean direct comparison - NO OPTION MAPPING
            is_correct = (predicted == correct)

            logger.info(
                f"Evaluated {question.question_id}",
                extra={
                    "predicted": predicted,
                    "correct": correct,
                    "is_correct": is_correct,
                    "latency": round(latency, 2),
                }
            )

            return EvaluationResult(
                question_id=question.question_id,
                question=question.question,
                correct_answer=correct,
                predicted_answer=predicted,
                is_correct=is_correct,
                raw_response=result.answer,
                sources=result.sources,
                metadata=getattr(result, "metadata", {}),
                latency_seconds=latency,
            )

        except ResourceExhausted as e:
            latency = time.time() - start_time
            logger.warning(
                "Resource exhausted during evaluation for %s: %s",
                question.question_id,
                e,
            )
            raise

        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Failed to evaluate {question.question_id}", extra={"error": str(e)})
            return EvaluationResult(
                question_id=question.question_id,
                question=question.question,
                correct_answer=question.correct_answer_text.lower(),
                predicted_answer="error",
                is_correct=False,
                latency_seconds=latency,
                error=str(e),
            )

    async def evaluate(
        self,
        limit: Optional[int] = None,
        sample_seed: Optional[int] = 42,
        resume_from: Optional[str] = None,
        dataset_name: str = "pubmedqa",
    ) -> DatasetResults:
        """
        Evaluate the agent on PubMedQA dataset.

        Args:
            limit: Maximum number of questions to evaluate. None for all.
            sample_seed: Random seed for sampling if limit is set.
            resume_from: Path to previous results JSON to resume from.

        Returns:
            DatasetResults with all evaluation results and statistics.
        """
        # Get questions to evaluate first so checkpoint validation can verify
        # the resume file belongs to the same dataset slice.
        if limit:
            questions = self.dataset.sample(limit, seed=sample_seed)
        else:
            questions = list(self.dataset)

        current_question_ids = {q.question_id for q in questions}

        results = DatasetResults(
            dataset_name=dataset_name,
            start_time=datetime.now().isoformat(),
        )

        # Resume from previous run if specified
        evaluated_ids = set()
        if resume_from and Path(resume_from).exists():
            with open(resume_from, "r", encoding="utf-8") as f:
                prev_results = json.load(f)

            previous_dataset_name = prev_results.get("dataset_name")
            if previous_dataset_name and previous_dataset_name != dataset_name:
                raise ValueError(
                    f"Resume file dataset_name={previous_dataset_name!r} does not match current dataset_name={dataset_name!r}"
                )

            previous_ids = {
                r.get("question_id")
                for r in prev_results.get("results", [])
                if r.get("question_id")
            }
            invalid_ids = sorted(previous_ids - current_question_ids)
            if invalid_ids:
                raise ValueError(
                    f"Resume file {resume_from} contains {len(invalid_ids)} question_id(s) not present in the current dataset slice; refusing to mix benchmarks."
                )

            for r in prev_results.get("results", []):
                qid = r.get("question_id")
                if not qid:
                    continue
                evaluated_ids.add(qid)
                results.add_result(EvaluationResult(
                    question_id=qid,
                    question=r["question"],
                    correct_answer=r["correct_answer"],
                    predicted_answer=r["predicted_answer"],
                    is_correct=r["is_correct"],
                    sources=r.get("sources", []),
                    metadata=r.get("metadata", {}),
                    latency_seconds=r.get("latency_seconds", 0),
                    error=r.get("error"),
                ))

            if prev_results.get("start_time"):
                results.start_time = prev_results.get("start_time")
            logger.info(f"Resumed from {len(evaluated_ids)} previous results")

        # Filter out already evaluated
        questions = [q for q in questions if q.question_id not in evaluated_ids]

        total = len(questions) + len(evaluated_ids)
        logger.info(
            f"Starting evaluation",
            extra={"questions": len(questions), "already_done": len(evaluated_ids)}
        )

        # Create output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"pubmedqa_eval_{timestamp}.json"

        for i, question in enumerate(questions):
            logger.info(
                f"Progress: {len(results.results)}/{total}",
                extra={"question_id": question.question_id}
            )

            # Evaluate
            try:
                result = await self.evaluate_question(question)
            except ResourceExhausted as exc:
                cooldown_seconds = 60
                logger.warning(
                    "429 Resource Exhausted during evaluation; cooling down for %ss",
                    cooldown_seconds,
                )
                time.sleep(cooldown_seconds)
                try:
                    result = await self.evaluate_question(question)
                except ResourceExhausted as exc_again:
                    logger.error(
                        "Resource exhausted after cooldown for %s: %s",
                        question.question_id,
                        exc_again,
                    )
                    result = EvaluationResult(
                        question_id=question.question_id,
                        question=question.question,
                        correct_answer=question.correct_answer_text.lower().strip(),
                        predicted_answer="error",
                        is_correct=False,
                        latency_seconds=0.0,
                        error=str(exc_again),
                    )
            results.add_result(result)

            # Save intermediate results
            if self.save_intermediate:
                results.save(str(output_path))

            # Delay to respect rate limits
            if i < len(questions) - 1:
                await asyncio.sleep(self.delay_seconds)

        results.end_time = datetime.now().isoformat()
        results.calibration_metrics = compute_calibration_metrics(results.results)
        results.calibration_metrics["invariant_checks"] = compute_evaluation_invariants(results.results)

        broken_invariants = results.calibration_metrics["invariant_checks"].get("broken_invariants", [])
        if broken_invariants:
            logger.error("Evaluation invariant failure(s): %s", "; ".join(broken_invariants))
        else:
            logger.info("Evaluation invariants satisfied.")

        # Final save
        results.save(str(output_path))

        print(results.summary())

        return results
