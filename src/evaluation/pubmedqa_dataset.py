
import json
import random
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterator
from pathlib import Path
import requests

from src.exceptions import IngestError

logger = logging.getLogger(__name__)

@dataclass
class PubMedQAQuestion:
    """
    Represents a single question from the PubMedQA dataset.
    """
    question_id: str
    question: str
    options: Dict[str, str]
    correct_answer: str
    correct_answer_text: str

    def to_prompt(self) -> str:
        """
        Generate a prompt formatted for the medical agent.
        """
        prompt = (
            f"Question: {self.question}\n\n"
            "Options:\n"
            f"A. {self.options.get('A', 'yes')}\n"
            f"B. {self.options.get('B', 'no')}\n"
            f"C. {self.options.get('C', 'maybe')}\n\n"
            "Answer using the scientific literature. "
            "Respond with yes, no, or maybe."
        )
        return prompt


class PubMedQADataset:
    """
    Loader for the PubMedQA benchmark dataset.
    """
    
    BENCHMARK_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"

    def __init__(self, data_path: str = "data/benchmark.json", auto_download: bool = True):
        self.data_path = Path(data_path)
        self.questions: List[PubMedQAQuestion] = []
        
        if not self.data_path.exists():
            if auto_download:
                self._download_dataset()
            else:
                raise IngestError(f"Benchmark file not found: {self.data_path}")
                
        self._load_dataset()

    def _download_dataset(self):
        """Download dataset from source."""
        logger.info(f"Downloading benchmark from {self.BENCHMARK_URL}...")
        try:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(self.BENCHMARK_URL, timeout=30)
            response.raise_for_status()

            try:
                data = response.json()
            except Exception as exc:
                raise IngestError(f"Downloaded benchmark is not valid JSON: {exc}")

            if "pubmedqa" not in data:
                if isinstance(data, dict):
                    data = {"pubmedqa": data}
                elif isinstance(data, list):
                    data = {"pubmedqa": data}

            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
                
        except Exception as e:
            raise IngestError(f"Failed to download benchmark: {e}")

    def _load_dataset(self):
        """Load and parse the dataset file."""
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if "pubmedqa" not in data:
                raise IngestError("No PubMedQA data found in benchmark file")
                
            raw_questions = data["pubmedqa"]
        
            # Handle both list and dict formats
            if isinstance(raw_questions, dict):
                items = [(qid, qdata) for qid, qdata in raw_questions.items()]
            else:
                items = [(f"pubmedqa_{i}", item) for i, item in enumerate(raw_questions)]
            
            for question_id, item in items:
                # Map answer text to options - handle both "A"/"B"/"C" and "yes"/"no"/"maybe"
                answer_raw = item.get("answer", "maybe")
                if answer_raw in ["A", "B", "C"]:
                    # Answer is already an option letter
                    correct_option = answer_raw
                    answer_text = {"A": "yes", "B": "no", "C": "maybe"}.get(answer_raw, "maybe")
                else:
                    # Answer is text like "yes", "no", "maybe"
                    answer_text = str(answer_raw).lower()
                    if answer_text == "yes":
                        correct_option = "A"
                    elif answer_text == "no":
                        correct_option = "B"
                    else:
                        correct_option = "C"
                        answer_text = "maybe"
                    
                q = PubMedQAQuestion(
                    question_id=question_id,
                    question=item.get("question", ""),
                    options={"A": "yes", "B": "no", "C": "maybe"},
                    correct_answer=correct_option,
                    correct_answer_text=answer_text
                )
                self.questions.append(q)
                
        except json.JSONDecodeError:
            raise IngestError(f"Invalid JSON in {self.data_path}")
        except Exception as e:
            if isinstance(e, IngestError):
                raise
            raise IngestError(f"Failed to load dataset: {e}")

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> PubMedQAQuestion:
        return self.questions[idx]

    def __iter__(self) -> Iterator[PubMedQAQuestion]:
        return iter(self.questions)

    def sample(self, k: int, seed: Optional[int] = None) -> List[PubMedQAQuestion]:
        """Return a random sample of questions."""
        if seed is not None:
            random.seed(seed)
        return random.sample(self.questions, min(k, len(self)))

    def get_statistics(self) -> Dict[str, int]:
        """Return dataset statistics."""
        stats = {
            "total": len(self),
            "yes": 0,
            "no": 0,
            "maybe": 0
        }
        
        for q in self.questions:
            if q.correct_answer_text in stats:
                stats[q.correct_answer_text] += 1
                
        return stats
