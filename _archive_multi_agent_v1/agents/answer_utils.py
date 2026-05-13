import re
from typing import Optional


def extract_final_answer_letter(text: Optional[str], fallback: str = "UNKNOWN") -> str:
    """Robust extraction of a final answer letter (A-D) or UNKNOWN from noisy LLM output.

    Tries several common patterns in order and returns the first match.
    """
    if not text:
        return fallback
    s = str(text)

    patterns = [
        r"\\boxed\{\s*([A-D]|UNKNOWN)\s*\}",
        r"\*\*\s*Final\s+Answer\s*[:\-]?\s*\[?([A-D]|UNKNOWN)\]?\s*\*\*",
        r"Final\s+Answer\s*[:\-]?\s*\[?([A-D]|UNKNOWN)\]?",
        r"\bAnswer\s*[:\-]?\s*([A-D])\b",
        r"\b([A-D])\b\s*(?:[:\)\.]|\-)?\s*(?:$|\n)",
    ]

    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            letter = (m.group(1) or "").strip()
            if letter:
                letter = letter.upper()
                if letter in {"A", "B", "C", "D"} or letter == "UNKNOWN":
                    return letter

    # Last-resort: look for explicit single-letter tokens at end of text
    tail = s.strip().splitlines()[-1].strip()
    if len(tail) == 1 and tail.upper() in {"A", "B", "C", "D"}:
        return tail.upper()

    return fallback
