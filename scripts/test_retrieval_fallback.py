#!/usr/bin/env python3
"""Test retrieval with fallback queries when initial PubMed searches return no PMIDs.

Run: python scripts/test_retrieval_fallback.py --dataset data/benchmark.json --question-id 0000
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any

from src.nodes.contrastive_retrieval import ContrastiveRetriever

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def broaden_query(q: str) -> str:
    # Remove quotes and relax AND to OR, add common ethics/reporting terms
    s = q.replace('"', '')
    s = s.replace(' AND ', ' OR ')
    suffix = ' OR ethics OR disclosure OR "medical error" OR "operative report" OR reporting'
    return s + suffix


async def run_for_question(dataset_path: Path, question_id: str) -> Dict[str, Any]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    medqa = data.get("medqa", {}) if isinstance(data, dict) else {}
    if question_id and question_id in medqa:
        item = medqa[question_id]
    else:
        first_id = next(iter(medqa))
        question_id = first_id
        item = medqa[first_id]

    question = str(item.get("question", "")).strip()
    # Simple vignette extraction similar to pico_extraction_node
    import re
    parts = re.split(r"(?:\n\s*Options:\s*\n|\n\s*A:|\n\s*A\))", question, maxsplit=1)
    vignette = parts[0].strip()

    retriever = ContrastiveRetriever()
    queries_by_candidate = await retriever.generate_queries(vignette, ["intervention"])

    results = {
        "question_id": question_id,
        "queries_generated": queries_by_candidate,
        "results_per_candidate": {},
        "total_unique": 0,
        "fallback_used": False,
        "docs": {},
    }

    dedup_seen = set()

    # Try original queries first
    for candidate, queries in queries_by_candidate.items():
        results["results_per_candidate"][candidate] = 0
        results["docs"][candidate] = []
        for p_type, qs in queries.items():
            if isinstance(qs, list):
                qs = qs[0]
            docs = await retriever.pubmed.search(qs, max_results=10)
            for d in docs:
                pmid = getattr(d, "pmid", None) or d.get("pmid")
                if pmid and pmid not in dedup_seen:
                    dedup_seen.add(pmid)
                    doc_dict = d.model_dump() if hasattr(d, "model_dump") else dict(d)
                    doc_dict.update({"candidate": candidate, "perspective": p_type})
                    results["docs"][candidate].append(doc_dict)
                    results["results_per_candidate"][candidate] += 1

    results["total_unique"] = len(dedup_seen)

    # If nothing found, run fallback broadened queries
    if results["total_unique"] == 0:
        logger.info("No PMIDs found for initial queries. Running fallback broadened queries.")
        results["fallback_used"] = True
        for candidate, queries in queries_by_candidate.items():
            for p_type, qs in queries.items():
                if isinstance(qs, list):
                    qs = qs[0]
                alt = broaden_query(qs)
                docs = await retriever.pubmed.search(alt, max_results=20)
                for d in docs:
                    pmid = getattr(d, "pmid", None) or d.get("pmid")
                    if pmid and pmid not in dedup_seen:
                        dedup_seen.add(pmid)
                        doc_dict = d.model_dump() if hasattr(d, "model_dump") else dict(d)
                        doc_dict.update({"candidate": candidate, "perspective": p_type, "query_used": alt})
                        results["docs"][candidate].append(doc_dict)
                        results["results_per_candidate"][candidate] = results["results_per_candidate"].get(candidate, 0) + 1

        results["total_unique"] = len(dedup_seen)

    return results


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/benchmark.json")
    parser.add_argument("--question-id", default="0000")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out = await run_for_question(dataset_path, args.question_id)
    out_path = Path("results") / f"retrieval_fallback_{args.question_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    logger.info("Saved retrieval fallback results to %s", out_path)


if __name__ == "__main__":
    asyncio.run(main())
