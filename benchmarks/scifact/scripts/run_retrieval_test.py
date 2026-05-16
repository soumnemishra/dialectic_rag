import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
import sys

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.retrieval.pipeline import RetrievalPipeline

CLAIM = "Dexamethasone reduces mortality in hospitalized patients with severe COVID-19."

OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

async def main():
    # Ensure debug artifacts are written
    os.environ.setdefault("DEBUG_MODE", "true")
    os.environ.setdefault("DEBUG_DIR", "debug")
    os.environ.setdefault("DEBUG_RETRIEVAL", "true")

    pipeline = RetrievalPipeline()

    try:
        res = await pipeline.run_for_hypothesis(claim=CLAIM, pico={}, hypothesis=CLAIM, top_k=10)
    except Exception as e:
        print("Retrieval pipeline failed:", e)
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"retrieval_test_{ts}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)

    # Build concise report
    meta = res.get("retrieval_metadata", {})
    supportive_query = meta.get("supportive_query")
    contradictory_query = meta.get("contradictory_query")
    bm25 = meta.get("bm25_scores", {})
    dense = meta.get("dense_scores", {})
    fused = meta.get("fused_scores", {})
    rerank = meta.get("rerank_scores", {})

    merged_docs = res.get("merged_ranked_docs", [])
    supporting_docs = res.get("supporting_docs", [])
    contradictory_docs = res.get("contradictory_docs", [])

    print("\n=== Retrieval Test Report ===\n")
    print("1) Supportive query:\n", supportive_query)
    print("\n2) Contradictory query:\n", contradictory_query)

    print("\n3) Top 10 ranked articles:")
    top10 = merged_docs[:10]
    for doc in top10:
        pmid = str(doc.get("pmid"))
        title = doc.get("title", "")[:200]
        year = doc.get("year") or doc.get("publication_year")
        stype = (doc.get("publication_types") or [None])[0]
        score = fused.get(pmid) if fused.get(pmid) is not None else rerank.get(pmid, 0.0)
        print(f"- PMID {pmid} | {year} | {stype} | score={score:.4f} | {title}")

    # 4) distribution of study types
    dist = {}
    for d in merged_docs:
        st = (d.get("publication_types") or ["Unknown"])[0]
        dist[st] = dist.get(st, 0) + 1
    print("\n4) Study type distribution:")
    for k, v in dist.items():
        print(f"- {k}: {v}")

    # 5) Number supportive vs contradictory retained
    print(f"\n5) Supportive docs retained: {len(supporting_docs)}")
    print(f"   Contradictory docs retained: {len(contradictory_docs)}")

    # 6) Any retrieval failures/anomalies
    anomalies = []
    # detect empty abstracts or retracted flags
    for d in merged_docs:
        if not d.get("abstract"):
            anomalies.append((d.get("pmid"), "empty_abstract"))
        if d.get("is_retracted"):
            anomalies.append((d.get("pmid"), "retracted"))

    print("\n6) Retrieval anomalies:")
    if anomalies:
        for a in anomalies:
            print(f"- PMID {a[0]}: {a[1]}")
    else:
        print("- None detected")

    print(f"\nFull result JSON saved to: {out_file}")
    print("Debug artifacts (queries/scores) should be in the debug/ folder if enabled.")

if __name__ == '__main__':
    asyncio.run(main())
