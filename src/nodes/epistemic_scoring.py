from __future__ import annotations
import logging
from typing import Any
from src.models.state import GraphState
from src.models.schemas import EvidenceItem, StudyMetadata, EvidenceStance
from src.epistemic.metadata_extractor import MetadataExtractor
from src.epistemic.reproducibility_scorer import ReproducibilityScorer
from src.epistemic.applicability_scorer import ApplicabilityScorer

logger = logging.getLogger(__name__)

async def epistemic_scoring_node(state: GraphState) -> dict[str, Any]:
    """
    Node to perform two-pass epistemic scoring:
    1. Extract metadata & Reproducibility (Pass 1 & 2)
    2. Compute Applicability relative to patient PICO
    """
    retrieved_docs_dict = state.get("retrieved_docs", {})
    patient_pico = state.get("pico")
    
    # Flatten and deduplicate by PMID (just in case)
    unique_articles = {}
    for perspective, articles in retrieved_docs_dict.items():
        for art in articles:
            pmid = art.get("pmid")
            if pmid not in unique_articles:
                unique_articles[pmid] = art
    
    extractor = MetadataExtractor()
    rep_scorer = ReproducibilityScorer()
    app_scorer = ApplicabilityScorer()
    
    evidence_pool: list[EvidenceItem] = []
    
    scored_details = []
    for pmid, art in unique_articles.items():
        try:
            abstract = art.get("abstract", "")
            title = art.get("title", "")
            
            # 1. Extract Metadata
            metadata = await extractor.extract(abstract, pmid=pmid)
            # Ensure year is populated from retrieved article XML if LLM extractor didn't provide it
            if getattr(metadata, "year", None) is None:
                metadata.year = art.get("year") or art.get("publication_year")
            
            # 2. Compute Reproducibility Score with decomposition
            rps_data = rep_scorer.compute(metadata, return_components=True)
            rps = rps_data["final_rps"]
            
            scored_details.append({
                "pmid": pmid,
                "rps_decomposition": rps_data
            })
            
            # 3. Compute Applicability Score (Placeholder study PICO for now)
            from src.models.schemas import PICO
            
            # Ensure patient_pico is a PICO instance for the scorer
            if patient_pico and isinstance(patient_pico, dict):
                try:
                    patient_pico_obj = PICO(**patient_pico)
                except Exception:
                    patient_pico_obj = PICO(population="unknown", intervention="unknown", outcome="unknown")
            elif patient_pico:
                patient_pico_obj = patient_pico
            else:
                patient_pico_obj = PICO(population="unknown", intervention="unknown", outcome="unknown")

            try:
                apps = app_scorer.compute(patient_pico_obj, study_abstract=abstract)
            except Exception as e:
                logger.warning(f"Applicability scoring failed for {pmid}: {e}")
                apps = 0.5 # Neutral fallback
            
            # Diagnostic Logging
            logger.info(f"Scored PMID {pmid}: Design={metadata.study_design}, N={metadata.sample_size}, P-val={metadata.has_p_value}, RPS={rps:.2f}, Apps={apps:.2f}")

            # 4. Create Evidence Item
            item = EvidenceItem(
                pmid=pmid,
                title=title,
                abstract=abstract,
                claim=title, # Placeholder, will be refined in claim_clustering
                metadata=metadata,
                reproducibility_score=rps,
                applicability_score=apps,
                year=metadata.year,
                stance=EvidenceStance.NEUTRAL,
                nli_contradiction_prob=0.0
            )
            evidence_pool.append(item)
        except Exception as e:
            logger.error(f"Failed to score abstract {pmid}: {e}")

    # 5. Evidence Gating
    import yaml
    from pathlib import Path
    config_path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    gate = config.get("evidence_gating", {"min_reproducibility": 0.3, "min_applicability": 0.3})
    min_repro = gate.get("min_reproducibility", 0.3)
    min_applic = gate.get("min_applicability", 0.3)
    
    filtered_pool = []
    dropped_pmids = []
    for item in evidence_pool:
        if item.reproducibility_score >= min_repro:
            filtered_pool.append(item)
        else:
            reason = [f"RPS {item.reproducibility_score:.2f} < {min_repro}"]
            if item.applicability_score < min_applic:
                reason.append(f"Apps {item.applicability_score:.2f} < {min_applic}")
            logger.info(f"Dropped PMID {item.pmid}: {', '.join(reason)}")
            dropped_pmids.append({"pmid": item.pmid, "reason": reason})

    # Summary stats (Fix: Calculate only on filtered_pool as requested in audit)
    if filtered_pool:
        import statistics
        mean_rps = statistics.mean([item.reproducibility_score for item in filtered_pool])
        mean_apps = statistics.mean([item.applicability_score for item in filtered_pool])
        max_rps = max([item.reproducibility_score for item in filtered_pool])
        min_rps = min([item.reproducibility_score for item in filtered_pool])
    else:
        mean_rps = mean_apps = max_rps = min_rps = 0.0

    trace_event = {
        "node": "epistemic_scoring",
        "section": "epistemic_scoring",
        "input": {"total_articles_scored": len(unique_articles)},
        "output": {
            "passed_articles": len(filtered_pool),
            "dropped_articles": len(dropped_pmids),
            "summary_stats": {
                "mean_rps": round(mean_rps, 2),
                "max_rps": round(max_rps, 2),
                "min_rps": round(min_rps, 2),
                "mean_applicability": round(mean_apps, 2)
            }
        },
        "scored_details": scored_details,
        "dropped_details": dropped_pmids
    }

    return {
        "evidence_pool": filtered_pool,
        "trace_events": [trace_event]
    }
