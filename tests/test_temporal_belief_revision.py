import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.models.schemas import EvidenceItem, StudyMetadata
from src.models.enums import EpistemicState, EvidenceStance
from src.epistemic.temporal_belief_revision import TemporalBeliefRevision
from src.epistemic.nli_engine import NLIEngine

@pytest.fixture
def mock_nli_engine():
    engine = MagicMock(spec=NLIEngine)
    engine.classify = AsyncMock()
    return engine

@pytest.fixture
def tbr(mock_nli_engine):
    return TemporalBeliefRevision(nli_engine=mock_nli_engine)

@pytest.mark.asyncio
async def test_settled_consensus(tbr, mock_nli_engine):
    """Test that consistent entailment stays SETTLED."""
    evidences = [
        EvidenceItem(
            pmid="1", title="S1", abstract="", 
            metadata=StudyMetadata(year=2020),
            claim="Treatment X is beneficial",
            stance=EvidenceStance.SUPPORT,
            reproducibility_score=0.8,
            applicability_score=0.8
        ),
        EvidenceItem(
            pmid="2", title="S2", abstract="",
            metadata=StudyMetadata(year=2021),
            claim="Treatment X shows benefit",
            stance=EvidenceStance.SUPPORT,
            reproducibility_score=0.7,
            applicability_score=0.8
        )
    ]
    
    mock_nli_engine.classify.return_value = {"label": "ENTAILMENT", "confidence": 0.9}
    
    result = await tbr.detect_consensus_shift(evidences)
    
    assert result.state == EpistemicState.SETTLED
    assert result.temporal_shift_detected is False

@pytest.mark.asyncio
async def test_evolving_consensus(tbr, mock_nli_engine):
    """Test that new high-quality contradictions (>=2) trigger EVOLVING."""
    evidences = [
        EvidenceItem(
            pmid="1", title="S1", abstract="",
            metadata=StudyMetadata(year=2020),
            claim="Treatment X is beneficial",
            stance=EvidenceStance.SUPPORT,
            reproducibility_score=0.6,
            applicability_score=0.8
        ),
        EvidenceItem(
            pmid="2", title="S2", abstract="",
            metadata=StudyMetadata(year=2023),
            claim="Treatment X has no effect",
            stance=EvidenceStance.OPPOSE,
            reproducibility_score=0.85,
            applicability_score=0.8
        ),
        EvidenceItem(
            pmid="3", title="S3", abstract="",
            metadata=StudyMetadata(year=2024),
            claim="Treatment X is ineffective",
            stance=EvidenceStance.OPPOSE,
            reproducibility_score=0.9,
            applicability_score=0.8
        )
    ]
    
    mock_nli_engine.classify.return_value = {"label": "CONTRADICTION", "confidence": 0.95}
    
    result = await tbr.detect_consensus_shift(evidences)
    
    assert result.state == EpistemicState.EVOLVING
    assert result.temporal_shift_detected is True
    assert result.current_belief == "Treatment X is ineffective"

@pytest.mark.asyncio
async def test_conservative_consensus(tbr, mock_nli_engine):
    """Test that a single contradiction DOES NOT trigger EVOLVING (Confirmation Rule)."""
    evidences = [
        EvidenceItem(
            pmid="1", title="S1", abstract="",
            metadata=StudyMetadata(year=2020),
            claim="X works",
            stance=EvidenceStance.SUPPORT,
            reproducibility_score=0.7,
            applicability_score=0.8
        ),
        EvidenceItem(
            pmid="2", title="S2", abstract="",
            metadata=StudyMetadata(year=2024),
            claim="X fails",
            stance=EvidenceStance.OPPOSE,
            reproducibility_score=0.8,
            applicability_score=0.8
        )
    ]
    
    mock_nli_engine.classify.return_value = {"label": "CONTRADICTION", "confidence": 0.95}
    
    result = await tbr.detect_consensus_shift(evidences)
    
    # State should remain SETTLED because only 1 contradiction was found (min_confirming=2)
    assert result.state == EpistemicState.SETTLED
    assert result.temporal_shift_detected is False

@pytest.mark.asyncio
async def test_undated_evidence_does_not_become_historical_baseline(tbr, mock_nli_engine):
    """Undated evidence sorts last so it cannot corrupt the baseline claim."""
    evidences = [
        EvidenceItem(
            pmid="undated",
            title="Undated low quality",
            abstract="",
            metadata=StudyMetadata(year=None),
            claim="X fails",
            stance=EvidenceStance.OPPOSE,
            reproducibility_score=0.1,
            applicability_score=0.8
        ),
        EvidenceItem(
            pmid="dated",
            title="Dated baseline",
            abstract="",
            metadata=StudyMetadata(year=2020),
            claim="X works",
            stance=EvidenceStance.SUPPORT,
            reproducibility_score=0.8,
            applicability_score=0.8
        )
    ]

    mock_nli_engine.classify.return_value = {"label": "NEUTRAL", "confidence": 0.5}

    result = await tbr.detect_consensus_shift(evidences)

    assert result.baseline_claim == "X works"
