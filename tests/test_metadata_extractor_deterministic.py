import pytest

from src.epistemic.metadata_extractor import MetadataExtractor
from src.models.enums import StudyDesign


@pytest.mark.asyncio
async def test_metadata_extractor_prefers_xml_and_regex(monkeypatch):
    monkeypatch.setattr(
        "src.epistemic.metadata_extractor.ModelRegistry.get_flash_llm",
        lambda *args, **kwargs: None,
    )
    extractor = MetadataExtractor()
    article = {
        "pmid": "123",
        "abstract": (
            "We enrolled 1,234 participants. The primary outcome improved "
            "(p < 0.001; 95% CI 1.2 to 2.3). Registered as NCT12345678."
        ),
        "year": 2024,
        "publication_types": ["Journal Article", "Randomized Controlled Trial"],
        "study_types": ["Journal Article", "Randomized Controlled Trial"],
        "mesh_terms": ["Humans"],
        "source": "PubMed",
    }

    metadata = await extractor.extract(article_dict=article, pmid="123")

    assert metadata.study_design == StudyDesign.RCT
    assert metadata.sample_size == 1234
    assert metadata.has_p_value is True
    assert metadata.has_CI is True
    assert metadata.preregistration_id == "NCT12345678"
    assert metadata.year == 2024
    assert metadata.publication_types == ["Journal Article", "Randomized Controlled Trial"]
    assert metadata.mesh_terms == ["Humans"]
