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


@pytest.mark.asyncio
async def test_metadata_extractor_skips_llm_when_xml_design_is_known(monkeypatch):
    monkeypatch.setattr(
        "src.epistemic.metadata_extractor.ModelRegistry.get_flash_llm",
        lambda *args, **kwargs: object(),
    )
    extractor = MetadataExtractor()

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM fallback should not run when XML gives study design")

    monkeypatch.setattr(extractor, "_infer_design_with_llm", fail_if_called)

    metadata = await extractor.extract(
        article_dict={
            "pmid": "123",
            "abstract": "A randomized trial enrolled 200 participants.",
            "year": 2024,
            "publication_types": ["Randomized Controlled Trial"],
        },
        pmid="123",
    )

    assert metadata.study_design == StudyDesign.RCT
