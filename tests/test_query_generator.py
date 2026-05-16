from src.retrieval.query_generator import QueryGenerator, PICO


class FakeMeshExpander:
    def expand(self, term: str):
        term = term.lower()
        if "covid" in term or "sars" in term:
            return ["COVID-19", "SARS-CoV-2"]
        if "dexamethasone" in term:
            return ["dexamethasone", "corticosteroids"]
        if "mortality" in term:
            return ["mortality", "survival"]
        if "hospital" in term or "severe" in term:
            return ["hospitalized", "severe"]
        return [term]


def test_structured_query_building():
    claim = "Dexamethasone reduces mortality in hospitalized patients with severe COVID-19."
    pico = PICO(
        population="COVID-19",
        intervention="dexamethasone",
        comparator=None,
        outcome="mortality",
        intent=None,
        risk_level=None,
    )

    qg = QueryGenerator(mesh_expander=FakeMeshExpander())
    res = qg.generate(claim=claim, pico=pico, hypothesis=claim)
    supportive = res["supportive"]

    expected = '("COVID-19"[Mesh] OR COVID-19[tiab] OR SARS-CoV-2[tiab]) AND (dexamethasone[tiab] OR corticosteroids[tiab]) AND (mortality[tiab] OR survival[tiab]) AND (hospitalized[tiab] OR severe[tiab])'

    assert supportive == expected, f"Supportive query did not match.\nExpected:\n{expected}\nGot:\n{supportive}"
