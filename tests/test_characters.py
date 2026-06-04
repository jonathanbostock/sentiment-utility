from question_consistency.characters import PERSONA_SUBFOLDERS, model_specs


def test_specs_include_base_and_all_personas():
    specs = model_specs()
    names = [s["name"] for s in specs]
    assert "base" in names
    assert "loving" in names
    assert "misalignment" in names
    assert len([s for s in specs if s["name"] != "base"]) == 11
    base = next(s for s in specs if s["name"] == "base")
    assert base["repo"] is None
    loving = next(s for s in specs if s["name"] == "loving")
    assert loving["repo"].endswith("personas") and loving["subfolder"] == "loving"
