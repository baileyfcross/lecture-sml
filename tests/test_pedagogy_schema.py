from pathlib import Path

from lecture_slm.config.loader import load_pedagogy_config

ROOT = Path(__file__).parents[1]


def test_pedagogy_config_parses() -> None:
    pedagogy = load_pedagogy_config(ROOT / "configs/pedagogy/default.yaml")
    ids = {principle.id for principle in pedagogy.principles}
    assert "scaffold_difficulty" in ids
    assert "formative_checks" in ids
