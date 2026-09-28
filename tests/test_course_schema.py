from pathlib import Path

from lecture_slm.config.loader import load_course_config

ROOT = Path(__file__).parents[1]


def test_course_config_parses() -> None:
    course = load_course_config(ROOT / "configs/courses/example-course.yaml")
    assert course.id == "example-course"
    assert course.session.duration_minutes == 70
    assert "unexplained jargon" in course.preferences.avoid
