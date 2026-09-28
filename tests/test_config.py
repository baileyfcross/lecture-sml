from pathlib import Path

import pytest
from pydantic import ValidationError

from lecture_slm.config.loader import load_model_config

ROOT = Path(__file__).parents[1]


def test_model_config_parses() -> None:
    config = load_model_config(ROOT / "configs/models/qwen35-9b.yaml", environ={})
    assert config.ollama_name == "qwen3.5:9b"
    assert config.inference.seed == 3407


def test_ollama_host_environment_override() -> None:
    config = load_model_config(
        ROOT / "configs/models/qwen35-9b.yaml",
        environ={"OLLAMA_HOST": "http://example.test:9999/"},
    )
    assert config.inference.host == "http://example.test:9999"


def test_missing_model_config_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_model_config(tmp_path / "missing.yaml", environ={})


def test_invalid_yaml_fails_clearly(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("model: [unterminated", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_model_config(config_path, environ={})


def test_schema_invalid_yaml_fails_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-schema.yaml"
    config_path.write_text("model: {}", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_model_config(config_path, environ={})
