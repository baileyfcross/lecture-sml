from pathlib import Path

import pytest
from pydantic import ValidationError

from lecture_slm.config.loader import InferenceConfig, load_model_config
from lecture_slm.schemas.dataset import TaskType

ROOT = Path(__file__).parents[1]


def test_model_config_parses() -> None:
    config = load_model_config(ROOT / "configs/models/qwen35-9b.yaml", environ={})
    assert config.ollama_name == "qwen3.5:9b"
    assert config.inference.seed == 3407
    assert config.inference.output_tokens_for(TaskType.LECTURE) == 2048


def test_task_output_budget_overrides_global_fallback() -> None:
    inference = InferenceConfig(
        max_output_tokens=2048,
        task_output_tokens={TaskType.EXPLANATION: 512},
    )
    assert inference.output_tokens_for(TaskType.EXPLANATION) == 512
    assert inference.output_tokens_for(TaskType.LECTURE) == 2048


def test_task_output_budget_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="must be positive"):
        InferenceConfig(task_output_tokens={TaskType.LECTURE: 0})


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
