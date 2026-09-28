"""YAML configuration loading with environment overrides."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lecture_slm.schemas.course import CourseProfile
from lecture_slm.schemas.dataset import TaskType
from lecture_slm.schemas.pedagogy import PedagogyProfile


class InferenceConfig(BaseModel):
    """Provider settings for a model request."""

    model_config = ConfigDict(extra="forbid")

    provider: str = "ollama"
    host: str = "http://localhost:11434"
    context_length: int = Field(default=32768, gt=0)
    temperature: float = Field(default=0.5, ge=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    seed: int = 3407
    think: bool = True
    max_output_tokens: int = Field(default=2048, gt=0)
    task_output_tokens: dict[TaskType, int] = Field(default_factory=dict)
    keep_alive: str | int = "10m"
    request_timeout_seconds: float = Field(default=600.0, gt=0.0)

    @field_validator("task_output_tokens")
    @classmethod
    def task_budgets_must_be_positive(cls, budgets: dict[TaskType, int]) -> dict[TaskType, int]:
        if any(limit <= 0 for limit in budgets.values()):
            raise ValueError("task-specific output token limits must be positive")
        return budgets

    def output_tokens_for(self, task: TaskType) -> int:
        """Return a task-specific budget or the global fallback."""

        return self.task_output_tokens.get(task, self.max_output_tokens)


class FutureTrainingConfig(BaseModel):
    """Placeholder for future training metadata; no training is run by v0."""

    method: str = "qlora"
    enabled: bool = False


class ModelIdentity(BaseModel):
    """Identity and purpose of the configured model."""

    model_config = ConfigDict(extra="forbid")

    ollama_name: str = Field(min_length=1)
    family: str = Field(min_length=1)
    role: str = Field(min_length=1)


class ModelConfig(BaseModel):
    """Model and inference configuration."""

    model_config = ConfigDict(extra="forbid")

    model: ModelIdentity
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    future_training: FutureTrainingConfig = Field(default_factory=FutureTrainingConfig)

    @property
    def ollama_name(self) -> str:
        return self.model.ollama_name


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _load[ModelT: BaseModel](path: Path, schema: type[ModelT]) -> ModelT:
    return schema.model_validate(_load_yaml(path))


def load_model_config(path: Path, *, environ: dict[str, str] | None = None) -> ModelConfig:
    """Load model YAML and allow ``OLLAMA_HOST`` to override its host."""

    config = _load(path, ModelConfig)
    environment: Mapping[str, str]
    if environ is None:
        load_dotenv()
        environment = os.environ
    else:
        environment = environ
    if host := environment.get("OLLAMA_HOST"):
        config.inference.host = host.rstrip("/")
    return config


def load_course_config(path: Path) -> CourseProfile:
    """Load a course profile from YAML."""

    return _load(path, CourseProfile)


def load_pedagogy_config(path: Path) -> PedagogyProfile:
    """Load a pedagogy profile from YAML."""

    return _load(path, PedagogyProfile)
