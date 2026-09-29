"""Versioned evaluation-only inference profiles."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lecture_slm.schemas.dataset import TaskType


class BaselineEvaluationProfile(BaseModel):
    """Settings for a specific baseline evaluation, separate from model defaults."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    model: str = Field(min_length=1)
    context_length: int = Field(gt=0)
    think: bool
    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    seed: int
    request_timeout_seconds: float = Field(gt=0.0)
    max_output_tokens: int = Field(gt=0)
    task_output_tokens: dict[TaskType, int] = Field(default_factory=dict)
    tag_output_tokens: dict[str, int] = Field(default_factory=dict)

    @field_validator("task_output_tokens", "tag_output_tokens")
    @classmethod
    def output_limits_must_be_positive(cls, budgets: dict[Any, int]) -> dict[Any, int]:
        if any(limit <= 0 for limit in budgets.values()):
            raise ValueError("evaluation output-token limits must be positive")
        return budgets

    def output_tokens_for(self, task: TaskType, tags: list[str]) -> int:
        """Resolve prompt-tag budgets first, then canonical task and global fallback."""

        for tag in tags:
            if tag in self.tag_output_tokens:
                return self.tag_output_tokens[tag]
        return self.task_output_tokens.get(task, self.max_output_tokens)


def load_baseline_evaluation_profile(path: Path) -> BaselineEvaluationProfile:
    """Load and validate a separate baseline evaluation profile from YAML."""

    if not path.is_file():
        raise FileNotFoundError(f"Baseline evaluation profile not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            payload: Any = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid evaluation profile YAML in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in evaluation profile {path}")
    return BaselineEvaluationProfile.model_validate(payload)
