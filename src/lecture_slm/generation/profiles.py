"""Typed loading and validation of Quick, Standard, and Deep workflow profiles."""

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lecture_slm.generation.models import GenerationProfileName
from lecture_slm.schemas.dataset import TaskType


class StageProfile(BaseModel):
    """Configuration for one planner or writer stage."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    think: bool
    thinking_reserve_tokens: int = Field(default=0, ge=0)
    temperature: float | None = Field(default=None, ge=0.0)
    context_tiers: list[int] = Field(min_length=1)
    max_output_tokens: int = Field(gt=0)
    task_output_tokens: dict[TaskType, int] = Field(default_factory=dict)
    timeout_seconds: float = Field(gt=0.0)
    retry_count: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_contexts_and_budgets(self) -> Self:
        if any(tier not in {4096, 8192, 16384, 32768} for tier in self.context_tiers):
            raise ValueError("context tiers must be selected from 4096, 8192, 16384, 32768")
        if self.context_tiers != sorted(set(self.context_tiers)):
            raise ValueError("context tiers must be unique and ascending")
        if not self.think and self.thinking_reserve_tokens:
            raise ValueError("thinking reserve must be zero when thinking is disabled")
        if any(limit <= 0 for limit in self.task_output_tokens.values()):
            raise ValueError("task-specific stage output budgets must be positive")
        return self

    def output_budget(self, task: TaskType) -> int:
        """Resolve a per-task stage budget or the profile-level fallback."""

        return self.task_output_tokens.get(task, self.max_output_tokens)

    def generation_budget(self, task: TaskType) -> int:
        """Resolve total generation allowance, including reasoning when enabled."""

        return self.output_budget(task) + (self.thinking_reserve_tokens if self.think else 0)

    def for_reassessment(self) -> Self:
        """Return deterministic, non-thinking settings for one plan reassessment."""

        return self.model_copy(
            update={
                "think": False,
                "temperature": 0,
                "thinking_reserve_tokens": 0,
                "retry_count": 0,
            }
        )


class GenerationProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: GenerationProfileName
    purpose: str = Field(min_length=1)
    planner: StageProfile
    writer: StageProfile
    review_enabled: bool = False
    fallback_to_writer: bool = False

    @model_validator(mode="after")
    def validate_pipeline_shape(self) -> Self:
        if self.name is GenerationProfileName.QUICK and self.planner.enabled:
            raise ValueError("quick profile must disable the planner")
        if (
            self.name in {GenerationProfileName.STANDARD, GenerationProfileName.DEEP}
            and not self.planner.enabled
        ):
            raise ValueError(f"{self.name.value} profile must enable the planner")
        if self.writer.think:
            raise ValueError("writer stages must disable thinking; planning is handled separately")
        if self.fallback_to_writer:
            raise ValueError("writer fallback after planner failure is not supported")
        return self


class GenerationProfiles(BaseModel):
    """Versioned collection of runtime generation profiles."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    default_profile: GenerationProfileName = GenerationProfileName.STANDARD
    estimated_fallback_tokens_per_second: float = Field(default=2.9, gt=0.0)
    context_safety_margin: float = Field(default=1.5, ge=1.0)
    profiles: dict[GenerationProfileName, GenerationProfile]

    @model_validator(mode="after")
    def all_profiles_are_present_and_named(self) -> Self:
        expected = set(GenerationProfileName)
        if set(self.profiles) != expected:
            raise ValueError("profiles must define quick, standard, and deep")
        for key, profile in self.profiles.items():
            if profile.name is not key:
                raise ValueError(f"profile key '{key.value}' does not match profile name")
        return self


def load_generation_profiles(path: Path) -> GenerationProfiles:
    """Load and validate generation workflow profiles from YAML."""

    if not path.is_file():
        raise FileNotFoundError(f"Generation profiles not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            data: Any = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid generation profile YAML in {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in generation profiles {path}")
    return GenerationProfiles.model_validate(data)
