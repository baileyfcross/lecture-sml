"""Run, response, and human-review records for baseline evaluation."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lecture_slm.evaluation.prompts import ExpectedCharacteristic
from lecture_slm.evaluation.rubric import EvaluationDimension
from lecture_slm.schemas.dataset import TaskType

HumanScore = Literal[1, 2, 3, 4, 5, "N/A"]


class CompletionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class GenerationConfiguration(BaseModel):
    """Exact request-time generation settings for reproducibility."""

    model_config = ConfigDict(extra="forbid")

    stream: bool = False
    think: bool
    keep_alive: str | int
    temperature: float
    top_p: float
    seed: int
    num_ctx: int = Field(gt=0)
    num_predict: int = Field(gt=0)
    request_timeout_seconds: float = Field(default=600.0, gt=0.0)


class EvaluationTiming(BaseModel):
    """Raw Ollama nanosecond measurements plus human-readable seconds."""

    total_duration_ns: int | None = Field(default=None, ge=0)
    load_duration_ns: int | None = Field(default=None, ge=0)
    prompt_eval_duration_ns: int | None = Field(default=None, ge=0)
    eval_duration_ns: int | None = Field(default=None, ge=0)
    total_duration_seconds: float | None = Field(default=None, ge=0.0)
    load_duration_seconds: float | None = Field(default=None, ge=0.0)
    prompt_eval_duration_seconds: float | None = Field(default=None, ge=0.0)
    eval_duration_seconds: float | None = Field(default=None, ge=0.0)
    elapsed_seconds: float | None = Field(default=None, ge=0.0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    generated_tokens: int | None = Field(default=None, ge=0)


class EvaluationResult(BaseModel):
    """One success or failure record; failed attempts are never discarded."""

    model_config = ConfigDict(extra="forbid")

    prompt_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    task: TaskType
    instruction: str = Field(min_length=1)
    course_profile: str | None = None
    pedagogy_profile: str | None = None
    context: str | None = None
    source_material: str | None = None
    expected_characteristics: list[ExpectedCharacteristic]
    evaluation_dimensions: list[EvaluationDimension] = Field(min_length=1)
    response: str | None = None
    generation_configuration: GenerationConfiguration
    timing: EvaluationTiming = Field(default_factory=EvaluationTiming)
    completion_status: CompletionStatus
    error_type: str | None = None
    error_message: str | None = None
    attempt: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def completion_fields_are_consistent(self) -> Self:
        if self.completion_status is CompletionStatus.COMPLETED:
            if not self.response or self.error_type or self.error_message:
                raise ValueError("completed results require a response and no error")
        elif self.response is not None or not self.error_type or not self.error_message:
            raise ValueError("failed results require error details and no response")
        return self


class RunManifest(BaseModel):
    """Reproducibility metadata for one isolated evaluation run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str = Field(min_length=1)
    ollama_provider: str = "ollama"
    server_fingerprint: str = Field(min_length=1)
    model_configuration: dict[str, Any]
    generation_configuration: GenerationConfiguration
    git_commit: str | None = None
    evaluation_dataset_version: str = Field(min_length=1)
    evaluation_dataset_sha256: str = Field(min_length=64, max_length=64)
    profile_config_sha256: dict[str, str] = Field(default_factory=dict)
    prompt_count: int = Field(ge=1)
    random_seed: int
    project_version: str = Field(min_length=1)


class DimensionReview(BaseModel):
    """A human score, explicit N/A, or pending score for one dimension."""

    dimension: EvaluationDimension
    score: HumanScore | None = None
    notes: str | None = None
    observed_strengths: list[str] = Field(default_factory=list)
    observed_problems: list[str] = Field(default_factory=list)


class HumanReview(BaseModel):
    """Per-prompt human review saved independently from model responses."""

    prompt_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dimensions: list[DimensionReview] = Field(min_length=1)

    @model_validator(mode="after")
    def dimensions_are_unique(self) -> Self:
        dimensions = [review.dimension for review in self.dimensions]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("human review dimensions must be unique")
        return self


class EvaluationRunSummary(BaseModel):
    """Counts for a run without collapsing quality into an overall score."""

    run_id: str = Field(min_length=1)
    prompt_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
