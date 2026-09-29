"""Typed request, planning, stage, and result models for generation workflows."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from lecture_slm.evaluation.rubric import EvaluationDimension
from lecture_slm.schemas.course import CourseProfile
from lecture_slm.schemas.dataset import TaskType
from lecture_slm.schemas.pedagogy import PedagogyProfile


class GenerationProfileName(StrEnum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class GenerationStage(StrEnum):
    PREPARING = "preparing"
    PLANNING = "planning"
    WRITING = "writing"
    REVIEWING = "reviewing"
    REVISING = "revising"
    COMPLETE = "complete"
    FAILED = "failed"


class GenerationStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class SourceMaterial(BaseModel):
    """Source chunk/material prepared by a caller; retrieval is out of scope."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    section: str | None = None
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreviousCourseContext(BaseModel):
    """Caller-provided history describing what learners have already covered."""

    model_config = ConfigDict(extra="forbid")

    recently_taught_topics: list[str] = Field(default_factory=list)
    previous_lecture_summary: str | None = None
    previous_lab_summary: str | None = None
    known_terminology: list[str] = Field(default_factory=list)
    not_yet_taught: list[str] = Field(default_factory=list)
    additional_context: str | None = None


class OutputPreferences(BaseModel):
    """Presentation and length preferences for the final artifact."""

    model_config = ConfigDict(extra="forbid")

    format: str | None = None
    tone: str | None = None
    duration_minutes: int | None = Field(default=None, gt=0)
    constraints: list[str] = Field(default_factory=list)


class GenerationRequest(BaseModel):
    """Validated generation input independent of training or transport details."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    task: TaskType
    profile: GenerationProfileName = GenerationProfileName.STANDARD
    instruction: str = Field(min_length=1)
    course: CourseProfile | None = None
    pedagogy: PedagogyProfile | None = None
    source_material: list[SourceMaterial] = Field(default_factory=list)
    previous_topics: list[str] = Field(default_factory=list)
    previous_course_context: PreviousCourseContext | None = None
    output_preferences: OutputPreferences = Field(default_factory=OutputPreferences)
    metadata: dict[str, Any] = Field(default_factory=dict)
    enable_review: bool = False


class PlanSequenceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    concepts: list[str] = Field(default_factory=list)
    learner_action: str | None = None
    duration_minutes: int | None = Field(default=None, gt=0)


class PlannedExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concept: str = Field(min_length=1)
    example_description: str = Field(min_length=1)
    worked_steps: list[str] = Field(default_factory=list)


class PlannedPractice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["guided", "independent", "retrieval", "discussion", "formative"]
    description: str = Field(min_length=1)
    scaffolding: str | None = None


class TeachingPlan(BaseModel):
    """Structured instructional blueprint for the writer, never the final artifact."""

    model_config = ConfigDict(extra="forbid")

    task: TaskType
    objectives: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    prior_knowledge_connections: list[str] = Field(default_factory=list)
    sequence: list[PlanSequenceStep] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    examples: list[PlannedExample] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list)
    practice: list[PlannedPractice] = Field(default_factory=list)
    assessment_checks: list[str] = Field(default_factory=list)
    synthesis: list[str] = Field(default_factory=list)
    source_usage: list[str] = Field(default_factory=list)
    artifact_structure: list[str] = Field(default_factory=list)
    notes_for_writer: list[str] = Field(default_factory=list)


class StageTiming(BaseModel):
    duration_seconds: float = Field(ge=0.0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    generated_tokens: int | None = Field(default=None, ge=0)
    tokens_per_second: float | None = Field(default=None, ge=0.0)
    stop_reason: str | None = None
    selected_context: int | None = Field(default=None, gt=0)
    estimated_input_tokens: int | None = Field(default=None, ge=0)
    output_budget: int | None = Field(default=None, ge=0)


class StageRecord(BaseModel):
    status: GenerationStatus
    timing: StageTiming | None = None
    raw_response: str | None = None
    plan: TeachingPlan | None = None
    error_type: str | None = None
    error_message: str | None = None


class ReviewFeedback(BaseModel):
    """Prepared actionable review contract; no automatic reviewer is enabled."""

    revision_instructions: list[str] = Field(default_factory=list)
    source_consistency_notes: list[str] = Field(default_factory=list)
    dimensions_to_revisit: list[EvaluationDimension] = Field(default_factory=list)


class ProgressEvent(BaseModel):
    stage: GenerationStage
    message: str = Field(min_length=1)
    elapsed_seconds: float = Field(ge=0.0)
    generated_tokens: int | None = Field(default=None, ge=0)
    tokens_per_second: float | None = Field(default=None, ge=0.0)
    estimate_seconds_remaining: float | None = Field(default=None, ge=0.0)
    estimate_is_approximate: bool = True


class GenerationResult(BaseModel):
    """End-to-end record retaining planner and writer details independently."""

    request_id: str
    task: TaskType
    profile: GenerationProfileName
    model: str
    status: GenerationStatus
    planner_result: StageRecord | None = None
    writer_result: StageRecord | None = None
    reviewer_result: ReviewFeedback | None = None
    reviewer_error: str | None = None
    final_output: str | None = None
    model_defaults: dict[str, Any] = Field(default_factory=dict)
    profile_configuration: dict[str, Any] = Field(default_factory=dict)
    selected_contexts: dict[str, int] = Field(default_factory=dict)
    estimated_input_tokens: dict[str, int] = Field(default_factory=dict)
    timing: StageTiming
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    planner_prompt_version: str
    writer_prompt_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
