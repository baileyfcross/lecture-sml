"""Typed request, planning, stage, and result models for generation workflows."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    """Factual source material assembled by the caller for generation."""

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


class TeachingPlanBase(BaseModel):
    """Small common plan contract shared by task-specific planning schemas."""

    model_config = ConfigDict(extra="forbid")

    task: TaskType
    artifact_structure: list[str] = Field(min_length=1)
    notes_for_writer: list[str] = Field(default_factory=list)


class ExplanationPlan(TeachingPlanBase):
    task: Literal[TaskType.EXPLANATION]
    concept: str = Field(min_length=1)
    assumed_knowledge: list[str]
    explanation_sequence: list[str] = Field(min_length=1)
    example: str = Field(min_length=1)
    misconceptions: list[str]
    check_for_understanding: list[str] = Field(min_length=1)


class LecturePlan(TeachingPlanBase):
    task: Literal[TaskType.LECTURE]
    objectives: list[str] = Field(min_length=1)
    prerequisites: list[str]
    prior_knowledge_connections: list[str]
    concept_sequence: list[PlanSequenceStep] = Field(min_length=1)
    worked_examples: list[PlannedExample]
    misconceptions: list[str]
    guided_practice: list[PlannedPractice]
    independent_practice: list[PlannedPractice]
    formative_checks: list[str]
    synthesis: list[str] = Field(min_length=1)
    timing: list[str]
    source_coverage: list[str]


class SlidesPlan(TeachingPlanBase):
    task: Literal[TaskType.SLIDES]
    objectives: list[str]
    prior_knowledge: list[str]
    slide_sequence: list[PlanSequenceStep] = Field(min_length=2)
    examples: list[PlannedExample]
    exercises: list[str]
    synthesis: list[str]
    source_coverage: list[str]


class LabPlan(TeachingPlanBase):
    task: Literal[TaskType.LAB]
    objectives: list[str] = Field(min_length=1)
    prerequisites: list[str]
    prior_work: list[str]
    steps: list[PlanSequenceStep] = Field(min_length=1)
    application_tasks: list[str] = Field(min_length=1)
    checkpoints: list[str] = Field(min_length=1)
    deliverables: list[str] = Field(min_length=1)
    common_problems: list[str]


class ActivityPlan(TeachingPlanBase):
    task: Literal[TaskType.ACTIVITY]
    objectives: list[str] = Field(min_length=1)
    prior_knowledge: list[str]
    stages: list[PlanSequenceStep] = Field(min_length=1)
    timing: list[str]
    student_actions: list[str] = Field(min_length=1)
    instructor_actions: list[str]
    synthesis: list[str] = Field(min_length=1)


class InstructorGuidePlan(TeachingPlanBase):
    task: Literal[TaskType.INSTRUCTOR_GUIDE]
    objectives: list[str]
    key_explanations: list[str] = Field(min_length=1)
    misconceptions: list[str]
    worked_solutions: list[str]
    alternate_examples: list[str]
    instructor_questions: list[str]
    source_coverage: list[str]


class AssessmentPlan(TeachingPlanBase):
    task: Literal[TaskType.ASSESSMENT, TaskType.HOMEWORK]
    objectives: list[str] = Field(min_length=1)
    prerequisites: list[str]
    questions: list[str] = Field(min_length=1)
    answer_guidance: list[str]
    misconception_signals: list[str]
    scope_limits: list[str]


type TaskTeachingPlan = (
    ExplanationPlan
    | LecturePlan
    | SlidesPlan
    | LabPlan
    | ActivityPlan
    | InstructorGuidePlan
    | AssessmentPlan
)

# Compatibility name for code that only needs the shared plan fields.
TeachingPlan = TeachingPlanBase


class StageTiming(BaseModel):
    duration_seconds: float = Field(ge=0.0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    generated_tokens: int | None = Field(default=None, ge=0)
    tokens_per_second: float | None = Field(default=None, ge=0.0)
    stop_reason: str | None = None
    output_limit_reached: bool | None = None
    potentially_truncated: bool | None = None
    thinking_enabled: bool | None = None
    thinking_characters: int | None = Field(default=None, ge=0)
    thinking_token_count: int | None = Field(default=None, ge=0)
    selected_context: int | None = Field(default=None, gt=0)
    estimated_input_tokens: int | None = Field(default=None, ge=0)
    output_budget: int | None = Field(default=None, ge=0)


class StageRecord(BaseModel):
    status: GenerationStatus
    timing: StageTiming | None = None
    raw_response: str | None = None
    plan: TaskTeachingPlan | None = None
    prompt_version: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class ReviewFeedback(BaseModel):
    """Prepared actionable review contract; no automatic reviewer is enabled."""

    revision_instructions: list[str] = Field(default_factory=list)
    source_consistency_notes: list[str] = Field(default_factory=list)
    dimensions_to_revisit: list[EvaluationDimension] = Field(default_factory=list)


class GroundingDecision(StrEnum):
    PASS = "pass"  # noqa: S105
    REVISION_REQUIRED = "revision_required"


class GroundingIssueCategory(StrEnum):
    UNSUPPORTED_FACT = "unsupported_fact"
    UNSUPPORTED_FRAMING = "unsupported_framing"
    UNSUPPORTED_HISTORICAL = "unsupported_historical"
    UNSUPPORTED_SIGNIFICANCE = "unsupported_significance"
    UNSUPPORTED_CAUSAL = "unsupported_causal"
    UNSUPPORTED_APPLICATION = "unsupported_application"


class GroundingIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    excerpt: str = Field(min_length=1, alias="claim")
    category: GroundingIssueCategory = Field(alias="kind")
    reason: str = Field(min_length=1, alias="why")
    relevant_source_ids: list[str] = Field(default_factory=list, alias="sources")


class GroundingClaimStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class GroundingClaimAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    excerpt: str = Field(min_length=1, alias="claim")
    status: GroundingClaimStatus = Field(alias="status")
    source_ids: list[str] = Field(default_factory=list, alias="sources")

    @model_validator(mode="after")
    def validate_support_evidence(self) -> Self:
        if self.status is GroundingClaimStatus.SUPPORTED and not self.source_ids:
            raise ValueError("supported claims must cite at least one supplied source")
        return self


class GroundingReview(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    decision: GroundingDecision
    claim_assessments: list[GroundingClaimAssessment] = Field(min_length=1, alias="claims")
    issues: list[GroundingIssue] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list, alias="fixes")
    source_consistency_notes: list[str] = Field(default_factory=list, alias="notes")

    @model_validator(mode="after")
    def validate_decision_details(self) -> Self:
        unsupported = {
            " ".join(claim.excerpt.split()).casefold()
            for claim in self.claim_assessments
            if claim.status is GroundingClaimStatus.UNSUPPORTED
        }
        issue_excerpts = {" ".join(issue.excerpt.split()).casefold() for issue in self.issues}
        if self.decision is GroundingDecision.PASS and (self.issues or unsupported):
            raise ValueError("a passing grounding review cannot contain unsupported issues")
        if self.decision is GroundingDecision.REVISION_REQUIRED and not self.issues:
            raise ValueError("a required revision must identify at least one unsupported issue")
        if self.decision is GroundingDecision.REVISION_REQUIRED and unsupported != issue_excerpts:
            raise ValueError(
                "unsupported claim assessments must match the reported grounding issues"
            )
        if self.decision is GroundingDecision.REVISION_REQUIRED and not self.revision_instructions:
            raise ValueError("a required revision must include revision instructions")
        return self


class GroundingReviewRecord(BaseModel):
    status: GenerationStatus
    review: GroundingReview | None = None
    timing: StageTiming | None = None
    raw_response: str | None = None
    prompt_version: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class ProgressEvent(BaseModel):
    stage: GenerationStage
    message: str = Field(min_length=1)
    elapsed_seconds: float = Field(ge=0.0)
    generated_tokens: int | None = Field(default=None, ge=0)
    tokens_per_second: float | None = Field(default=None, ge=0.0)
    estimate_seconds_remaining: float | None = Field(default=None, ge=0.0)
    estimate_rate_source: Literal["fallback", "observed_previous_stage"] | None = None
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
    initial_grounding_review: GroundingReviewRecord | None = None
    revision_result: StageRecord | None = None
    final_grounding_review: GroundingReviewRecord | None = None
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
    planner_prompt_version: str | None
    writer_prompt_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
