"""Stable request and response models for the local API."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lecture_slm.generation.models import (
    GenerationProfileName,
    GenerationRequest,
    GenerationStatus,
    GroundingReviewRecord,
    OutputPreferences,
    PreviousCourseContext,
    SourceMaterial,
    StageRecord,
)
from lecture_slm.generation.profiles import GenerationProfiles
from lecture_slm.generation.service import GenerationExecution, RetrievalOptions
from lecture_slm.schemas.course import CourseProfile
from lecture_slm.schemas.dataset import TaskType
from lecture_slm.schemas.pedagogy import PedagogyProfile


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: TaskType
    profile: GenerationProfileName | None = None
    instruction: str = Field(min_length=1)
    retrieve: bool = False
    retrieval_top_k: int | None = Field(default=None, gt=0)
    source_title: str | None = None
    section: str | None = None
    knowledge_course: str | None = None
    tag: list[str] = Field(default_factory=list)
    knowledge_folder: str | None = None
    save_run: bool = False
    course: CourseProfile | None = None
    pedagogy: PedagogyProfile | None = None
    source_material: list[SourceMaterial] = Field(default_factory=list)
    previous_topics: list[str] = Field(default_factory=list)
    previous_course_context: PreviousCourseContext | None = None
    output_preferences: OutputPreferences = Field(default_factory=OutputPreferences)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_generation_request(
        self,
        default_profile: GenerationProfileName = GenerationProfileName.STANDARD,
    ) -> GenerationRequest:
        return GenerationRequest(
            task=self.task,
            profile=self.profile or default_profile,
            instruction=self.instruction,
            course=self.course,
            pedagogy=self.pedagogy,
            source_material=self.source_material,
            previous_topics=self.previous_topics,
            previous_course_context=self.previous_course_context,
            output_preferences=self.output_preferences,
            metadata=self.metadata,
        )

    def retrieval_options(self) -> RetrievalOptions | None:
        if not self.retrieve:
            return None
        return RetrievalOptions(
            enabled=True,
            top_k=self.retrieval_top_k,
            source_title=self.source_title,
            section=self.section,
            course=self.knowledge_course,
            tags=self.tag,
            folder=self.knowledge_folder,
        )


class SourceSummary(BaseModel):
    retrieved: int
    assembled: int


class GroundingSummary(BaseModel):
    reviewed: bool
    revision_performed: bool
    initial_decision: str | None = None
    final_decision: str | None = None


class TimingSummary(BaseModel):
    total_seconds: float
    planner_seconds: float | None = None
    writer_seconds: float | None = None
    initial_review_seconds: float | None = None
    revision_seconds: float | None = None
    final_review_seconds: float | None = None
    selected_contexts: dict[str, int] = Field(default_factory=dict)
    estimated_input_tokens: dict[str, int] = Field(default_factory=dict)


class GenerateResponse(BaseModel):
    request_id: str
    status: GenerationStatus
    task: TaskType
    profile: GenerationProfileName
    output: str | None = None
    errors: list[str] = Field(default_factory=list)
    grounding: GroundingSummary
    sources: SourceSummary
    timing: TimingSummary
    saved_run: str | None = None

    @classmethod
    def from_execution(cls, execution: GenerationExecution) -> "GenerateResponse":
        request = execution.request
        result = execution.result
        initial = result.initial_grounding_review
        final = result.final_grounding_review
        latest_review = final if final is not None else initial
        return cls(
            request_id=result.request_id,
            status=result.status,
            task=result.task,
            profile=result.profile,
            output=result.final_output,
            errors=result.errors,
            grounding=GroundingSummary(
                reviewed=initial is not None or final is not None,
                revision_performed=result.revision_result is not None,
                initial_decision=(
                    None
                    if initial is None or initial.review is None
                    else initial.review.decision.value
                ),
                final_decision=(
                    None
                    if latest_review is None or latest_review.review is None
                    else latest_review.review.decision.value
                ),
            ),
            sources=SourceSummary(
                retrieved=execution.retrieved_count,
                assembled=len(request.source_material),
            ),
            timing=TimingSummary(
                total_seconds=result.timing.duration_seconds,
                planner_seconds=_duration(result.planner_result),
                writer_seconds=_duration(result.writer_result),
                initial_review_seconds=_duration(initial),
                revision_seconds=_duration(result.revision_result),
                final_review_seconds=_duration(final),
                selected_contexts=result.selected_contexts,
                estimated_input_tokens=result.estimated_input_tokens,
            ),
            saved_run=(
                None
                if execution.saved_run_directory is None
                else str(execution.saved_run_directory)
            ),
        )


class ProfileSummary(BaseModel):
    id: GenerationProfileName
    purpose: str
    planner_enabled: bool
    grounding_review_for_sourced_requests: bool

    @classmethod
    def from_profiles(cls, profiles: GenerationProfiles) -> list["ProfileSummary"]:
        return [
            cls(
                id=profile.name,
                purpose=profile.purpose,
                planner_enabled=profile.planner.enabled,
                grounding_review_for_sourced_requests=profile.name
                in {GenerationProfileName.STANDARD, GenerationProfileName.DEEP},
            )
            for profile in profiles.profiles.values()
        ]


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "lecture-slm"
    generation_ready: bool
    ollama_reachable: bool
    model_available: bool
    model: str
    knowledge_index_available: bool | None = None
    knowledge_index_error: str | None = None


class StageProgress(BaseModel):
    stage: str
    message: str
    elapsed_seconds: float
    generated_tokens: int | None = None
    tokens_per_second: float | None = None
    estimate_seconds_remaining: float | None = None
    estimate_rate_source: str | None = None
    estimate_is_approximate: bool


def _duration(record: StageRecord | GroundingReviewRecord | None) -> float | None:
    if record is None or record.timing is None:
        return None
    return record.timing.duration_seconds
