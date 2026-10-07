import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lecture_slm.config.loader import load_model_config
from lecture_slm.generation.context import select_context_tier
from lecture_slm.generation.grounding_evidence import build_evidence_ledger
from lecture_slm.generation.models import (
    EvidenceSpan,
    ExplanationPlan,
    GenerationProfileName,
    GenerationRequest,
    GenerationStage,
    GenerationStatus,
    GroundingClaimClassification,
    GroundingClaimInput,
    GroundingDecision,
    GroundingReview,
    GroundingReviewerResponse,
    GroundingSupportMethod,
    PreviousCourseContext,
    SourceMaterial,
    SourceScopeStatus,
    TeachingPlan,
)
from lecture_slm.generation.persistence import create_generation_run_directory, save_generation_run
from lecture_slm.generation.plan_schemas import plan_schema_for_task
from lecture_slm.generation.profiles import StageProfile, load_generation_profiles
from lecture_slm.generation.prompts.planner import build_planner_prompt
from lecture_slm.generation.prompts.writer import build_writer_prompt
from lecture_slm.generation.reviewer import (
    _factual_sentence_excerpts,
    _validate_claim_coverage,
    merge_grounding_review,
    prepare_grounding_claims,
)
from lecture_slm.generation.router import GenerationRouter
from lecture_slm.inference.ollama_client import ChatResponse, OllamaTimeoutError
from lecture_slm.schemas.course import CourseProfile
from lecture_slm.schemas.dataset import TaskType
from lecture_slm.schemas.pedagogy import PedagogyPrinciple, PedagogyProfile

ROOT = Path(__file__).parents[1]


def _passing_review() -> str:
    return json.dumps(
        {
            "claims": [
                {
                    "claim_id": "C001",
                    "classification": "supported",
                    "reason": "The source directly states this mapping.",
                    "evidence_ids": ["S01-E001"],
                }
            ],
        }
    )


def _default_grounding_review(user_message: str) -> str:
    claims_match = re.search(
        r"## Unresolved factual claims to adjudicate\n```json\n(.*?)\n```",
        user_message,
        flags=re.DOTALL,
    )
    sources_match = re.search(
        r"## Deterministic evidence ledger \(cite evidence_id values\)\n```json\n(.*?)\n```",
        user_message,
        flags=re.DOTALL,
    )
    if claims_match is None or sources_match is None:
        return _passing_review()
    claims = json.loads(claims_match.group(1))
    sources = json.loads(sources_match.group(1))
    evidence_id = sources[0]["evidence_id"]
    return json.dumps(
        {
            "claims": [
                {
                    "claim_id": claim["claim_id"],
                    "classification": "supported",
                    "reason": "The cited source is the supplied supporting evidence.",
                    "evidence_ids": [evidence_id],
                }
                for claim in claims
            ]
        }
    )


def _assessment(
    claim_id: str,
    text: str,
    *,
    classification: str = "supported",
    evidence_ids: list[str] | None = None,
    reason: str = "The supplied evidence supports the decision.",
) -> dict[str, Any]:
    resolved_evidence_ids = evidence_ids
    if classification == "supported":
        resolved_evidence_ids = ["S01-E001"] if evidence_ids is None else evidence_ids
    return {
        "claim_id": claim_id,
        "text": text,
        "classification": classification,
        "support_method": (
            "reviewer_entailment"
            if classification == "supported"
            else "pedagogical"
            if classification == "pedagogical"
            else "unsupported"
        ),
        "evidence_ids": resolved_evidence_ids or [],
        "reason": reason,
        "category": "unsupported_fact" if classification == "unsupported" else None,
    }


class FakeOllamaClient:
    def __init__(
        self,
        *,
        planner_failure: str | None = None,
        writer_failure: bool = False,
        writer_tokens: int = 64,
        planner_tokens: int = 80,
        planner_eval_duration_ns: int | None = 2_000_000_000,
        grounding_responses: list[str] | None = None,
        grounding_failure_at: int | None = None,
        revision_output: str = "# DNS: revised explanation\n\nA domain name maps to an IP address.",
        revision_failure: bool = False,
        source_scope_status: str = "sufficient",
        source_scope_statuses: list[str] | None = None,
        planner_response: ChatResponse | None = None,
        writer_output: str = (
            "# DNS: a concise explanation\n\nDNS maps a domain name to an IP address."
        ),
    ) -> None:
        self.planner_failure = planner_failure
        self.writer_failure = writer_failure
        self.writer_tokens = writer_tokens
        self.planner_tokens = planner_tokens
        self.planner_eval_duration_ns = planner_eval_duration_ns
        self.grounding_responses = grounding_responses or []
        self.grounding_failure_at = grounding_failure_at
        self.grounding_response_count = 0
        self.revision_output = revision_output
        self.revision_failure = revision_failure
        self.source_scope_status = source_scope_status
        self.source_scope_statuses = source_scope_statuses or []
        self.planner_response = planner_response
        self.planner_response_count = 0
        self.writer_output = writer_output
        self.requests: list[dict[str, Any]] = []
        self.timeouts: list[float] = []

    def ensure_model_available(self, model: str) -> None:
        return None

    def chat(
        self,
        *,
        model: str,
        user_message: str,
        system_message: str | None = None,
        context: str | None = None,
        options: dict[str, str | int | float | bool] | None = None,
        think: bool | None = None,
        keep_alive: str | int | None = None,
        allow_empty_content: bool = False,
        format: str | dict[str, object] | None = None,
    ) -> ChatResponse:
        self.requests.append(
            {
                "model": model,
                "user_message": user_message,
                "system_message": system_message,
                "options": options,
                "think": think,
                "format": format,
                "keep_alive": keep_alive,
            }
        )
        if format is not None:
            if format.get("title") == "GroundingReviewerResponse":
                self.grounding_response_count += 1
                if self.grounding_failure_at == self.grounding_response_count:
                    raise OllamaTimeoutError("grounding review timed out")
                content = (
                    self.grounding_responses[self.grounding_response_count - 1]
                    if self.grounding_response_count <= len(self.grounding_responses)
                    else _default_grounding_review(user_message)
                )
                return ChatResponse(
                    model=model,
                    content=content,
                    prompt_tokens=120,
                    completion_tokens=24,
                    total_duration_ns=1_000_000_000,
                    eval_duration_ns=500_000_000,
                    completion_reason="stop",
                )
            source_scope_status = (
                self.source_scope_statuses[
                    min(self.planner_response_count, len(self.source_scope_statuses) - 1)
                ]
                if self.source_scope_statuses
                else self.source_scope_status
            )
            self.planner_response_count += 1
            if self.planner_failure == "timeout":
                raise OllamaTimeoutError("planner timed out")
            if self.planner_failure == "invalid_json":
                content = "not valid JSON"
            elif self.planner_failure == "invalid_schema":
                content = json.dumps({"task": "lecture", "sequence": [{"unexpected": True}]})
            elif self.planner_response is not None:
                return self.planner_response
            elif format.get("title") == "ExplanationPlan":
                source_scope = (
                    {
                        "status": source_scope_status,
                        "supported_topics": ["DNS name resolution"],
                        "unsupported_requested_topics": (
                            ["the broader requested topic"]
                            if source_scope_status != "sufficient"
                            else []
                        ),
                        "scope_note": (
                            "Available sources cover DNS name resolution only."
                            if source_scope_status == "partial"
                            else None
                        ),
                    }
                    if "## Supplied source material" in user_message
                    else None
                )
                content = json.dumps(
                    {
                        "task": "explanation",
                        "artifact_structure": ["concept", "example", "check"],
                        "source_scope": source_scope,
                        "concept": "DNS maps names to IP addresses.",
                        "assumed_knowledge": ["websites use network addresses"],
                        "explanation_sequence": ["name", "lookup", "address"],
                        "example": "example.com resolves to a numeric IP address.",
                        "misconceptions": ["DNS is the website itself."],
                        "check_for_understanding": ["What does DNS return?"],
                        "notes_for_writer": ["Keep it concise."],
                    }
                )
            else:
                source_scope = (
                    {
                        "status": source_scope_status,
                        "supported_topics": ["DNS name resolution"],
                        "unsupported_requested_topics": (
                            ["the broader requested topic"]
                            if source_scope_status != "sufficient"
                            else []
                        ),
                        "scope_note": (
                            "Available sources cover DNS name resolution only."
                            if source_scope_status == "partial"
                            else None
                        ),
                    }
                    if "## Supplied source material" in user_message
                    else None
                )
                content = json.dumps(
                    {
                        "task": "lecture",
                        "artifact_structure": ["objectives", "example", "practice"],
                        "source_scope": source_scope,
                        "objectives": ["Explain DNS name resolution."],
                        "prerequisites": ["domain names and IP addresses"],
                        "prior_knowledge_connections": ["URLs contain domain names"],
                        "concept_sequence": [
                            {
                                "title": "Concrete lookup",
                                "purpose": "Introduce DNS through an example.",
                                "concepts": ["resolver", "IP address"],
                                "learner_action": "Trace a lookup.",
                            }
                        ],
                        "worked_examples": [],
                        "misconceptions": ["DNS is the website itself."],
                        "guided_practice": [],
                        "independent_practice": [],
                        "formative_checks": ["What does DNS return?"],
                        "synthesis": ["Connect names to network addresses."],
                        "timing": [],
                        "source_coverage": ["Use supplied source only."],
                        "notes_for_writer": ["Keep this freshman-friendly."],
                    }
                )
            return ChatResponse(
                model=model,
                content=content,
                prompt_tokens=120,
                completion_tokens=self.planner_tokens,
                total_duration_ns=3_000_000_000,
                prompt_eval_duration_ns=1_000_000_000,
                eval_duration_ns=self.planner_eval_duration_ns,
                completion_reason="stop",
                thinking_content="private planner trace for metadata test",
            )
        if system_message and "Revise the supplied complete artifact" in system_message:
            if self.revision_failure:
                raise OllamaTimeoutError("grounding revision timed out")
            return ChatResponse(
                model=model,
                content=self.revision_output,
                prompt_tokens=180,
                completion_tokens=self.writer_tokens,
                total_duration_ns=5_000_000_000,
                eval_duration_ns=4_000_000_000,
                completion_reason="stop",
            )
        if self.writer_failure:
            raise OllamaTimeoutError("writer timed out")
        return ChatResponse(
            model=model,
            content=self.writer_output,
            prompt_tokens=180,
            completion_tokens=self.writer_tokens,
            total_duration_ns=5_000_000_000,
            prompt_eval_duration_ns=1_000_000_000,
            eval_duration_ns=4_000_000_000,
            completion_reason="stop",
        )


@pytest.fixture
def configs() -> tuple[Any, Any]:
    model = load_model_config(
        ROOT / "configs/models/qwen35-9b.yaml",
        environ={"OLLAMA_HOST": "http://generation.test:11434"},
    )
    profiles = load_generation_profiles(ROOT / "configs/generation/profiles.yaml")
    return model, profiles


def sample_request(
    profile: GenerationProfileName,
    task: TaskType = TaskType.LECTURE,
) -> GenerationRequest:
    return GenerationRequest(
        task=task,
        profile=profile,
        instruction="Create a short introduction to DNS.",
        course=CourseProfile(id="cis-intro", name="CIS Intro", level="freshman"),
        pedagogy=PedagogyProfile(
            id="default",
            version="1",
            principles=[
                PedagogyPrinciple(
                    id="worked_examples",
                    name="Worked examples",
                    description="Use one example before independent practice.",
                )
            ],
        ),
        source_material=[
            SourceMaterial(
                source_id="dns-notes",
                title="DNS notes",
                section="Resolution",
                text="DNS maps a human-readable domain name to an IP address.",
            )
        ],
        previous_topics=["URLs and domain names"],
        previous_course_context=PreviousCourseContext(
            recently_taught_topics=["URLs"],
            not_yet_taught=["recursive DNS lookup details"],
        ),
    )


def make_pipeline(model: Any, profiles: Any, fake: FakeOllamaClient) -> GenerationRouter:
    def client_factory(host: str, timeout: float) -> FakeOllamaClient:
        fake.timeouts.append(timeout)
        return fake

    return GenerationRouter(
        model_config=model,
        profiles=profiles,
        client_factory=client_factory,  # type: ignore[arg-type]
    )


def test_quick_pipeline_is_writer_only(configs: tuple[Any, Any]) -> None:
    model, profiles = configs
    fake = FakeOllamaClient()
    events = []
    result = make_pipeline(model, profiles, fake).route(
        sample_request(GenerationProfileName.QUICK),
        on_progress=events.append,
    )

    assert result.status is GenerationStatus.COMPLETED
    assert result.planner_result is None
    assert result.writer_result is not None
    assert len(fake.requests) == 1
    assert fake.requests[0]["format"] is None
    assert fake.requests[0]["think"] is False
    assert fake.requests[0]["options"]["num_predict"] == 512
    assert fake.timeouts == [
        30.0,
        profiles.profiles[GenerationProfileName.QUICK].writer.timeout_seconds,
    ]
    assert [event.stage for event in events] == [
        GenerationStage.PREPARING,
        GenerationStage.WRITING,
        GenerationStage.WRITING,
        GenerationStage.COMPLETE,
    ]
    assert events[1].estimate_is_approximate is True
    assert events[1].estimate_seconds_remaining is not None
    assert events[1].estimate_rate_source == "fallback"
    assert events[1].tokens_per_second == profiles.estimated_fallback_tokens_per_second


def test_standard_calls_planner_then_writer_with_structured_context(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    fake = FakeOllamaClient()
    events = []
    result = make_pipeline(model, profiles, fake).route(
        sample_request(GenerationProfileName.STANDARD, TaskType.EXPLANATION),
        on_progress=events.append,
    )

    assert result.status is GenerationStatus.COMPLETED, result.errors
    assert result.planner_result is not None and result.planner_result.plan is not None
    assert len(fake.requests) == 3
    planner_request, writer_request, grounding_request = fake.requests
    assert planner_request["format"] is not None
    assert planner_request["format"]["title"] == "ExplanationPlan"
    assert planner_request["think"] is False
    assert planner_request["options"]["temperature"] == 0.0
    assert "Do not explain your reasoning" in planner_request["system_message"]
    assert "step by step" not in planner_request["system_message"].lower()
    standard_planner_budget = profiles.profiles[
        GenerationProfileName.STANDARD
    ].planner.output_budget(TaskType.EXPLANATION)
    assert planner_request["options"]["num_predict"] == standard_planner_budget
    assert result.planner_result.timing.output_budget == standard_planner_budget
    assert result.planner_result.timing.thinking_reserve_tokens == 0
    assert result.planner_result.timing.generation_budget == standard_planner_budget
    assert profiles.profiles[GenerationProfileName.STANDARD].planner.context_tiers == [
        4096,
        8192,
    ]
    assert writer_request["format"] is None
    assert writer_request["think"] is False
    assert writer_request["options"]["num_predict"] == profiles.profiles[
        GenerationProfileName.STANDARD
    ].writer.output_budget(TaskType.EXPLANATION)
    standard_profile = profiles.profiles[GenerationProfileName.STANDARD]
    assert fake.timeouts == [
        30.0,
        standard_profile.planner.timeout_seconds,
        standard_profile.writer.timeout_seconds,
        standard_profile.writer.timeout_seconds,
    ]
    assert "Supplied source material" in planner_request["user_message"]
    assert "Previous course context" in planner_request["user_message"]
    assert "Authoritative user request" in writer_request["user_message"]
    assert "Teaching plan to follow" in writer_request["user_message"]
    assert "Supplied source material" in writer_request["user_message"]
    assert '"source_scope"' in writer_request["user_message"]
    assert '"status": "sufficient"' in writer_request["user_message"]
    assert "when partial, stay within supported_topics" in writer_request["system_message"]
    assert grounding_request["format"]["title"] == "GroundingReviewerResponse"
    assert grounding_request["think"] is False
    assert grounding_request["options"]["temperature"] == 0
    assert "Deterministic evidence ledger" in grounding_request["user_message"]
    assert result.initial_grounding_review is not None
    assert result.initial_grounding_review.review is not None
    assert result.initial_grounding_review.review.decision is GroundingDecision.PASS
    assert result.final_output == result.writer_result.raw_response
    assert result.planner_result.timing.generated_tokens == 80
    assert result.planner_result.timing.thinking_enabled is False
    assert result.planner_result.timing.thinking_characters is None
    assert result.planner_result.timing.potentially_truncated is False
    assert result.writer_result is not None
    assert result.writer_result.timing.generated_tokens == 64
    assert result.writer_result.timing.potentially_truncated is False
    assert result.metadata["planner_prompt_version"] == "planner-standard-v4"
    assert result.metadata["writer_prompt_version"] == "writer-v8"
    assert [event.stage for event in events] == [
        GenerationStage.PREPARING,
        GenerationStage.PLANNING,
        GenerationStage.PLANNING,
        GenerationStage.WRITING,
        GenerationStage.WRITING,
        GenerationStage.REVIEWING,
        GenerationStage.REVIEWING,
        GenerationStage.COMPLETE,
    ]
    assert events[1].estimate_rate_source == "fallback"
    assert events[1].estimate_seconds_remaining == pytest.approx(
        profiles.profiles[GenerationProfileName.STANDARD].planner.output_budget(
            TaskType.EXPLANATION
        )
        / profiles.estimated_fallback_tokens_per_second
    )


def test_writer_estimate_uses_successful_planner_rate(configs: tuple[Any, Any]) -> None:
    model, profiles = configs
    fake = FakeOllamaClient(planner_tokens=93)
    events = []
    result = make_pipeline(model, profiles, fake).route(
        sample_request(GenerationProfileName.STANDARD, TaskType.EXPLANATION),
        on_progress=events.append,
    )

    assert result.status is GenerationStatus.COMPLETED
    writer_start = next(
        event
        for event in events
        if event.stage is GenerationStage.WRITING and event.estimate_seconds_remaining is not None
    )
    output_budget = profiles.profiles[GenerationProfileName.STANDARD].writer.output_budget(
        TaskType.EXPLANATION
    )
    assert writer_start.tokens_per_second == pytest.approx(46.5)
    assert writer_start.estimate_seconds_remaining == pytest.approx(output_budget / 46.5)
    assert writer_start.estimate_rate_source == "observed_previous_stage"


@pytest.mark.parametrize("planner_eval_duration_ns", [0, None])
def test_writer_estimate_falls_back_without_usable_planner_rate(
    configs: tuple[Any, Any],
    planner_eval_duration_ns: int | None,
) -> None:
    model, profiles = configs
    fake = FakeOllamaClient(planner_eval_duration_ns=planner_eval_duration_ns)
    events = []
    result = make_pipeline(model, profiles, fake).route(
        sample_request(GenerationProfileName.STANDARD),
        on_progress=events.append,
    )

    assert result.status is GenerationStatus.COMPLETED
    writer_start = next(
        event
        for event in events
        if event.stage is GenerationStage.WRITING and event.estimate_seconds_remaining is not None
    )
    assert writer_start.tokens_per_second == profiles.estimated_fallback_tokens_per_second
    assert writer_start.estimate_rate_source == "fallback"


def test_failed_planner_rate_is_not_reused(configs: tuple[Any, Any]) -> None:
    model, profiles = configs
    fake = FakeOllamaClient(planner_failure="invalid_schema", planner_tokens=93)
    events = []
    result = make_pipeline(model, profiles, fake).route(
        sample_request(GenerationProfileName.STANDARD),
        on_progress=events.append,
    )

    assert result.status is GenerationStatus.FAILED
    assert result.planner_result is not None
    assert result.planner_result.status is GenerationStatus.FAILED
    assert result.planner_result.timing is not None
    assert result.planner_result.timing.tokens_per_second == pytest.approx(46.5)
    assert not any(event.stage is GenerationStage.WRITING for event in events)


def test_explanation_uses_compact_task_specific_plan(configs: tuple[Any, Any]) -> None:
    model, profiles = configs
    fake = FakeOllamaClient()
    result = make_pipeline(model, profiles, fake).route(
        sample_request(GenerationProfileName.STANDARD, TaskType.EXPLANATION)
    )
    assert result.status is GenerationStatus.COMPLETED
    assert isinstance(result.planner_result.plan, ExplanationPlan)
    assert fake.requests[0]["format"]["title"] == "ExplanationPlan"
    assert fake.requests[0]["options"]["num_predict"] == profiles.profiles[
        GenerationProfileName.STANDARD
    ].planner.output_budget(TaskType.EXPLANATION)
    assert fake.requests[0]["think"] is False


def test_task_plan_schema_registry_maps_canonical_tasks() -> None:
    expected_schema_names = {
        TaskType.EXPLANATION: "ExplanationPlan",
        TaskType.LECTURE: "LecturePlan",
        TaskType.SLIDES: "SlidesPlan",
        TaskType.LAB: "LabPlan",
        TaskType.ACTIVITY: "ActivityPlan",
        TaskType.INSTRUCTOR_GUIDE: "InstructorGuidePlan",
        TaskType.ASSESSMENT: "AssessmentPlan",
        TaskType.HOMEWORK: "AssessmentPlan",
    }
    assert {task: plan_schema_for_task(task).__name__ for task in TaskType} == expected_schema_names


def test_deep_uses_larger_planner_and_context_tiers(configs: tuple[Any, Any]) -> None:
    model, profiles = configs
    fake = FakeOllamaClient()
    result = make_pipeline(model, profiles, fake).route(sample_request(GenerationProfileName.DEEP))

    assert result.status is GenerationStatus.COMPLETED
    assert fake.requests[0]["format"]["title"] == "LecturePlan"
    assert fake.requests[0]["think"] is True
    assert fake.requests[0]["options"]["num_predict"] == 6144
    assert fake.requests[0]["options"]["temperature"] == 0.5
    assert result.planner_result is not None
    assert result.planner_result.timing.selected_context == 8192
    assert result.planner_result.prompt_version == "planner-v3"
    assert result.planner_result.timing.thinking_enabled is True
    assert result.planner_result.timing.output_budget == 3072
    assert result.planner_result.timing.thinking_reserve_tokens == 3072
    assert result.planner_result.timing.generation_budget == 6144
    assert result.planner_result.timing.thinking_characters == len(
        "private planner trace for metadata test"
    )
    assert result.planner_result.timing.thinking_token_count is None
    assert "private planner trace" not in result.model_dump_json()
    assert fake.requests[1]["think"] is False
    assert fake.requests[1]["options"]["num_predict"] == 4096
    assert fake.timeouts == [30.0, 1500.0, 1800.0, 1800.0]
    assert result.writer_result is not None
    assert result.writer_result.timing.selected_context in {8192, 16384, 32768}
    assert profiles.profiles[GenerationProfileName.DEEP].review_enabled is False
    assert result.reviewer_result is None


def test_stage_profile_generation_budget_and_thinking_reserve_validation(
    configs: tuple[Any, Any],
) -> None:
    _, profiles = configs
    standard = profiles.profiles[GenerationProfileName.STANDARD].planner.model_copy(
        update={
            "task_output_tokens": {TaskType.EXPLANATION: 512},
            "thinking_reserve_tokens": 0,
        }
    )
    deep = profiles.profiles[GenerationProfileName.DEEP].planner

    assert standard.think is False
    assert standard.output_budget(TaskType.EXPLANATION) == 512
    assert standard.generation_budget(TaskType.EXPLANATION) == 512
    assert deep.output_budget(TaskType.INSTRUCTOR_GUIDE) == 3072
    assert deep.generation_budget(TaskType.INSTRUCTOR_GUIDE) == 6144
    assert deep.output_budget(TaskType.SLIDES) == 1536
    assert deep.generation_budget(TaskType.SLIDES) == 4608

    invalid = standard.model_dump(mode="python") | {
        "think": False,
        "thinking_reserve_tokens": 1024,
    }
    with pytest.raises(ValidationError, match="thinking reserve must be zero"):
        StageProfile.model_validate(invalid)
    negative = standard.model_dump(mode="python") | {"thinking_reserve_tokens": -1}
    with pytest.raises(ValidationError):
        StageProfile.model_validate(negative)


def test_deep_planner_context_selection_reserves_structured_and_thinking_budgets(
    configs: tuple[Any, Any],
) -> None:
    _, profiles = configs
    deep = profiles.profiles[GenerationProfileName.DEEP].planner
    structured_budget = deep.output_budget(TaskType.SLIDES)
    generation_budget = deep.generation_budget(TaskType.SLIDES)

    without_thinking_reserve = select_context_tier(
        "x" * 10_000,
        deep.context_tiers,
        safety_margin=profiles.context_safety_margin,
        output_reserve_tokens=structured_budget,
    )
    with_thinking_reserve = select_context_tier(
        "x" * 10_000,
        deep.context_tiers,
        safety_margin=profiles.context_safety_margin,
        output_reserve_tokens=generation_budget,
    )

    assert without_thinking_reserve.required_tokens_with_margin < 8192
    assert without_thinking_reserve.selected_context == 8192
    assert with_thinking_reserve.required_tokens_with_margin > 8192
    assert with_thinking_reserve.selected_context == 16384


def test_deep_slides_planner_receives_combined_generation_budget(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    fake = FakeOllamaClient(planner_failure="invalid_json")
    request = GenerationRequest(
        task=TaskType.SLIDES,
        profile=GenerationProfileName.DEEP,
        instruction="Create slides introducing C Sharp.",
    )

    result = make_pipeline(model, profiles, fake).route(request)

    assert result.status is GenerationStatus.FAILED
    planner_request = fake.requests[0]
    assert planner_request["format"]["title"] == "SlidesPlan"
    assert planner_request["think"] is True
    assert planner_request["options"]["num_predict"] == 4608
    assert result.planner_result is not None and result.planner_result.timing is not None
    assert result.planner_result.timing.output_budget == 1536
    assert result.planner_result.timing.thinking_reserve_tokens == 3072
    assert result.planner_result.timing.generation_budget == 4608


def test_deep_slides_plan_can_exceed_structured_budget_without_hitting_total_limit(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    plan_payload = {
        "task": "slides",
        "artifact_structure": ["Title", "Concept", "Practice"],
        "notes_for_writer": [],
        "source_scope": None,
        "objectives": ["Explain C Sharp variables."],
        "prior_knowledge": ["Basic programming concepts."],
        "slide_sequence": [
            {
                "title": "Variables",
                "purpose": "Introduce named values.",
                "concepts": ["declaration", "assignment"],
            },
            {
                "title": "Practice",
                "purpose": "Apply variable declarations.",
                "concepts": ["types"],
            },
        ],
        "examples": [],
        "exercises": ["Declare an integer variable."],
        "synthesis": ["Variables associate names with values."],
        "source_coverage": [],
    }
    fake = FakeOllamaClient(
        planner_response=ChatResponse(
            model=model.ollama_name,
            content=json.dumps(plan_payload),
            completion_tokens=3000,
            completion_reason="stop",
        )
    )
    request = GenerationRequest(
        task=TaskType.SLIDES,
        profile=GenerationProfileName.DEEP,
        instruction="Create slides introducing C Sharp variables.",
    )

    result = make_pipeline(model, profiles, fake).route(request)

    assert result.status is GenerationStatus.COMPLETED, result.errors
    assert result.planner_result is not None and result.planner_result.timing is not None
    assert result.planner_result.timing.output_budget == 1536
    assert result.planner_result.timing.generation_budget == 4608
    assert result.planner_result.timing.generated_tokens == 3000
    assert result.planner_result.timing.output_limit_reached is False


def test_deep_instructor_guide_planner_can_reason_and_validate_a_plan(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    plan_payload = {
        "task": "instructor_guide",
        "artifact_structure": ["Overview", "Key explanations"],
        "notes_for_writer": [],
        "source_scope": None,
        "objectives": ["Explain the .NET course overview."],
        "key_explanations": ["Describe the module sequence."],
        "misconceptions": [],
        "worked_solutions": [],
        "alternate_examples": [],
        "instructor_questions": ["What prior knowledge is needed?"],
        "source_coverage": [],
    }
    planner_response = ChatResponse(
        model=model.ollama_name,
        content=json.dumps(plan_payload),
        completion_tokens=3000,
        completion_reason="stop",
        thinking_content="private reasoning",
    )
    fake = FakeOllamaClient(planner_response=planner_response)
    request = GenerationRequest(
        task=TaskType.INSTRUCTOR_GUIDE,
        profile=GenerationProfileName.DEEP,
        instruction="Create an instructor guide for the C Sharp and .NET course overview.",
    )

    result = make_pipeline(model, profiles, fake).route(request)

    assert result.status is GenerationStatus.COMPLETED, result.errors
    assert result.planner_result is not None and result.planner_result.plan is not None
    assert result.planner_result.plan.task is TaskType.INSTRUCTOR_GUIDE
    planner_request = fake.requests[0]
    assert planner_request["think"] is True
    assert planner_request["options"]["num_predict"] == 6144
    assert planner_request["options"]["num_ctx"] == result.planner_result.timing.selected_context
    assert result.planner_result.timing.output_budget == 3072
    assert result.planner_result.timing.thinking_reserve_tokens == 3072
    assert result.planner_result.timing.generation_budget == 6144
    assert result.planner_result.timing.output_limit_reached is False
    assert result.planner_result.timing.thinking_characters == len("private reasoning")
    assert "private reasoning" not in result.model_dump_json()


def test_deep_planner_reports_reasoning_budget_exhaustion_without_exposing_reasoning(
    configs: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    model, profiles = configs
    planner_response = ChatResponse(
        model=model.ollama_name,
        content="",
        completion_tokens=6144,
        completion_reason="length",
        thinking_content="private reasoning used the entire generation allowance",
    )
    fake = FakeOllamaClient(planner_response=planner_response)
    request = GenerationRequest(
        task=TaskType.INSTRUCTOR_GUIDE,
        profile=GenerationProfileName.DEEP,
        instruction="Create an instructor guide for the C Sharp and .NET course overview.",
    )

    result = make_pipeline(model, profiles, fake).route(request)

    assert result.status is GenerationStatus.FAILED
    assert result.planner_result is not None
    assert result.planner_result.plan is None
    assert result.planner_result.raw_response == ""
    assert result.planner_result.error_type == "PlannerThinkingBudgetExhaustedError"
    assert "exhausted its generation budget during reasoning" in (
        result.planner_result.error_message or ""
    )
    assert "ValidationError" not in (result.planner_result.error_message or "")
    assert result.planner_result.timing is not None
    assert result.planner_result.timing.output_budget == 3072
    assert result.planner_result.timing.thinking_reserve_tokens == 3072
    assert result.planner_result.timing.generation_budget == 6144
    assert result.planner_result.timing.generated_tokens == 6144
    assert result.planner_result.timing.thinking_characters == len(
        "private reasoning used the entire generation allowance"
    )
    assert result.planner_result.timing.stop_reason == "length"
    assert result.planner_result.timing.output_limit_reached is True
    assert "private reasoning used" not in result.model_dump_json()
    assert len(fake.requests) == 1
    run_directory = tmp_path / "deep-budget-exhaustion"
    save_generation_run(run_directory, request, result)
    diagnostics = (run_directory / "diagnostics.md").read_text(encoding="utf-8")
    planner_prompt = json.loads((run_directory / "planner_prompt.json").read_text(encoding="utf-8"))
    assert "- Structured output budget: 3072" in diagnostics
    assert "- Thinking reserve: 3072" in diagnostics
    assert "- Total generation budget: 6144" in diagnostics
    assert "- Generated tokens: 6144" in diagnostics
    assert "- Thinking characters: " in diagnostics
    assert "- Stop reason: length" in diagnostics
    assert "private reasoning used" not in diagnostics
    assert planner_prompt["output_budget"] == 3072
    assert planner_prompt["thinking_reserve_tokens"] == 3072
    assert planner_prompt["generation_budget"] == 6144


def test_planner_reports_truncated_json_only_when_generation_budget_was_reached(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    truncated = ChatResponse(
        model=model.ollama_name,
        content='{"task":"explanation","concept":"test"',
        completion_tokens=6144,
        completion_reason="length",
    )
    fake = FakeOllamaClient(planner_response=truncated)
    request = GenerationRequest(
        task=TaskType.EXPLANATION,
        profile=GenerationProfileName.DEEP,
        instruction="Explain a concept.",
    )
    result = make_pipeline(model, profiles, fake).route(request)

    assert result.status is GenerationStatus.FAILED
    assert result.planner_result is not None
    assert result.planner_result.raw_response == truncated.content
    assert result.planner_result.error_type == "PlannerOutputTruncatedError"
    assert "truncated after reaching its generation budget" in (
        result.planner_result.error_message or ""
    )
    assert result.planner_result.timing is not None
    assert result.planner_result.timing.output_limit_reached is True


def test_malformed_planner_json_without_length_stop_keeps_validation_error(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    fake = FakeOllamaClient(
        planner_response=ChatResponse(
            model=model.ollama_name,
            content="not JSON",
            completion_tokens=100,
            completion_reason="stop",
        )
    )
    request = GenerationRequest(
        task=TaskType.EXPLANATION,
        profile=GenerationProfileName.DEEP,
        instruction="Explain a concept.",
    )

    result = make_pipeline(model, profiles, fake).route(request)

    assert result.planner_result is not None
    assert result.planner_result.error_type == "ValidationError"
    assert result.planner_result.error_message is not None
    assert "truncated after reaching" not in result.planner_result.error_message


def test_planner_and_writer_limit_statuses_are_recorded_separately(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    profile = profiles.profiles[GenerationProfileName.STANDARD].model_copy(
        update={
            "planner": profiles.profiles[GenerationProfileName.STANDARD].planner.model_copy(
                update={"task_output_tokens": {TaskType.EXPLANATION: 64}}
            ),
            "writer": profiles.profiles[GenerationProfileName.STANDARD].writer.model_copy(
                update={"task_output_tokens": {TaskType.EXPLANATION: 64}}
            ),
        }
    )
    profiles = profiles.model_copy(
        update={"profiles": {**profiles.profiles, GenerationProfileName.STANDARD: profile}}
    )
    fake = FakeOllamaClient(writer_tokens=64, planner_tokens=64)
    result = make_pipeline(model, profiles, fake).route(
        sample_request(GenerationProfileName.STANDARD, TaskType.EXPLANATION)
    )
    assert result.planner_result is not None
    assert result.writer_result is not None
    assert result.planner_result.timing.output_limit_reached is True
    assert result.planner_result.timing.potentially_truncated is True
    assert result.writer_result.timing.output_limit_reached is True
    assert result.writer_result.timing.potentially_truncated is True


def test_invalid_planner_json_preserves_raw_output_and_skips_writer(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    fake = FakeOllamaClient(planner_failure="invalid_json")
    result = make_pipeline(model, profiles, fake).route(
        sample_request(GenerationProfileName.STANDARD)
    )
    assert result.status is GenerationStatus.FAILED
    assert result.planner_result is not None
    assert result.planner_result.raw_response == "not valid JSON"
    assert result.planner_result.error_type == "ValidationError"
    assert result.writer_result is None
    assert len(fake.requests) == 1


def test_planner_timeout_is_distinct_and_does_not_run_writer(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    fake = FakeOllamaClient(planner_failure="timeout")
    result = make_pipeline(model, profiles, fake).route(
        sample_request(GenerationProfileName.STANDARD)
    )
    assert result.status is GenerationStatus.FAILED
    assert result.planner_result is not None
    assert result.planner_result.error_type == "OllamaTimeoutError"
    assert result.writer_result is None
    assert len(fake.requests) == 1


def test_invalid_teaching_plan_schema_preserves_raw_planner_response(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    fake = FakeOllamaClient(planner_failure="invalid_schema")
    result = make_pipeline(model, profiles, fake).route(
        sample_request(GenerationProfileName.STANDARD)
    )
    assert result.planner_result is not None
    assert result.planner_result.error_type == "ValidationError"
    assert result.planner_result.raw_response is not None
    assert result.planner_result.timing is not None
    assert result.planner_result.timing.generated_tokens == 80
    assert result.planner_result.timing.stop_reason == "stop"
    assert result.writer_result is None


def test_partial_source_scope_is_passed_to_writer_and_pipeline_continues(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    fake = FakeOllamaClient(source_scope_status="partial")
    result = make_pipeline(model, profiles, fake).route(
        sample_request(GenerationProfileName.STANDARD, TaskType.EXPLANATION)
    )

    assert result.status is GenerationStatus.COMPLETED
    assert result.planner_result is not None and result.planner_result.plan is not None
    scope = result.planner_result.plan.source_scope
    assert scope is not None and scope.status is SourceScopeStatus.PARTIAL
    assert scope.supported_topics == ["DNS name resolution"]
    assert scope.unsupported_requested_topics == ["the broader requested topic"]
    assert result.writer_result is not None
    writer_request = next(item for item in fake.requests if item["format"] is None)
    assert '"status": "partial"' in writer_request["user_message"]
    assert "do not fill unsupported_requested_topics" in writer_request["system_message"]
    assert result.initial_grounding_review is not None


def test_insufficient_source_scope_fails_before_writer_without_fallback(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    fake = FakeOllamaClient(source_scope_status="insufficient")
    result = make_pipeline(model, profiles, fake).route(
        sample_request(GenerationProfileName.STANDARD, TaskType.EXPLANATION)
    )

    assert result.status is GenerationStatus.FAILED
    assert result.final_output is None
    assert result.planner_result is not None and result.planner_result.plan is not None
    assert result.planner_result.plan.source_scope is not None
    assert result.planner_result.plan.source_scope.status is SourceScopeStatus.INSUFFICIENT
    assert result.writer_result is None
    assert len(fake.requests) == 1
    assert any("sources are insufficient" in error for error in result.errors)


@pytest.mark.parametrize(
    ("final_status", "expected_status", "writer_runs"),
    [
        ("sufficient", GenerationStatus.COMPLETED, True),
        ("partial", GenerationStatus.COMPLETED, True),
        ("insufficient", GenerationStatus.FAILED, False),
    ],
)
def test_partial_scope_expands_once_and_uses_final_assessment(
    configs: tuple[Any, Any],
    final_status: str,
    expected_status: GenerationStatus,
    writer_runs: bool,
) -> None:
    model, profiles = configs
    fake = FakeOllamaClient(source_scope_statuses=["partial", final_status])
    request = sample_request(GenerationProfileName.STANDARD, TaskType.EXPLANATION)
    retrieval_metadata = {
        "knowledge_retrieval": {
            "diagnostics": {
                "retrieval_rounds": [{"round": 1, "type": "initial", "retrieved_count": 1}]
            }
        }
    }
    request = request.model_copy(update={"metadata": retrieval_metadata})
    expansion_calls: list[SourceScopeStatus] = []
    expanded_requests: list[GenerationRequest] = []

    def expand(
        sourced_request: GenerationRequest,
        scope: Any,
    ) -> GenerationRequest:
        expansion_calls.append(scope.status)
        metadata = dict(sourced_request.metadata)
        diagnostics = dict(metadata["knowledge_retrieval"]["diagnostics"])
        diagnostics["scope_reassessment"] = True
        diagnostics["retrieval_rounds"] = [
            *diagnostics["retrieval_rounds"],
            {"round": 2, "type": "expansion", "retrieved_count": 1},
        ]
        metadata["knowledge_retrieval"] = {
            **metadata["knowledge_retrieval"],
            "diagnostics": diagnostics,
        }
        expanded_request = sourced_request.model_copy(
            update={
                "source_material": [
                    *sourced_request.source_material,
                    SourceMaterial(
                        source_id="dns-expansion",
                        title="Expanded DNS source",
                        text="Expanded source details about DNS.",
                    ),
                ],
                "metadata": metadata,
            }
        )
        expanded_requests.append(expanded_request)
        return expanded_request

    result = make_pipeline(model, profiles, fake).route(
        request,
        expand_retrieval=expand,
    )

    assert result.status is expected_status
    assert expansion_calls == [SourceScopeStatus.PARTIAL]
    planner_requests = [
        item
        for item in fake.requests
        if item["format"] is not None and item["format"].get("title") == "ExplanationPlan"
    ]
    assert len(planner_requests) == 2
    assert "Expanded DNS source" in planner_requests[1]["user_message"]
    assert "## Scope reassessment" in planner_requests[1]["user_message"]
    assert (result.writer_result is not None) is writer_runs
    final_scope = result.planner_result.plan.source_scope
    assert final_scope is not None and final_scope.status.value == final_status
    diagnostics = expanded_requests[0].metadata["knowledge_retrieval"]["diagnostics"]
    assert diagnostics["source_scope_assessments"]["initial"]["status"] == "partial"
    assert diagnostics["source_scope_assessments"]["final"]["status"] == final_status
    assert len(diagnostics["retrieval_rounds"]) == 2


@pytest.mark.parametrize("first_status", ["sufficient", "insufficient"])
def test_non_partial_initial_scope_does_not_expand(
    configs: tuple[Any, Any],
    first_status: str,
) -> None:
    model, profiles = configs
    fake = FakeOllamaClient(source_scope_status=first_status)
    request = sample_request(GenerationProfileName.STANDARD, TaskType.EXPLANATION)
    request = request.model_copy(
        update={
            "metadata": {
                "knowledge_retrieval": {
                    "diagnostics": {"retrieval_rounds": [{"round": 1, "type": "initial"}]}
                }
            }
        }
    )
    expansion_calls: list[bool] = []

    def expand(
        sourced_request: GenerationRequest,
        scope: Any,
    ) -> GenerationRequest:
        expansion_calls.append(True)
        return sourced_request

    result = make_pipeline(model, profiles, fake).route(
        request,
        expand_retrieval=expand,
    )

    assert expansion_calls == []
    planner_requests = [
        item
        for item in fake.requests
        if item["format"] is not None and item["format"].get("title") == "ExplanationPlan"
    ]
    assert len(planner_requests) == 1
    assert (result.writer_result is not None) is (first_status == "sufficient")
    diagnostics = request.metadata["knowledge_retrieval"]["diagnostics"]
    assert diagnostics["source_scope_assessments"]["initial"]["status"] == first_status
    assert diagnostics["source_scope_assessments"]["final"]["status"] == first_status


def test_teaching_plan_schema_rejects_unknown_task() -> None:
    with pytest.raises(ValidationError):
        TeachingPlan.model_validate({"task": "unknown_task"})


def test_writer_timeout_preserves_successful_plan(configs: tuple[Any, Any]) -> None:
    model, profiles = configs
    fake = FakeOllamaClient(writer_failure=True)
    result = make_pipeline(model, profiles, fake).route(
        sample_request(GenerationProfileName.STANDARD)
    )
    assert result.status is GenerationStatus.FAILED
    assert result.planner_result is not None and result.planner_result.plan is not None
    assert result.writer_result is not None
    assert result.writer_result.error_type == "OllamaTimeoutError"
    assert result.final_output is None


def _revision_required_review(
    excerpt: str = "The primary motivation for predicate logic was to simplify the web.",
) -> str:
    return json.dumps(
        {
            "claims": [
                {
                    "claim_id": "C001",
                    "classification": "unsupported",
                    "reason": f"The supplied source does not support: {excerpt}",
                    "category": "unsupported_framing",
                    "evidence_ids": [],
                }
            ],
        }
    )


def test_grounding_review_rejects_inconsistent_decisions() -> None:
    with pytest.raises(ValidationError):
        GroundingReview.model_validate(
            {
                "decision": "pass",
                "claim_assessments": [
                    _assessment(
                        "C001",
                        "unsupported claim",
                        classification="unsupported",
                    )
                ],
                "issues": [
                    {
                        "claim_id": "C001",
                        "excerpt": "unsupported claim",
                        "category": "unsupported_fact",
                        "reason": "No supporting source.",
                    }
                ],
            }
        )
    with pytest.raises(ValidationError):
        GroundingReview.model_validate(
            {
                "decision": "revision_required",
                "claim_assessments": [
                    _assessment(
                        "C001",
                        "unsupported claim",
                        classification="unsupported",
                    )
                ],
                "issues": [{"claim_id": "C001", "excerpt": "unsupported claim"}],
                "revision_instructions": ["Remove the claim."],
            }
        )


def test_claim_coverage_normalizes_markdown_math_and_sentence_quotes() -> None:
    artifact = (
        'The **Universal Quantifier** ($∀$) applies to "all members." '
        "Predicate logic uses predicates."
    )
    review = GroundingReview.model_validate(
        {
            "decision": "pass",
            "claim_assessments": [
                _assessment(
                    "C001",
                    (
                        'The **Universal Quantifier** ($\\forall$) applies to "all members.", '
                        "Predicate logic uses predicates."
                    ),
                    evidence_ids=["S01-E001"],
                )
            ],
            "issues": [],
        }
    )

    _validate_claim_coverage(artifact, review)


def test_claim_coverage_normalizes_alternate_quotes_inside_excerpts() -> None:
    artifact = 'Predicate logic expresses claims about "all" or "some" members.'
    review = GroundingReview.model_validate(
        {
            "decision": "pass",
            "claim_assessments": [
                _assessment(
                    "C001",
                    "Predicate logic expresses claims about 'all' or 'some' members.",
                    evidence_ids=["S01-E001"],
                )
            ],
            "issues": [],
        }
    )

    _validate_claim_coverage(artifact, review)


def test_claim_coverage_preserves_identifiers_inside_inline_math() -> None:
    artifact = "A one-place predicate $P$ maps an entity to the proposition $P(a)$."
    review = GroundingReview.model_validate(
        {
            "decision": "pass",
            "claim_assessments": [
                _assessment(
                    "C001",
                    "A one-place predicate P maps an entity to the proposition P(a).",
                    evidence_ids=["S01-E001"],
                )
            ],
            "issues": [],
        }
    )

    _validate_claim_coverage(artifact, review)


def test_factual_sentence_extraction_skips_labeled_formal_examples() -> None:
    artifact = (
        "**Expression:** $\\forall x P(x)$\n"
        'For example: "$x$ is tall."\n'
        "This foundation supports rigorous reasoning."
    )

    assert _factual_sentence_excerpts(artifact) == ["This foundation supports rigorous reasoning."]


def test_factual_sentence_extraction_does_not_split_vs_abbreviation() -> None:
    artifact = (
        "**Free vs. Bound Variables**: In an open statement, variables remain free "
        "until a quantifier binds them."
    )

    claims = _factual_sentence_excerpts(artifact)

    assert claims == [
        "**Free vs. Bound Variables**: In an open statement, variables remain free "
        "until a quantifier binds them."
    ]
    assert "**Free vs." not in claims
    assert not any(claim.startswith("Bound Variables**") for claim in claims)


def test_factual_sentence_extraction_preserves_and_excludes_labeled_quoted_relation() -> None:
    artifact = (
        '*Multi-Place*: Consider the relation "is taller than." '
        "Predicates with several places express relations among several entities."
    )

    claims = _factual_sentence_excerpts(artifact)

    assert claims == ["Predicates with several places express relations among several entities."]
    assert not any("is taller than." in claim for claim in claims)


def test_factual_sentence_extraction_skips_instructional_question_stems() -> None:
    artifact = (
        'Consider the statement "All swans are white." How do quantifiers express "all"?\n'
        "Explain the difference between a free variable and a bound variable. "
        "Why is binding needed?\n"
        "Predicate logic uses quantifiers."
    )

    assert _factual_sentence_excerpts(artifact) == ["Predicate logic uses quantifiers."]


def test_claim_coverage_splits_sentence_before_wiki_link() -> None:
    artifact = (
        "This resemblance was historically important. "
        "[[The Decision Problem|The later decision problem]] asked whether a procedure could "
        "decide every proposition."
    )
    review = GroundingReview.model_validate(
        {
            "decision": "pass",
            "claim_assessments": [
                _assessment(
                    "C001",
                    "This resemblance was historically important.",
                    evidence_ids=["S01-E001"],
                ),
                _assessment(
                    "C002",
                    (
                        "[[The Decision Problem|The later decision problem]] asked whether a "
                        "procedure could decide every proposition."
                    ),
                    evidence_ids=["S01-E001"],
                ),
            ],
            "issues": [],
        }
    )

    _validate_claim_coverage(artifact, review)


def test_claim_coverage_rejects_an_unassessed_factual_sentence() -> None:
    artifact = "Predicate logic uses predicates. Quantifiers bind variables."
    review = GroundingReview.model_validate(
        {
            "decision": "pass",
            "claim_assessments": [
                _assessment("C001", "Predicate logic uses predicates.", evidence_ids=["S01-E001"])
            ],
            "issues": [],
        }
    )

    with pytest.raises(ValueError, match="omitted 1 factual sentence"):
        _validate_claim_coverage(artifact, review)


@pytest.mark.parametrize(
    ("source_text", "claim_text"),
    [
        (
            "This resemblance to a programming language was historically important: reasoning "
            "could be studied as a process performed by rule.",
            "This resemblance to a programming language was historically important: reasoning "
            "could be studied as a process performed by rule.",
        ),
        ("The predicate $P(a)$ is true.", "The predicate P(a) is true."),
        (
            "The [[Universal Quantifier|universal quantifier]] binds variables.",
            "The universal quantifier binds variables.",
        ),
        (
            "Predicate logic, extends propositional logic!",
            "Predicate logic extends propositional logic.",
        ),
    ],
)
def test_direct_source_matches_use_stable_ids_and_bypass_reviewer(
    source_text: str,
    claim_text: str,
) -> None:
    request = sample_request(GenerationProfileName.STANDARD).model_copy(
        update={
            "source_material": [
                SourceMaterial(source_id="source-1", title="Logic notes", text=source_text)
            ]
        }
    )

    claims, direct, unresolved = prepare_grounding_claims(request, claim_text)

    assert [claim.claim_id for claim in claims] == ["C001"]
    assert unresolved == []
    assert direct[0].classification is GroundingClaimClassification.DIRECT_SUPPORTED
    assert direct[0].support_method is GroundingSupportMethod.NORMALIZED_DIRECT_MATCH
    assert direct[0].evidence_ids == ["S01-E001"]


@pytest.mark.parametrize(
    ("artifact_claim", "source_text"),
    [
        (
            "**Core Definition**: Predicate logic extends propositional logic.",
            "Predicate logic extends propositional logic.",
        ),
        (
            "**The Decision Problem**: The later decision problem asked whether a procedure "
            "could decide every proposition.",
            "The later decision problem asked whether a procedure could decide every proposition.",
        ),
        (
            "> **Key Distinction**: Variables remain free until a quantifier binds them.",
            "Variables remain free until a quantifier binds them.",
        ),
        (
            "*Multi-Place*: Predicates with several places express relations among "
            "several entities.",
            "Predicates with several places express relations among several entities.",
        ),
    ],
)
def test_presentation_labels_do_not_prevent_direct_source_matches(
    artifact_claim: str,
    source_text: str,
) -> None:
    request = sample_request(GenerationProfileName.STANDARD).model_copy(
        update={
            "source_material": [
                SourceMaterial(source_id="source-1", title="Logic notes", text=source_text)
            ]
        }
    )

    claims, direct, unresolved = prepare_grounding_claims(request, artifact_claim)

    assert len(claims) == 1
    assert direct[0].classification is GroundingClaimClassification.DIRECT_SUPPORTED
    assert direct[0].evidence_ids == ["S01-E001"]
    assert unresolved == []
    assert claims[0].text == artifact_claim


def test_presentation_label_does_not_strip_factual_qualifiers() -> None:
    request = sample_request(GenerationProfileName.STANDARD).model_copy(
        update={
            "source_material": [
                SourceMaterial(
                    source_id="source-1",
                    title="Logic notes",
                    text="Predicate logic extends propositional logic.",
                )
            ]
        }
    )

    _, direct, unresolved = prepare_grounding_claims(
        request,
        "**Important Fact**: Predicate logic is essential to computer science.",
    )

    assert direct == []
    assert len(unresolved) == 1
    assert unresolved[0].text == (
        "**Important Fact**: Predicate logic is essential to computer science."
    )


def test_evidence_ledger_is_deterministic_and_ordered() -> None:
    request = sample_request(GenerationProfileName.STANDARD).model_copy(
        update={
            "source_material": [
                SourceMaterial(
                    source_id="logic-1",
                    title="Logic 1",
                    text="First sentence. Second sentence.",
                ),
                SourceMaterial(
                    source_id="logic-2",
                    title="Logic 2",
                    text="Third sentence.",
                ),
            ]
        }
    )

    ledger = build_evidence_ledger(request)

    assert [span.evidence_id for span in ledger] == ["S01-E001", "S01-E002", "S02-E001"]
    assert [span.source_id for span in ledger] == ["logic-1", "logic-1", "logic-2"]
    assert [span.order for span in ledger] == [1, 2, 3]


def test_evidence_ledger_allows_empty_source_material() -> None:
    request = sample_request(GenerationProfileName.STANDARD).model_copy(
        update={"source_material": []}
    )

    assert build_evidence_ledger(request) == []


def test_invalid_source_material_is_still_rejected() -> None:
    with pytest.raises(ValidationError):
        SourceMaterial(source_id="logic-1", title="Logic 1", text="")


def test_verbatim_historical_source_match_cannot_be_overturned_by_reviewer(
    configs: tuple[Any, Any],
) -> None:
    source_sentence = (
        "This resemblance to a programming language was historically important: reasoning "
        "could be studied as a process performed by rule."
    )
    request = sample_request(GenerationProfileName.STANDARD).model_copy(
        update={
            "source_material": [
                SourceMaterial(source_id="source-1", title="History", text=source_sentence)
            ]
        }
    )
    model, profiles = configs
    fake = FakeOllamaClient(writer_output=source_sentence)
    result = make_pipeline(
        model,
        profiles,
        fake,
    ).route(request)

    assert result.status is GenerationStatus.COMPLETED
    assert result.final_output == source_sentence
    assert result.initial_grounding_review is not None
    assert result.initial_grounding_review.review is not None
    assert result.initial_grounding_review.review.claim_assessments[0].classification is (
        GroundingClaimClassification.DIRECT_SUPPORTED
    )
    assert result.initial_grounding_review.review.claim_assessments[0].evidence_ids == ["S01-E001"]
    assert not any(
        record["format"] is not None
        and record["format"].get("title") == "GroundingReviewerResponse"
        for record in fake.requests
    )


def test_presentation_labeled_direct_match_bypasses_reviewer(
    configs: tuple[Any, Any],
) -> None:
    source_sentence = (
        "Predicate logic extends propositional logic by representing properties and relations "
        "that become propositions when applied to entities."
    )
    artifact = f"**Core Definition**: {source_sentence}"
    request = sample_request(GenerationProfileName.STANDARD).model_copy(
        update={
            "source_material": [
                SourceMaterial(source_id="logic", title="Logic notes", text=source_sentence)
            ]
        }
    )
    model, profiles = configs
    fake = FakeOllamaClient(writer_output=artifact)

    result = make_pipeline(model, profiles, fake).route(request)

    assert result.status is GenerationStatus.COMPLETED
    assert result.final_output == artifact
    assert result.initial_grounding_review is not None
    assert result.initial_grounding_review.review is not None
    claim = result.initial_grounding_review.review.claim_assessments[0]
    assert claim.classification is GroundingClaimClassification.DIRECT_SUPPORTED
    assert claim.evidence_ids == ["S01-E001"]
    assert not any(
        record["format"] is not None
        and record["format"].get("title") == "GroundingReviewerResponse"
        for record in fake.requests
    )


def test_reviewer_entailment_requires_verifiable_evidence() -> None:
    source_text = (
        "Variables in an open statement remain free until a Universal Quantifier or Existential "
        "Quantifier binds them. This lets logic express general claims."
    )
    request_claim = GroundingClaimInput(
        claim_id="C001",
        text="Quantifiers bind variables so predicate logic can express general claims.",
    )
    valid_response = GroundingReviewerResponse.model_validate(
        {
            "claims": [
                {
                    "claim_id": "C001",
                    "classification": "supported",
                    "reason": (
                        "The source says binding variables lets logic express general claims."
                    ),
                    "evidence_ids": ["S01-E001"],
                }
            ]
        }
    )
    review = merge_grounding_review(
        [request_claim],
        [],
        valid_response,
        [
            EvidenceSpan(
                evidence_id="S01-E001",
                source_id="logic-notes",
                source_title="Logic notes",
                text=source_text,
                order=1,
            )
        ],
    )

    assert review.decision is GroundingDecision.PASS
    assert review.claim_assessments[0].classification is GroundingClaimClassification.SUPPORTED
    assert review.claim_assessments[0].evidence_ids == ["S01-E001"]


def test_reviewer_response_models_require_classification_specific_evidence_ids() -> None:
    valid_claims = [
        {
            "claim_id": "C001",
            "classification": "supported",
            "evidence_ids": ["S01-E001"],
            "reason": "The selected evidence supports the claim.",
        },
        {
            "claim_id": "C002",
            "classification": "unsupported",
            "evidence_ids": [],
            "reason": "No supplied evidence establishes the claim.",
        },
        {
            "claim_id": "C003",
            "classification": "pedagogical",
            "evidence_ids": [],
            "reason": "This is a hypothetical teaching example.",
        },
    ]
    response = GroundingReviewerResponse.model_validate({"claims": valid_claims})
    assert len(response.claim_assessments) == 3

    invalid_claims = [
        {
            "claim_id": "C001",
            "classification": "supported",
            "reason": "Supported.",
        },
        {
            "claim_id": "C001",
            "classification": "supported",
            "evidence_ids": [],
            "reason": "Supported.",
        },
        {
            "claim_id": "C001",
            "classification": "unsupported",
            "evidence_ids": ["S01-E001"],
            "reason": "Unsupported.",
        },
        {
            "claim_id": "C001",
            "classification": "pedagogical",
            "evidence_ids": ["S01-E001"],
            "reason": "Example.",
        },
        {
            "claim_id": "C001",
            "classification": "direct_supported",
            "evidence_ids": ["S01-E001"],
            "reason": "Directly supported.",
        },
    ]
    for claim in invalid_claims:
        with pytest.raises(ValidationError):
            GroundingReviewerResponse.model_validate({"claims": [claim]})


def test_reviewer_response_json_schema_requires_evidence_ids_by_classification() -> None:
    schema = GroundingReviewerResponse.model_json_schema()
    definitions = schema["$defs"]
    claim_items = schema["properties"]["claims"]["items"]
    assert claim_items["discriminator"]["propertyName"] == "classification"
    assert len(claim_items["oneOf"]) == 3

    supported = definitions["SupportedReviewerClaim"]
    assert "evidence_ids" in supported["required"]
    assert supported["properties"]["evidence_ids"]["minItems"] == 1
    pedagogical = definitions["PedagogicalReviewerClaim"]
    assert "evidence_ids" in pedagogical["required"]
    assert pedagogical["properties"]["evidence_ids"]["minItems"] == 0
    assert pedagogical["properties"]["evidence_ids"]["maxItems"] == 0
    unsupported = definitions["UnsupportedReviewerClaim"]
    assert "evidence_ids" in unsupported["required"]
    assert unsupported["properties"]["evidence_ids"]["minItems"] == 0
    assert unsupported["properties"]["evidence_ids"]["maxItems"] == 0


@pytest.mark.parametrize(
    ("evidence_ids",),
    [
        (["unknown-ref"],),
    ],
)
def test_fabricated_or_uncited_reviewer_evidence_fails_closed(
    evidence_ids: list[str],
) -> None:
    claim = GroundingClaimInput(claim_id="C001", text="Predicate logic is essential.")
    response = GroundingReviewerResponse.model_validate(
        {
            "claims": [
                {
                    "claim_id": "C001",
                    "classification": "supported",
                    "reason": "The reviewer claims support.",
                    "evidence_ids": evidence_ids,
                }
            ]
        }
    )
    review = merge_grounding_review(
        [claim],
        [],
        response,
        [
            EvidenceSpan(
                evidence_id="S01-E001",
                source_id="logic-notes",
                source_title="Logic notes",
                text="Predicate logic extends propositional logic.",
                order=1,
            )
        ],
    )

    assert review.decision is GroundingDecision.REVISION_REQUIRED
    assert review.claim_assessments[0].classification is GroundingClaimClassification.UNSUPPORTED
    assert review.evidence_validation_failures == 1
    assert review.issues[0].claim_id == "C001"


def test_global_ledger_preserves_direct_supported_pedagogical_and_unsupported_claims() -> None:
    request = sample_request(GenerationProfileName.STANDARD).model_copy(
        update={
            "source_material": [
                SourceMaterial(
                    source_id="logic-notes",
                    title="Logic notes",
                    text="Predicate logic extends propositional logic.",
                )
            ]
        }
    )
    all_claims, direct, unresolved = prepare_grounding_claims(
        request,
        "Predicate logic extends propositional logic. "
        "Quantifiers bind variables. "
        "Let P(x) mean x is wise. "
        "Predicate logic is essential to computer science.",
    )
    assert [claim.claim_id for claim in unresolved] == ["C002", "C003", "C004"]
    response = GroundingReviewerResponse.model_validate(
        {
            "claims": [
                {
                    "claim_id": "C002",
                    "classification": "supported",
                    "reason": "The source explains quantifier binding.",
                    "evidence_ids": ["S01-E001"],
                },
                {
                    "claim_id": "C003",
                    "classification": "pedagogical",
                    "reason": "This is a stipulated illustrative predicate.",
                    "evidence_ids": [],
                },
                {
                    "claim_id": "C004",
                    "classification": "unsupported",
                    "reason": "The source does not establish essential status.",
                    "category": "unsupported_significance",
                    "evidence_ids": [],
                },
            ]
        }
    )

    review = merge_grounding_review(
        all_claims,
        direct,
        response,
        build_evidence_ledger(request),
    )

    assert review.decision is GroundingDecision.REVISION_REQUIRED
    assert [claim.claim_id for claim in review.claim_assessments] == [
        "C001",
        "C002",
        "C003",
        "C004",
    ]
    assert [claim.classification.value for claim in review.claim_assessments] == [
        "direct_supported",
        "supported",
        "pedagogical",
        "unsupported",
    ]
    assert [issue.claim_id for issue in review.issues] == ["C004"]


def test_grounding_review_revises_once_and_validates_revised_output(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    candidate = "# DNS\n\nThe primary motivation for predicate logic was to simplify the web."
    revised = "# DNS\n\nDNS maps a human-readable domain name to an IP address."
    fake = FakeOllamaClient(
        writer_output=candidate,
        revision_output=revised,
        grounding_responses=[
            _revision_required_review(),
            _passing_review(),
        ],
    )
    events = []
    result = make_pipeline(model, profiles, fake).route(
        sample_request(GenerationProfileName.STANDARD),
        on_progress=events.append,
    )

    assert result.status is GenerationStatus.COMPLETED
    assert result.writer_result is not None and result.writer_result.raw_response == candidate
    assert result.revision_result is not None
    assert result.revision_result.raw_response == revised
    assert result.revision_result.prompt_version == "grounding-revision-v2"
    assert result.initial_grounding_review is not None
    assert result.initial_grounding_review.review is not None
    assert result.initial_grounding_review.review.decision is GroundingDecision.REVISION_REQUIRED
    assert (
        result.initial_grounding_review.review.issues[0].excerpt
        == "The primary motivation for predicate logic was to simplify the web."
    )
    assert result.final_grounding_review is not None
    assert result.final_grounding_review.review is not None
    assert result.final_grounding_review.review.decision is GroundingDecision.PASS
    assert result.final_output == revised
    assert sum(event.stage is GenerationStage.REVISING for event in events) == 2
    assert sum(event.stage is GenerationStage.REVIEWING for event in events) == 4
    assert fake.grounding_response_count == 1
    assert (
        result.final_grounding_review.review.claim_assessments[0].classification
        is GroundingClaimClassification.DIRECT_SUPPORTED
    )


def test_unsupported_claims_override_inconsistent_model_pass(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    unsupported_sentence = "Predicate logic is essential to all computer science."
    contradictory_response = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "C001",
                    "classification": "unsupported",
                    "reason": "The sources do not establish this significance claim.",
                    "category": "unsupported_significance",
                    "evidence_ids": [],
                }
            ],
        }
    )
    fake = FakeOllamaClient(
        writer_output=unsupported_sentence,
        revision_output="DNS maps a human-readable domain name to an IP address.",
        grounding_responses=[contradictory_response, _passing_review()],
    )

    result = make_pipeline(model, profiles, fake).route(
        sample_request(GenerationProfileName.STANDARD)
    )

    assert result.status is GenerationStatus.COMPLETED
    assert result.initial_grounding_review is not None
    assert result.initial_grounding_review.review is not None
    assert result.initial_grounding_review.review.decision is GroundingDecision.REVISION_REQUIRED
    assert result.initial_grounding_review.review.issues[0].excerpt == unsupported_sentence
    assert result.initial_grounding_review.review.claim_assessments[0].classification.value == (
        "unsupported"
    )
    assert result.final_output == "DNS maps a human-readable domain name to an IP address."


def test_grounding_review_fails_closed_after_second_revision_request(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    fake = FakeOllamaClient(
        grounding_responses=[
            _revision_required_review(),
            _revision_required_review("Predicate logic is essential to computing."),
        ],
        writer_output="The primary motivation for predicate logic was to simplify the web.",
        revision_output="Predicate logic is essential to computing.",
    )
    result = make_pipeline(model, profiles, fake).route(sample_request(GenerationProfileName.DEEP))

    assert result.status is GenerationStatus.FAILED
    assert result.final_output is None
    assert result.revision_result is not None
    assert result.revision_result.status is GenerationStatus.COMPLETED
    assert result.final_grounding_review is not None
    assert result.final_grounding_review.review is not None
    assert result.final_grounding_review.review.decision is GroundingDecision.REVISION_REQUIRED
    assert fake.grounding_response_count == 2
    assert any("one-revision limit" in error for error in result.errors)


@pytest.mark.parametrize(
    ("fake_options", "expected_record"),
    [
        ({"grounding_responses": ["not valid JSON"]}, "initial_grounding_review"),
        ({"grounding_failure_at": 1}, "initial_grounding_review"),
        (
            {
                "grounding_responses": [_revision_required_review()],
                "writer_output": (
                    "The primary motivation for predicate logic was to simplify the web."
                ),
                "revision_failure": True,
            },
            "revision_result",
        ),
        (
            {
                "grounding_responses": [
                    _revision_required_review(),
                    _passing_review(),
                ],
                "writer_output": (
                    "The primary motivation for predicate logic was to simplify the web."
                ),
                "revision_output": "Predicate logic is essential to computing.",
                "grounding_failure_at": 2,
            },
            "final_grounding_review",
        ),
    ],
)
def test_grounding_stage_errors_fail_closed(
    configs: tuple[Any, Any],
    fake_options: dict[str, Any],
    expected_record: str,
) -> None:
    model, profiles = configs
    result = make_pipeline(model, profiles, FakeOllamaClient(**fake_options)).route(
        sample_request(GenerationProfileName.STANDARD)
    )

    assert result.status is GenerationStatus.FAILED
    assert result.final_output is None
    assert getattr(result, expected_record) is not None
    if expected_record == "initial_grounding_review":
        assert result.initial_grounding_review.status is GenerationStatus.FAILED
    if expected_record == "revision_result":
        assert result.revision_result.status is GenerationStatus.FAILED
    if expected_record == "final_grounding_review":
        assert result.final_grounding_review.status is GenerationStatus.FAILED


def test_reviewer_omitting_evidence_ids_fails_schema_validation_closed(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    raw_response = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "C001",
                    "classification": "supported",
                    "reason": "Directly matches S01-E001.",
                }
            ]
        }
    )
    result = make_pipeline(
        model,
        profiles,
        FakeOllamaClient(
            writer_output="Predicate logic is essential to computer science.",
            grounding_responses=[raw_response],
        ),
    ).route(sample_request(GenerationProfileName.STANDARD))

    assert result.status is GenerationStatus.FAILED
    assert result.final_output is None
    assert result.initial_grounding_review is not None
    assert result.initial_grounding_review.status is GenerationStatus.FAILED
    assert result.initial_grounding_review.error_type == "ValidationError"
    assert result.initial_grounding_review.raw_response == raw_response
    assert result.initial_grounding_review.review is None
    assert "evidence_ids" in (result.initial_grounding_review.error_message or "")


def test_grounding_ledger_combines_direct_and_reviewer_claims(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    candidate = (
        "DNS maps a human-readable domain name to an IP address. "
        "Predicate logic is essential to all computer science."
    )
    result = make_pipeline(
        model,
        profiles,
        FakeOllamaClient(writer_output=candidate),
    ).route(sample_request(GenerationProfileName.STANDARD))

    assert result.status is GenerationStatus.COMPLETED
    assert result.initial_grounding_review is not None
    assert result.initial_grounding_review.review is not None
    assert result.initial_grounding_review.review.decision is GroundingDecision.PASS
    claims = result.initial_grounding_review.review.claim_assessments
    assert [(claim.claim_id, claim.classification.value) for claim in claims] == [
        ("C001", "direct_supported"),
        ("C002", "supported"),
    ]
    assert result.initial_grounding_review.review.coverage_complete is True


def test_grounding_review_fails_closed_when_unresolved_claim_id_is_missing(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    response = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "C999",
                    "classification": "unsupported",
                    "reason": "The source does not support the claim.",
                    "evidence_ids": [],
                }
            ]
        }
    )
    result = make_pipeline(
        model,
        profiles,
        FakeOllamaClient(
            writer_output=(
                "DNS maps a human-readable domain name to an IP address. "
                "Predicate logic is essential to all computer science."
            ),
            grounding_responses=[response],
        ),
    ).route(sample_request(GenerationProfileName.STANDARD))

    assert result.status is GenerationStatus.FAILED
    assert result.final_output is None
    assert result.initial_grounding_review is not None
    assert result.initial_grounding_review.status is GenerationStatus.FAILED
    assert "missing C002" in (result.initial_grounding_review.error_message or "")
    assert "unknown C999" in (result.initial_grounding_review.error_message or "")


@pytest.mark.parametrize(
    ("profile", "sources"),
    [
        (GenerationProfileName.QUICK, True),
        (GenerationProfileName.STANDARD, False),
    ],
)
def test_grounding_review_is_inactive_for_quick_or_source_free_generation(
    configs: tuple[Any, Any],
    profile: GenerationProfileName,
    sources: bool,
) -> None:
    model, profiles = configs
    fake = FakeOllamaClient()
    request = sample_request(profile)
    if not sources:
        request = request.model_copy(update={"source_material": []})

    result = make_pipeline(model, profiles, fake).route(request)

    assert result.status is GenerationStatus.COMPLETED
    assert result.initial_grounding_review is None
    assert result.revision_result is None
    assert result.final_grounding_review is None
    assert fake.grounding_response_count == 0
    assert result.final_output == result.writer_result.raw_response


def test_generation_keeps_model_defaults_separate_from_profile(configs: tuple[Any, Any]) -> None:
    model, profiles = configs
    original_defaults = model.model_dump(mode="json")
    result = make_pipeline(model, profiles, FakeOllamaClient()).route(
        sample_request(GenerationProfileName.STANDARD)
    )
    assert model.model_dump(mode="json") == original_defaults
    assert result.model_defaults["inference"]["context_length"] == 32768
    assert result.model_defaults["inference"]["think"] is True
    assert result.profile_configuration["writer"]["context_tiers"] == (
        profiles.profiles[GenerationProfileName.STANDARD].writer.context_tiers
    )
    assert result.selected_contexts["planner"] == 4096
    assert (
        result.selected_contexts["writer"]
        in result.profile_configuration["writer"]["context_tiers"]
    )
    assert result.estimated_input_tokens["planner"] > 0


def test_context_selection_chooses_smallest_tier_with_margin() -> None:
    selection = select_context_tier(
        "x" * 10_000,
        [8192, 16384, 32768],
        safety_margin=1.5,
    )
    assert selection.estimated_input_tokens == 2500
    assert selection.required_tokens_with_margin == 3750
    assert selection.selected_context == 8192


def test_context_selection_reserves_output_budget() -> None:
    selection = select_context_tier(
        "x" * 10_000,
        [4096, 8192],
        safety_margin=1.5,
        output_reserve_tokens=4096,
    )
    assert selection.required_tokens_with_margin == 7846
    assert selection.selected_context == 8192


def test_quick_context_overflow_is_explicit_not_silently_raised(configs: tuple[Any, Any]) -> None:
    model, profiles = configs
    huge_request = sample_request(GenerationProfileName.QUICK).model_copy(
        update={"instruction": "Explain. " + "DNS context. " * 3000}
    )
    result = make_pipeline(model, profiles, FakeOllamaClient()).route(huge_request)
    assert result.status is GenerationStatus.FAILED
    assert result.writer_result is not None
    assert result.writer_result.error_type == "ValueError"
    maximum_context = max(profiles.profiles[GenerationProfileName.QUICK].writer.context_tiers)
    assert f"configured tiers stop at {maximum_context}" in (
        result.writer_result.error_message or ""
    )


def test_profiles_config_has_quick_standard_deep_and_reviewer_disabled(
    configs: tuple[Any, Any],
) -> None:
    _, profiles = configs
    assert set(profiles.profiles) == {
        GenerationProfileName.QUICK,
        GenerationProfileName.STANDARD,
        GenerationProfileName.DEEP,
    }
    assert profiles.profiles[GenerationProfileName.QUICK].planner.enabled is False
    assert profiles.profiles[GenerationProfileName.QUICK].planner.think is False
    assert profiles.profiles[GenerationProfileName.STANDARD].planner.enabled is True
    assert profiles.profiles[GenerationProfileName.DEEP].planner.enabled is True
    assert (
        profiles.profiles[GenerationProfileName.DEEP].planner.task_output_tokens[
            TaskType.EXPLANATION
        ]
        == 3072
    )
    assert profiles.profiles[GenerationProfileName.DEEP].review_enabled is False
    assert profiles.profiles[GenerationProfileName.DEEP].fallback_to_writer is False


def test_generation_run_persistence_is_opt_in_and_stage_separated(
    configs: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    model, profiles = configs
    retrieval = {
        "query": "DNS",
        "original_query": "Can you explain DNS?",
        "canonical_query": "DNS",
        "matches": [
            {
                "chunk_id": "dns-notes:0",
                "source_id": "dns-notes",
                "source_title": "DNS notes",
                "fused_score": 0.82,
                "neighbor_of": None,
            }
        ],
        "warnings": ["source resolution was exact"],
        "diagnostics": {"ranking": "hybrid"},
    }
    request = sample_request(GenerationProfileName.STANDARD).model_copy(
        update={"metadata": {"knowledge_retrieval": retrieval}}
    )
    result = make_pipeline(model, profiles, FakeOllamaClient()).route(request)
    run_dir = create_generation_run_directory(tmp_path, request.request_id)
    save_generation_run(run_dir, request, result)

    assert {path.name for path in run_dir.iterdir()} == {
        "request.json",
        "retrieval.json",
        "sources.json",
        "sources.md",
        "planner_prompt.json",
        "plan.json",
        "writer_prompt.json",
        "writer_output.md",
        "grounding_review.json",
        "grounding_review_initial.json",
        "grounding_review_initial_prompt.json",
        "result.json",
        "output.md",
        "diagnostics.md",
    }
    assert json.loads((run_dir / "retrieval.json").read_text(encoding="utf-8")) == retrieval
    saved_sources = json.loads((run_dir / "sources.json").read_text(encoding="utf-8"))
    assert saved_sources == [source.model_dump(mode="json") for source in request.source_material]
    source_markdown = (run_dir / "sources.md").read_text(encoding="utf-8")
    assert request.source_material[0].text in source_markdown
    assert request.source_material[0].title in source_markdown
    assert request.source_material[0].source_id in source_markdown

    saved_planner_prompt = json.loads((run_dir / "planner_prompt.json").read_text(encoding="utf-8"))
    expected_planner_prompt = build_planner_prompt(request, concise=True)
    assert saved_planner_prompt["system_message"] == expected_planner_prompt.system_message
    assert saved_planner_prompt["user_message"] == expected_planner_prompt.user_message
    assert saved_planner_prompt["prompt_version"] == expected_planner_prompt.version
    assert saved_planner_prompt["selected_context"] == result.planner_result.timing.selected_context
    assert (
        saved_planner_prompt["estimated_input_tokens"]
        == result.planner_result.timing.estimated_input_tokens
    )
    assert saved_planner_prompt["output_budget"] == result.planner_result.timing.output_budget

    saved_writer_prompt = json.loads((run_dir / "writer_prompt.json").read_text(encoding="utf-8"))
    expected_writer_prompt = build_writer_prompt(request, result.planner_result.plan)
    assert saved_writer_prompt["system_message"] == expected_writer_prompt.system_message
    assert saved_writer_prompt["user_message"] == expected_writer_prompt.user_message
    assert saved_writer_prompt["prompt_version"] == expected_writer_prompt.version
    assert saved_writer_prompt["selected_context"] == result.writer_result.timing.selected_context
    assert (
        saved_writer_prompt["estimated_input_tokens"]
        == result.writer_result.timing.estimated_input_tokens
    )
    assert saved_writer_prompt["output_budget"] == result.writer_result.timing.output_budget

    saved_result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert saved_result["planner_result"]["plan"]["task"] == "lecture"
    assert saved_result["planner_result"]["plan"]["source_scope"]["status"] == "sufficient"
    assert (run_dir / "output.md").read_text(encoding="utf-8") == result.final_output
    assert (
        json.loads((run_dir / "grounding_review_initial.json").read_text(encoding="utf-8"))[
            "review"
        ]["decision"]
        == "pass"
    )
    saved_review_prompt = json.loads(
        (run_dir / "grounding_review_initial_prompt.json").read_text(encoding="utf-8")
    )
    assert saved_review_prompt["prompt_version"] == "grounding-review-v6"
    assert "Unresolved factual claims to adjudicate" in saved_review_prompt["user_message"]
    saved_review = json.loads((run_dir / "grounding_review.json").read_text(encoding="utf-8"))
    saved_claim = saved_review["review"]["claim_assessments"][0]
    assert saved_claim["claim_id"] == "C001"
    assert saved_claim["text"] == "DNS maps a domain name to an IP address."
    assert saved_claim["classification"] == "supported"
    assert saved_claim["support_method"] == "reviewer_entailment"
    assert saved_claim["evidence_ids"] == ["S01-E001"]
    diagnostics = (run_dir / "diagnostics.md").read_text(encoding="utf-8")
    assert "- Retrieval enabled: yes" in diagnostics
    assert "- Original instruction: Can you explain DNS?" in diagnostics
    assert "- Canonical query: DNS" in diagnostics
    assert "- Status: sufficient" in diagnostics
    assert "- Supported topics:" in diagnostics
    assert "- Number of retrieved matches: 1" in diagnostics
    assert "- Number of assembled source materials: 1" in diagnostics
    assert "Initial grounding review" in diagnostics
    assert "- Claims extracted: 1" in diagnostics
    assert "- Direct source matches: 0" in diagnostics
    assert "- Reviewer-supported claims: 1" in diagnostics
    assert "- Evidence validation failures / unknown evidence IDs: 0" in diagnostics
    assert "- Coverage complete: yes" in diagnostics


def test_failed_final_grounding_review_persists_both_candidates_and_prompts(
    configs: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    model, profiles = configs
    original = "The primary motivation for predicate logic was to simplify the web."
    revised = "Predicate logic is essential to computing."
    request = sample_request(GenerationProfileName.STANDARD)
    fake = FakeOllamaClient(
        writer_output=original,
        revision_output=revised,
        grounding_responses=[
            _revision_required_review(),
            _revision_required_review("Predicate logic is essential to computing."),
        ],
    )
    result = make_pipeline(model, profiles, fake).route(request)
    run_dir = create_generation_run_directory(tmp_path, request.request_id)
    save_generation_run(run_dir, request, result)

    names = {path.name for path in run_dir.iterdir()}
    assert result.status is GenerationStatus.FAILED
    assert {
        "writer_output.md",
        "grounding_review_initial.json",
        "grounding_review_initial_prompt.json",
        "grounding_revision.json",
        "grounding_revision_prompt.json",
        "revised_output.md",
        "grounding_review_final.json",
        "grounding_review_final_prompt.json",
    } <= names
    assert "output.md" not in names
    assert (run_dir / "writer_output.md").read_text(encoding="utf-8") == original
    assert (run_dir / "revised_output.md").read_text(encoding="utf-8") == revised
    assert "Predicate logic is essential to computing." in (
        run_dir / "grounding_review_final.json"
    ).read_text(encoding="utf-8")


def test_quick_persistence_saves_writer_prompt_without_planner_files(
    configs: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    model, profiles = configs
    request = sample_request(GenerationProfileName.QUICK)
    fake = FakeOllamaClient()
    result = make_pipeline(model, profiles, fake).route(request)
    run_dir = create_generation_run_directory(tmp_path, request.request_id)
    save_generation_run(run_dir, request, result)

    assert "writer_prompt.json" in {path.name for path in run_dir.iterdir()}
    assert "planner_prompt.json" not in {path.name for path in run_dir.iterdir()}
    assert "plan.json" not in {path.name for path in run_dir.iterdir()}
    assert "retrieval.json" not in {path.name for path in run_dir.iterdir()}
    saved_prompt = json.loads((run_dir / "writer_prompt.json").read_text(encoding="utf-8"))
    expected_prompt = build_writer_prompt(request, None)
    assert saved_prompt["system_message"] == expected_prompt.system_message
    assert saved_prompt["user_message"] == expected_prompt.user_message
    assert "- Retrieval enabled: no" in (run_dir / "diagnostics.md").read_text(encoding="utf-8")


def test_failed_planner_persistence_keeps_available_diagnostics(
    configs: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    model, profiles = configs
    request = sample_request(GenerationProfileName.STANDARD)
    result = make_pipeline(
        model,
        profiles,
        FakeOllamaClient(planner_failure="invalid_json"),
    ).route(request)
    run_dir = create_generation_run_directory(tmp_path, request.request_id)
    save_generation_run(run_dir, request, result)

    names = {path.name for path in run_dir.iterdir()}
    assert "planner_prompt.json" in names
    assert "plan.json" not in names
    assert "writer_prompt.json" not in names
    assert "output.md" not in names
    assert "result.json" in names
    assert "diagnostics.md" in names
    assert "Writer" in (run_dir / "diagnostics.md").read_text(encoding="utf-8")


def test_failed_writer_persistence_keeps_plan_and_prompts(
    configs: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    model, profiles = configs
    request = sample_request(GenerationProfileName.STANDARD)
    result = make_pipeline(model, profiles, FakeOllamaClient(writer_failure=True)).route(request)
    run_dir = create_generation_run_directory(tmp_path, request.request_id)
    save_generation_run(run_dir, request, result)

    names = {path.name for path in run_dir.iterdir()}
    assert result.status is GenerationStatus.FAILED
    assert {"planner_prompt.json", "plan.json", "writer_prompt.json", "result.json"} <= names
    assert "output.md" not in names
    assert "diagnostics.md" in names
    assert "- Status: failed" in (run_dir / "diagnostics.md").read_text(encoding="utf-8")
