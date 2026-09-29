import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lecture_slm.config.loader import load_model_config
from lecture_slm.generation.context import select_context_tier
from lecture_slm.generation.models import (
    GenerationProfileName,
    GenerationRequest,
    GenerationStage,
    GenerationStatus,
    PreviousCourseContext,
    SourceMaterial,
    TeachingPlan,
)
from lecture_slm.generation.persistence import create_generation_run_directory, save_generation_run
from lecture_slm.generation.profiles import load_generation_profiles
from lecture_slm.generation.router import GenerationRouter
from lecture_slm.inference.ollama_client import ChatResponse, OllamaTimeoutError
from lecture_slm.schemas.course import CourseProfile
from lecture_slm.schemas.dataset import TaskType
from lecture_slm.schemas.pedagogy import PedagogyPrinciple, PedagogyProfile

ROOT = Path(__file__).parents[1]


class FakeOllamaClient:
    def __init__(
        self,
        *,
        planner_failure: str | None = None,
        writer_failure: bool = False,
        writer_tokens: int = 64,
    ) -> None:
        self.planner_failure = planner_failure
        self.writer_failure = writer_failure
        self.writer_tokens = writer_tokens
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
            if self.planner_failure == "timeout":
                raise OllamaTimeoutError("planner timed out")
            if self.planner_failure == "invalid_json":
                content = "not valid JSON"
            elif self.planner_failure == "invalid_schema":
                content = json.dumps({"task": "lecture", "sequence": [{"unexpected": True}]})
            else:
                content = json.dumps(
                    {
                        "task": "lecture",
                        "objectives": ["Explain DNS name resolution."],
                        "prerequisites": ["domain names and IP addresses"],
                        "prior_knowledge_connections": ["URLs contain domain names"],
                        "sequence": [
                            {
                                "title": "Concrete lookup",
                                "purpose": "Introduce DNS through an example.",
                                "concepts": ["resolver", "IP address"],
                                "learner_action": "Trace a lookup.",
                            }
                        ],
                        "concepts": ["DNS maps names to addresses."],
                        "examples": [],
                        "misconceptions": ["DNS is the website itself."],
                        "practice": [],
                        "assessment_checks": ["What does DNS return?"],
                        "synthesis": ["Connect names to network addresses."],
                        "source_usage": ["Use supplied source only."],
                        "artifact_structure": ["objectives", "example", "practice"],
                        "notes_for_writer": ["Keep this freshman-friendly."],
                    }
                )
            return ChatResponse(
                model=model,
                content=content,
                prompt_tokens=120,
                completion_tokens=80,
                total_duration_ns=3_000_000_000,
                prompt_eval_duration_ns=1_000_000_000,
                eval_duration_ns=2_000_000_000,
                completion_reason="stop",
            )
        if self.writer_failure:
            raise OllamaTimeoutError("writer timed out")
        return ChatResponse(
            model=model,
            content="# DNS: a concise explanation\n\nA domain name maps to an IP address.",
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


def sample_request(profile: GenerationProfileName) -> GenerationRequest:
    return GenerationRequest(
        task=TaskType.LECTURE,
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
    assert fake.timeouts == [30.0, 300.0]
    assert [event.stage for event in events] == [
        GenerationStage.PREPARING,
        GenerationStage.WRITING,
        GenerationStage.WRITING,
        GenerationStage.COMPLETE,
    ]
    assert events[1].estimate_is_approximate is True
    assert events[1].estimate_seconds_remaining is not None


def test_standard_calls_planner_then_writer_with_structured_context(
    configs: tuple[Any, Any],
) -> None:
    model, profiles = configs
    fake = FakeOllamaClient()
    events = []
    result = make_pipeline(model, profiles, fake).route(
        sample_request(GenerationProfileName.STANDARD),
        on_progress=events.append,
    )

    assert result.status is GenerationStatus.COMPLETED
    assert result.planner_result is not None and result.planner_result.plan is not None
    assert len(fake.requests) == 2
    planner_request, writer_request = fake.requests
    assert planner_request["format"] is not None
    assert planner_request["think"] is True
    assert planner_request["options"]["num_predict"] == 3072
    assert profiles.profiles[GenerationProfileName.STANDARD].planner.context_tiers == [
        4096,
        8192,
    ]
    assert writer_request["format"] is None
    assert writer_request["think"] is False
    assert writer_request["options"]["num_predict"] == 2048
    assert fake.timeouts == [30.0, 1500.0, 900.0]
    assert "Supplied source material" in planner_request["user_message"]
    assert "Previous course context" in planner_request["user_message"]
    assert "Authoritative user request" in writer_request["user_message"]
    assert "Teaching plan to follow" in writer_request["user_message"]
    assert "Supplied source material" in writer_request["user_message"]
    assert result.planner_result.timing.generated_tokens == 80
    assert result.writer_result is not None
    assert result.writer_result.timing.generated_tokens == 64
    assert result.metadata["planner_prompt_version"] == "planner-v1"
    assert result.metadata["writer_prompt_version"] == "writer-v1"
    assert [event.stage for event in events] == [
        GenerationStage.PREPARING,
        GenerationStage.PLANNING,
        GenerationStage.PLANNING,
        GenerationStage.WRITING,
        GenerationStage.WRITING,
        GenerationStage.COMPLETE,
    ]


def test_deep_uses_larger_planner_and_context_tiers(configs: tuple[Any, Any]) -> None:
    model, profiles = configs
    fake = FakeOllamaClient()
    result = make_pipeline(model, profiles, fake).route(sample_request(GenerationProfileName.DEEP))

    assert result.status is GenerationStatus.COMPLETED
    assert fake.requests[0]["think"] is True
    assert fake.requests[0]["options"]["num_predict"] == 3072
    assert result.planner_result is not None
    assert result.planner_result.timing.selected_context == 8192
    assert fake.requests[1]["think"] is False
    assert fake.requests[1]["options"]["num_predict"] == 4096
    assert fake.timeouts == [30.0, 1500.0, 1800.0]
    assert result.writer_result is not None
    assert result.writer_result.timing.selected_context in {8192, 16384, 32768}
    assert profiles.profiles[GenerationProfileName.DEEP].review_enabled is False
    assert result.reviewer_result is None


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


def test_generation_keeps_model_defaults_separate_from_profile(configs: tuple[Any, Any]) -> None:
    model, profiles = configs
    original_defaults = model.model_dump(mode="json")
    result = make_pipeline(model, profiles, FakeOllamaClient()).route(
        sample_request(GenerationProfileName.STANDARD)
    )
    assert model.model_dump(mode="json") == original_defaults
    assert result.model_defaults["inference"]["context_length"] == 32768
    assert result.model_defaults["inference"]["think"] is True
    assert result.profile_configuration["writer"]["context_tiers"] == [8192]
    assert result.selected_contexts["planner"] == 4096
    assert result.selected_contexts["writer"] == 8192
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
    assert "configured tiers stop at 4096" in (result.writer_result.error_message or "")


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
    assert profiles.profiles[GenerationProfileName.STANDARD].planner.enabled is True
    assert profiles.profiles[GenerationProfileName.DEEP].planner.enabled is True
    assert profiles.profiles[GenerationProfileName.DEEP].review_enabled is False
    assert profiles.profiles[GenerationProfileName.DEEP].fallback_to_writer is False


def test_generation_run_persistence_is_opt_in_and_stage_separated(
    configs: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    model, profiles = configs
    request = sample_request(GenerationProfileName.STANDARD)
    result = make_pipeline(model, profiles, FakeOllamaClient()).route(request)
    run_dir = create_generation_run_directory(tmp_path, request.request_id)
    save_generation_run(run_dir, request, result)

    assert {path.name for path in run_dir.iterdir()} == {
        "request.json",
        "plan.json",
        "result.json",
        "output.md",
    }
    saved_result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert saved_result["planner_result"]["plan"]["task"] == "lecture"
    assert (run_dir / "output.md").read_text(encoding="utf-8") == result.final_output
