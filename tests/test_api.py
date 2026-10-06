import re
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from lecture_slm.api.app import create_app
from lecture_slm.config.loader import load_model_config
from lecture_slm.generation.models import (
    GenerationProfileName,
    GenerationRequest,
    GenerationResult,
    GenerationStage,
    GenerationStatus,
    GroundingClaimStatus,
    GroundingDecision,
    GroundingIssueCategory,
    GroundingReview,
    GroundingReviewRecord,
    ProgressEvent,
    StageRecord,
    StageTiming,
)
from lecture_slm.generation.profiles import load_generation_profiles
from lecture_slm.generation.service import GenerationExecution, RetrievalOptions
from lecture_slm.inference.ollama_client import OllamaConnectionError
from lecture_slm.schemas.dataset import TaskType

ROOT = Path(__file__).parents[1]


class FakeService:
    def __init__(
        self,
        *,
        status: GenerationStatus = GenerationStatus.COMPLETED,
        retrieved_count: int = 0,
    ) -> None:
        self.model_config = load_model_config(ROOT / "configs/models/qwen35-9b.yaml")
        self.profiles = load_generation_profiles(ROOT / "configs/generation/profiles.yaml")
        self.status = status
        self.retrieved_count = retrieved_count
        self.calls: list[tuple[GenerationRequest, RetrievalOptions | None, bool]] = []

    def knowledge_index_available(self) -> bool:
        return True

    def generate(
        self,
        request: GenerationRequest,
        *,
        retrieval: RetrievalOptions | None = None,
        save_run: bool = False,
        on_progress: Any = None,
    ) -> GenerationExecution:
        self.calls.append((request, retrieval, save_run))
        stages = (
            [GenerationStage.FAILED]
            if self.status is GenerationStatus.FAILED
            else [
                GenerationStage.PREPARING,
                GenerationStage.PLANNING,
                GenerationStage.WRITING,
                GenerationStage.REVIEWING,
                GenerationStage.COMPLETE,
            ]
        )
        if on_progress is not None:
            for index, stage in enumerate(stages):
                on_progress(
                    ProgressEvent(
                        stage=stage,
                        message=f"{stage.value} request",
                        elapsed_seconds=float(index),
                    )
                )

        review = None
        revision = None
        errors = []
        if self.status is GenerationStatus.FAILED:
            excerpt = "An unsupported claim."
            review = GroundingReviewRecord(
                status=GenerationStatus.COMPLETED,
                review=GroundingReview(
                    decision=GroundingDecision.REVISION_REQUIRED,
                    claim_assessments=[
                        {
                            "excerpt": excerpt,
                            "status": GroundingClaimStatus.UNSUPPORTED,
                        }
                    ],
                    issues=[
                        {
                            "excerpt": excerpt,
                            "category": GroundingIssueCategory.UNSUPPORTED_FACT,
                            "reason": "No source supports this claim.",
                        }
                    ],
                    revision_instructions=["Remove the unsupported claim."],
                ),
            )
            revision = StageRecord(
                status=GenerationStatus.COMPLETED,
                timing=StageTiming(duration_seconds=0.5),
                raw_response="Revised candidate",
            )
            errors.append("Final grounding review still requires revision")
        result = GenerationResult(
            request_id=request.request_id,
            task=request.task,
            profile=request.profile,
            model="local-test-model",
            status=self.status,
            final_grounding_review=review,
            revision_result=revision,
            final_output="Approved output" if self.status is GenerationStatus.COMPLETED else None,
            timing=StageTiming(duration_seconds=4.0),
            errors=errors,
            planner_prompt_version=None,
            writer_prompt_version="writer-v1",
        )
        return GenerationExecution(request, result, None, self.retrieved_count)


def test_tasks_profiles_and_health_use_configured_values(monkeypatch: Any) -> None:
    service = FakeService()

    class FakeOllama:
        def __init__(self, host: str, *, timeout: float) -> None:
            assert host == service.model_config.inference.host
            assert timeout == 2.0

        def available_models(self) -> set[str]:
            return {service.model_config.ollama_name}

    monkeypatch.setattr("lecture_slm.api.app.OllamaClient", FakeOllama)
    client = TestClient(create_app(service))

    tasks = client.get("/api/tasks")
    profiles = client.get("/api/profiles")
    health = client.get("/api/health")

    assert tasks.status_code == 200
    assert tasks.json()["tasks"] == [task.value for task in TaskType]
    assert profiles.status_code == 200
    assert {item["id"] for item in profiles.json()["profiles"]} == {
        profile.value for profile in GenerationProfileName
    }
    assert health.status_code == 200
    assert health.json()["generation_ready"] is True
    assert health.json()["ollama_reachable"] is True
    assert health.json()["model_available"] is True
    assert health.json()["knowledge_index_available"] is True


def test_health_reports_api_alive_when_ollama_is_unavailable(monkeypatch: Any) -> None:
    service = FakeService()

    class UnavailableOllama:
        def __init__(self, host: str, *, timeout: float) -> None:
            pass

        def available_models(self) -> set[str]:
            raise OllamaConnectionError("not reachable")

    monkeypatch.setattr("lecture_slm.api.app.OllamaClient", UnavailableOllama)
    response = TestClient(create_app(service)).get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["generation_ready"] is False
    assert response.json()["ollama_reachable"] is False


def test_generate_maps_request_and_returns_successful_result() -> None:
    service = FakeService(retrieved_count=2)
    app = create_app(service)
    client = TestClient(app)
    body = {
        "task": "explanation",
        "profile": "standard",
        "instruction": "Explain predicate logic.",
        "retrieve": True,
        "retrieval_top_k": 2,
        "knowledge_course": "CSC220",
        "tag": ["logic"],
        "save_run": True,
        "metadata": {
            "caller": "test",
            "knowledge_retrieval": {"matches": [{"source_id": "untrusted"}]},
        },
    }

    response = client.post("/api/generate", json=body)

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "completed"
    assert result["output"] == "Approved output"
    assert result["task"] == "explanation"
    assert result["sources"]["retrieved"] == 2
    assert result["sources"]["assembled"] == 0
    request, retrieval, save_run = service.calls[0]
    assert request.instruction == body["instruction"]
    assert request.metadata == body["metadata"]
    assert retrieval == RetrievalOptions(
        enabled=True,
        top_k=2,
        course="CSC220",
        tags=["logic"],
    )
    assert save_run is True
    assert client.get(f"/api/runs/{result['request_id']}").json() == result


def test_default_profile_comes_from_generation_configuration() -> None:
    service = FakeService()
    service.profiles.default_profile = GenerationProfileName.DEEP
    client = TestClient(create_app(service))

    response = client.post(
        "/api/generate",
        json={"task": "explanation", "instruction": "Explain predicate logic."},
    )

    assert response.status_code == 200
    assert response.json()["profile"] == "deep"
    assert service.calls[0][0].profile is GenerationProfileName.DEEP


def test_pipeline_grounding_failure_is_a_normal_response() -> None:
    client = TestClient(create_app(FakeService(status=GenerationStatus.FAILED)))

    response = client.post(
        "/api/generate",
        json={
            "task": "explanation",
            "profile": "standard",
            "instruction": "Explain predicate logic.",
            "retrieve": True,
            "retrieval_top_k": 2,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "failed"
    assert result["output"] is None
    assert result["errors"] == ["Final grounding review still requires revision"]
    assert result["grounding"]["reviewed"] is True
    assert result["grounding"]["revision_performed"] is True
    assert result["grounding"]["final_decision"] == "revision_required"


def test_invalid_task_and_profile_are_rejected() -> None:
    client = TestClient(create_app(FakeService()))
    for field, value in (("task", "unknown"), ("profile", "unknown")):
        response = client.post(
            "/api/generate",
            json={"task": "explanation", "profile": "standard", "instruction": "Test."}
            | {field: value},
        )
        assert response.status_code == 422


def test_stream_orders_progress_then_result_for_success_and_pipeline_failure() -> None:
    for status in (GenerationStatus.COMPLETED, GenerationStatus.FAILED):
        client = TestClient(create_app(FakeService(status=status)))
        response = client.post(
            "/api/generate/stream",
            json={"task": "explanation", "profile": "standard", "instruction": "Test."},
        )

        assert response.status_code == 200
        event_names = re.findall(r"event: (\w+)", response.text)
        assert event_names[-1] == "result"
        assert "progress" in event_names
        if status is GenerationStatus.FAILED:
            assert '"stage":"failed"' in response.text
            assert '"status":"failed"' in response.text
            assert "event: error" not in response.text
        else:
            assert event_names[0] == "progress"
            assert '"stage":"preparing"' in response.text


def test_unknown_run_and_bounded_registry() -> None:
    service = FakeService()
    client = TestClient(create_app(service, max_retained_runs=2))
    first_id = None
    for index in range(3):
        response = client.post(
            "/api/generate",
            json={
                "task": "explanation",
                "profile": "standard",
                "instruction": f"Request {index}",
            },
        )
        assert response.status_code == 200
        if index == 0:
            first_id = response.json()["request_id"]
    assert first_id is not None
    assert client.get(f"/api/runs/{first_id}").status_code == 404
    assert client.get("/api/runs/unknown").status_code == 404
