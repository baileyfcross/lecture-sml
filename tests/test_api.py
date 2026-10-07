import re
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from lecture_slm.api.app import create_app
from lecture_slm.api.models import GenerateResponse
from lecture_slm.config.loader import load_model_config
from lecture_slm.generation.models import (
    ExplanationPlan,
    GenerationProfileName,
    GenerationRequest,
    GenerationResult,
    GenerationStage,
    GenerationStatus,
    GroundingClaimClassification,
    GroundingDecision,
    GroundingIssueCategory,
    GroundingReview,
    GroundingReviewRecord,
    GroundingSupportMethod,
    ProgressEvent,
    SourceScopeAssessment,
    SourceScopeStatus,
    StageRecord,
    StageTiming,
)
from lecture_slm.generation.profiles import load_generation_profiles
from lecture_slm.generation.service import GenerationExecution, RetrievalOptions
from lecture_slm.inference.ollama_client import OllamaConnectionError
from lecture_slm.schemas.dataset import TaskType
from lecture_slm.workspaces.service import WorkspaceService

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
                            "claim_id": "C001",
                            "text": excerpt,
                            "classification": GroundingClaimClassification.UNSUPPORTED,
                            "support_method": GroundingSupportMethod.UNSUPPORTED,
                            "reason": "No source supports this claim.",
                        }
                    ],
                    issues=[
                        {
                            "claim_id": "C001",
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


def test_workspace_routes_save_only_completed_output_as_history(tmp_path: Path) -> None:
    service = FakeService()
    service.workspaces = WorkspaceService(database_path=tmp_path / "workspaces.sqlite")
    client = TestClient(create_app(service))
    created = client.post("/api/workspaces", json={"name": "CSC 220"})
    assert created.status_code == 201
    workspace_id = created.json()["id"]
    folder = client.post(
        f"/api/workspaces/{workspace_id}/folders",
        json={"name": "Week 1"},
    )
    assert folder.status_code == 201

    context = client.post(
        f"/api/workspaces/{workspace_id}/items",
        json={
            "title": "Class conventions",
            "role": "context",
            "content": "Use truth tables before natural deduction.",
            "folder_id": folder.json()["id"],
        },
    )
    assert context.status_code == 201
    rejected_raw_context = client.post(
        "/api/generate",
        json={
            "task": "explanation",
            "instruction": "Explain a proposition.",
            "workspace_id": workspace_id,
            "workspace_context": {"workspace_name": "untrusted", "items": []},
        },
    )
    assert rejected_raw_context.status_code == 422
    generation = client.post(
        "/api/generate",
        json={
            "task": "explanation",
            "instruction": "Explain a proposition.",
            "workspace_id": workspace_id,
        },
    )
    assert generation.status_code == 200
    assert service.calls[-1][0].workspace_id == workspace_id
    saved = client.post(
        f"/api/workspaces/{workspace_id}/history",
        json={
            "request_id": generation.json()["request_id"],
            "title": "Approved proposition explanation",
        },
    )
    assert saved.status_code == 201
    assert saved.json()["role"] == "history"
    assert saved.json()["content"] == "Approved output"
    promoted = client.patch(
        f"/api/workspaces/{workspace_id}/items/{context.json()['id']}",
        json={"role": "reference"},
    )
    assert promoted.status_code == 200
    assert promoted.json()["role"] == "reference"

    failed_service = FakeService(status=GenerationStatus.FAILED)
    failed_service.workspaces = service.workspaces
    failed_client = TestClient(create_app(failed_service))
    failed_run = failed_client.post(
        "/api/generate",
        json={"task": "explanation", "instruction": "Unsupported."},
    )
    assert failed_run.status_code == 200
    rejected_save = failed_client.post(
        f"/api/workspaces/{workspace_id}/history",
        json={"request_id": failed_run.json()["request_id"], "title": "Rejected"},
    )
    assert rejected_save.status_code == 409
    roles = {item["role"] for item in client.get(f"/api/workspaces/{workspace_id}").json()["items"]}
    assert roles == {"reference", "history"}
    assert client.get("/api/workspaces/not-a-workspace").status_code == 404


def test_generate_response_includes_source_coverage_assessment() -> None:
    execution = FakeService().generate(
        GenerationRequest(
            task=TaskType.EXPLANATION,
            profile=GenerationProfileName.STANDARD,
            instruction="Explain quantum computing.",
        )
    )
    plan = ExplanationPlan(
        task=TaskType.EXPLANATION,
        artifact_structure=["Quantum annealing"],
        notes_for_writer=[],
        source_scope=SourceScopeAssessment(
            status=SourceScopeStatus.PARTIAL,
            supported_topics=["quantum annealing"],
            unsupported_requested_topics=["error correction"],
            scope_note=(
                "The available materials support quantum annealing, but not a complete "
                "treatment of error correction."
            ),
        ),
        concept="Quantum computing",
        assumed_knowledge=[],
        explanation_sequence=["Quantum annealing"],
        example="A simple optimization example",
        misconceptions=[],
        check_for_understanding=["Describe quantum annealing."],
    )
    result = execution.result.model_copy(
        update={
            "planner_result": StageRecord(
                status=GenerationStatus.COMPLETED,
                plan=plan,
            )
        }
    )

    response = GenerateResponse.from_execution(
        GenerationExecution(
            execution.request,
            result,
            execution.saved_run_directory,
            execution.retrieved_count,
        )
    )

    assert response.model_dump(mode="json")["source_coverage"] == {
        "status": "partial",
        "supported_topics": ["quantum annealing"],
        "unsupported_requested_topics": ["error correction"],
        "scope_note": (
            "The available materials support quantum annealing, but not a complete "
            "treatment of error correction."
        ),
    }


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


def test_initial_grounding_decision_is_not_reported_as_final() -> None:
    request = GenerationRequest(
        task=TaskType.EXPLANATION,
        profile=GenerationProfileName.STANDARD,
        instruction="Explain predicate logic.",
    )
    execution = FakeService(status=GenerationStatus.FAILED).generate(request)
    result = execution.result
    assert result.final_grounding_review is not None
    initial_only_result = result.model_copy(
        update={
            "initial_grounding_review": result.final_grounding_review,
            "final_grounding_review": None,
        }
    )
    response = GenerateResponse.from_execution(
        GenerationExecution(
            execution.request,
            initial_only_result,
            execution.saved_run_directory,
            execution.retrieved_count,
        )
    )

    assert response.grounding.initial_decision == "revision_required"
    assert response.grounding.final_decision is None


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


def test_frontend_build_missing_keeps_api_and_docs_available(tmp_path: Path) -> None:
    client = TestClient(create_app(FakeService(), frontend_dist=tmp_path))

    root = client.get("/")
    assert root.status_code == 200
    assert "web UI has not been built" in root.text
    assert client.get("/api/tasks").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_built_frontend_is_served_without_capturing_api_routes(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (tmp_path / "index.html").write_text(
        "<!doctype html><html><body><main>Lecture SLM UI</main></body></html>",
        encoding="utf-8",
    )
    (assets / "app.css").write_text("body { color: black; }", encoding="utf-8")
    client = TestClient(create_app(FakeService(), frontend_dist=tmp_path))

    assert "Lecture SLM UI" in client.get("/").text
    assert client.get("/assets/app.css").text == "body { color: black; }"
    assert client.get("/api/tasks").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
