from pathlib import Path
from typing import Any

from lecture_slm.config.loader import load_model_config
from lecture_slm.generation.models import (
    GenerationRequest,
    GenerationResult,
    GenerationStage,
    GenerationStatus,
    ProgressEvent,
    SourceMaterial,
    StageTiming,
)
from lecture_slm.generation.profiles import load_generation_profiles
from lecture_slm.generation.service import GenerationService, RetrievalOptions
from lecture_slm.knowledge.config import KnowledgeConfig
from lecture_slm.knowledge.models import RetrievalResult
from lecture_slm.schemas.dataset import TaskType

ROOT = Path(__file__).parents[1]


def _result(request: GenerationRequest) -> GenerationResult:
    return GenerationResult(
        request_id=request.request_id,
        task=request.task,
        profile=request.profile,
        model="test-model",
        status=GenerationStatus.COMPLETED,
        final_output="Generated output.",
        timing=StageTiming(duration_seconds=1.5),
        planner_prompt_version=None,
        writer_prompt_version="writer-v1",
    )


class FakeRouter:
    def __init__(self, **_: Any) -> None:
        self.request: GenerationRequest | None = None

    def route(self, request: GenerationRequest, *, on_progress: Any = None) -> GenerationResult:
        self.request = request
        if on_progress is not None:
            on_progress(
                ProgressEvent(
                    stage=GenerationStage.COMPLETE,
                    message="Generation complete",
                    elapsed_seconds=1.5,
                )
            )
        return _result(request)


def _service(router: FakeRouter) -> GenerationService:
    model = load_model_config(ROOT / "configs/models/qwen35-9b.yaml")
    profiles = load_generation_profiles(ROOT / "configs/generation/profiles.yaml")
    return GenerationService(
        model_config=model,
        profiles=profiles,
        router_factory=lambda **kwargs: router,
    )


def test_generation_service_routes_the_request_through_the_pipeline() -> None:
    request = GenerationRequest(task=TaskType.EXPLANATION, instruction="Explain predicate logic.")
    router = FakeRouter()
    execution = _service(router).generate(request)

    assert router.request is request
    assert execution.request is request
    assert execution.result.final_output == "Generated output."
    assert execution.saved_run_directory is None


def test_generation_service_uses_shared_retrieval_and_saves_existing_run_format(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source = SourceMaterial(
        source_id="retrieved-1",
        title="Predicate Logic",
        text="Predicate logic represents properties and relations.",
    )
    router = FakeRouter()
    service = _service(router)
    service.artifact_root = tmp_path / "runs"
    knowledge = KnowledgeConfig()
    options = RetrievalOptions(enabled=True, top_k=2, course="CSC220", tags=["logic"])
    seen: dict[str, Any] = {}

    monkeypatch.setattr(
        "lecture_slm.generation.service.load_knowledge_config",
        lambda _: knowledge,
    )

    def retrieve(
        request: GenerationRequest,
        retrieval_options: RetrievalOptions,
        _: KnowledgeConfig,
    ) -> RetrievalResult:
        seen["request"] = request
        seen["options"] = retrieval_options
        return RetrievalResult(query=request.instruction)

    monkeypatch.setattr(service, "_retrieve", retrieve)
    monkeypatch.setattr(
        "lecture_slm.generation.service.KnowledgeContextAssembler.assemble",
        lambda self, result, *, source_context_budget: [source],
    )

    request = GenerationRequest(task=TaskType.EXPLANATION, instruction="Explain predicate logic.")
    execution = service.generate(request, retrieval=options, save_run=True)

    assert seen["request"] is request
    assert seen["options"] is options
    assert router.request is not None
    assert router.request.source_material == [source]
    assert router.request.metadata["knowledge_retrieval"]["query"] == request.instruction
    assert execution.saved_run_directory is not None
    assert execution.retrieved_count == 0
    assert (execution.saved_run_directory / "request.json").is_file()
    assert (execution.saved_run_directory / "result.json").is_file()
    assert (execution.saved_run_directory / "diagnostics.md").is_file()
