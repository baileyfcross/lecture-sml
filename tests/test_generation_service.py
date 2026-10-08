from pathlib import Path
from typing import Any

import pytest

from lecture_slm.config.loader import load_model_config
from lecture_slm.generation.models import (
    GenerationRequest,
    GenerationResult,
    GenerationStage,
    GenerationStatus,
    ProgressEvent,
    SourceMaterial,
    SourceScopeAssessment,
    SourceScopeStatus,
    StageTiming,
)
from lecture_slm.generation.profiles import load_generation_profiles
from lecture_slm.generation.prompts.base import request_blocks
from lecture_slm.generation.service import GenerationService, RetrievalOptions
from lecture_slm.knowledge.config import KnowledgeConfig
from lecture_slm.knowledge.models import RetrievalMatch, RetrievalResult
from lecture_slm.schemas.dataset import TaskType
from lecture_slm.workspaces.models import WorkspaceItemInput, WorkspaceItemRole
from lecture_slm.workspaces.service import WorkspaceService

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
    def __init__(self, **kwargs: Any) -> None:
        self.request: GenerationRequest | None = None
        self.expansion_callback: Any = None
        self.expand_scope: SourceScopeAssessment | None = kwargs.get("expand_scope")

    def route(
        self,
        request: GenerationRequest,
        *,
        on_progress: Any = None,
        expand_retrieval: Any = None,
    ) -> GenerationResult:
        self.expansion_callback = expand_retrieval
        if expand_retrieval is not None and self.expand_scope is not None:
            request = expand_retrieval(request, self.expand_scope) or request
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
    assert router.expansion_callback is None


def test_generation_materializes_reference_sources_and_continuity_only_workspace_items(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    router = FakeRouter()
    service = _service(router)
    service.artifact_root = tmp_path / "runs"
    service._workspace_service = WorkspaceService(database_path=tmp_path / "workspaces.sqlite")
    workspace = service.workspaces.store.create_workspace("CSC 220 Logic")
    context = service.workspaces.store.create_item(
        workspace.id,
        WorkspaceItemInput(
            title="Course terminology",
            role=WorkspaceItemRole.CONTEXT,
            content="Learners call truth tables valuation tables.",
        ),
    )
    history = service.workspaces.store.create_item(
        workspace.id,
        WorkspaceItemInput(
            title="Prior lesson",
            role=WorkspaceItemRole.HISTORY,
            content="Previously introduced propositions and connectives.",
        ),
    )
    reference = service.workspaces.store.create_item(
        workspace.id,
        WorkspaceItemInput(
            title="Logic reference",
            role=WorkspaceItemRole.REFERENCE,
            content="A proposition has a truth value.",
        ),
    )
    monkeypatch.setattr(
        "lecture_slm.generation.service.load_knowledge_config",
        lambda _: KnowledgeConfig(),
    )

    execution = service.generate(
        GenerationRequest(
            task=TaskType.EXPLANATION,
            instruction="Explain a proposition using a truth table.",
            workspace_id=workspace.id,
        ),
        save_run=True,
    )

    assert router.request is not None
    assert router.request.workspace_context is not None
    assert {item.id for item in router.request.workspace_context.items} == {
        context.id,
        history.id,
    }
    assert [
        material.metadata["workspace_item_id"] for material in router.request.source_material
    ] == [reference.id]
    prompt = "\n".join(request_blocks(router.request))
    assert "Course terminology" in prompt
    assert "Prior lesson" in prompt
    assert "truth value" in prompt
    assert router.request.source_material[0].metadata["source_origin"] == "workspace_reference"
    assert execution.saved_run_directory is not None
    snapshot_path = execution.saved_run_directory / "workspace_context.json"
    snapshot = snapshot_path.read_text(encoding="utf-8")
    assert context.id in snapshot and history.id in snapshot and reference.id in snapshot


def test_workspace_name_anchors_global_retrieval_without_replacing_canonical_query(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    router = FakeRouter()
    service = _service(router)
    service._workspace_service = WorkspaceService(database_path=tmp_path / "workspaces.sqlite")
    workspace = service.workspaces.store.create_workspace("CSC 220 Logic")
    knowledge = KnowledgeConfig()
    seen_queries: list[Any] = []
    monkeypatch.setattr(
        "lecture_slm.generation.service.load_knowledge_config",
        lambda _: knowledge,
    )

    def retrieve(
        request: GenerationRequest,
        retrieval_options: RetrievalOptions,
        _: KnowledgeConfig,
        *,
        query: Any,
        top_k: int | None = None,
    ) -> RetrievalResult:
        del request, retrieval_options, top_k
        seen_queries.append(query)
        return RetrievalResult(query=query.canonical)

    monkeypatch.setattr(service, "_retrieve", retrieve)

    execution = service.generate(
        GenerationRequest(
            task=TaskType.EXPLANATION,
            instruction="Explain predicate logic.",
            workspace_id=workspace.id,
        ),
        retrieval=RetrievalOptions(enabled=True),
    )

    assert len(seen_queries) == 1
    assert seen_queries[0].canonical == "CSC 220 Logic predicate logic"
    retrieval_diagnostics = execution.request.metadata["knowledge_retrieval"]
    assert retrieval_diagnostics["canonical_query"] == "predicate logic"
    assert (
        retrieval_diagnostics["diagnostics"]["workspace_anchored_query"]
        == seen_queries[0].canonical
    )


def test_explicit_sources_take_budget_before_workspace_references_and_global_retrieval(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    router = FakeRouter()
    service = _service(router)
    service._workspace_service = WorkspaceService(database_path=tmp_path / "workspaces.sqlite")
    workspace = service.workspaces.store.create_workspace("CSC 220")
    service.workspaces.store.create_item(
        workspace.id,
        WorkspaceItemInput(
            title="Relevant reference",
            role=WorkspaceItemRole.REFERENCE,
            pinned=True,
            content="Predicate logic extends propositional logic.",
        ),
    )
    explicit = SourceMaterial(
        source_id="explicit",
        title="Instructor source",
        text="Explicit instructor material. " * 2000,
    )
    monkeypatch.setattr(
        "lecture_slm.generation.service.load_knowledge_config",
        lambda _: KnowledgeConfig(),
    )

    def fail_global_retrieval(*args: Any, **kwargs: Any) -> RetrievalResult:
        del args, kwargs
        pytest.fail("global retrieval exceeded the source budget")

    monkeypatch.setattr(service, "_retrieve", fail_global_retrieval)

    execution = service.generate(
        GenerationRequest(
            task=TaskType.EXPLANATION,
            instruction="Explain predicate logic.",
            workspace_id=workspace.id,
            source_material=[explicit],
        ),
        retrieval=RetrievalOptions(enabled=True),
    )

    assert [material.source_id for material in execution.request.source_material] == ["explicit"]
    assert execution.request.metadata["workspace"]["reference_selected_item_ids"] == []
    assert execution.request.metadata["knowledge_retrieval"]["matches"] == []


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
        *,
        query: Any,
    ) -> RetrievalResult:
        seen["request"] = request
        seen["options"] = retrieval_options
        seen["query"] = query
        return RetrievalResult(query=query.canonical)

    monkeypatch.setattr(service, "_retrieve", retrieve)
    monkeypatch.setattr(
        "lecture_slm.generation.service.KnowledgeContextAssembler.assemble",
        lambda self, result, *, source_context_budget: [source],
    )

    request = GenerationRequest(task=TaskType.EXPLANATION, instruction="Explain predicate logic.")
    execution = service.generate(request, retrieval=options, save_run=True)

    assert seen["request"] is request
    assert seen["options"] is options
    assert seen["query"].original == request.instruction
    assert seen["query"].canonical == "predicate logic"
    assert router.request is not None
    assert router.request.source_material == [source]
    retrieval_diagnostics = router.request.metadata["knowledge_retrieval"]
    assert retrieval_diagnostics["query"] == "predicate logic"
    assert retrieval_diagnostics["canonical_query"] == "predicate logic"
    assert retrieval_diagnostics["original_query"] == request.instruction
    assert execution.saved_run_directory is not None
    assert execution.retrieved_count == 0
    assert (execution.saved_run_directory / "request.json").is_file()
    assert (execution.saved_run_directory / "result.json").is_file()
    assert (execution.saved_run_directory / "diagnostics.md").is_file()
    diagnostics = (execution.saved_run_directory / "diagnostics.md").read_text(encoding="utf-8")
    assert "- Original instruction: Explain predicate logic." in diagnostics
    assert "- Canonical query: predicate logic" in diagnostics


def test_no_workspace_keeps_existing_retrieval_when_explicit_sources_fill_budget(
    monkeypatch: Any,
) -> None:
    router = FakeRouter()
    service = _service(router)
    knowledge = KnowledgeConfig()
    knowledge.retrieval.context_budgets["standard"] = 1
    retrieval_calls: list[str] = []
    monkeypatch.setattr(
        "lecture_slm.generation.service.load_knowledge_config",
        lambda _: knowledge,
    )

    def retrieve(
        request: GenerationRequest,
        retrieval_options: RetrievalOptions,
        _: KnowledgeConfig,
        *,
        query: Any,
        top_k: int | None = None,
    ) -> RetrievalResult:
        del request, retrieval_options, top_k
        retrieval_calls.append(query.canonical)
        return RetrievalResult(query=query.canonical)

    monkeypatch.setattr(service, "_retrieve", retrieve)
    service.generate(
        GenerationRequest(
            task=TaskType.EXPLANATION,
            instruction="Explain predicate logic.",
            source_material=[
                SourceMaterial(
                    source_id="explicit",
                    title="Instructor source",
                    text="Explicit source content " * 100,
                )
            ],
        ),
        retrieval=RetrievalOptions(enabled=True),
    )

    assert retrieval_calls == ["predicate logic"]


def _match(chunk_id: str, *, fused_rank: int = 1) -> RetrievalMatch:
    return RetrievalMatch(
        chunk_id=chunk_id,
        source_id=f"source-{chunk_id}",
        source_title=f"Title {chunk_id}",
        source_path=f"{chunk_id}.md",
        text=f"Evidence for {chunk_id}.",
        fused_rank=fused_rank,
        fused_score=1 / fused_rank,
    )


def test_partial_retrieval_expands_once_and_deduplicates_chunk_matches(
    monkeypatch: Any,
) -> None:
    router = FakeRouter(
        expand_scope=SourceScopeAssessment(
            status=SourceScopeStatus.PARTIAL,
            supported_topics=["quantum annealing"],
            unsupported_requested_topics=["algorithms", "error correction"],
        )
    )
    service = _service(router)
    knowledge = KnowledgeConfig()
    options = RetrievalOptions(enabled=True, top_k=8)
    initial = RetrievalResult(query="quantum computing", matches=[_match("A"), _match("B")])
    expansion_results = {
        "quantum computing algorithms": RetrievalResult(
            query="quantum computing algorithms",
            matches=[_match("B"), _match("C")],
        ),
        "quantum computing error correction": RetrievalResult(
            query="quantum computing error correction",
            matches=[_match("C"), _match("D")],
        ),
    }
    retrieval_calls: list[tuple[str, int | None]] = []

    monkeypatch.setattr(
        "lecture_slm.generation.service.load_knowledge_config",
        lambda _: knowledge,
    )

    def retrieve(
        request: GenerationRequest,
        retrieval_options: RetrievalOptions,
        _: KnowledgeConfig,
        *,
        query: Any,
        top_k: int | None = None,
    ) -> RetrievalResult:
        retrieval_calls.append((query.canonical, top_k))
        return initial if len(retrieval_calls) == 1 else expansion_results[query.canonical]

    def assemble(
        self: Any,
        result: RetrievalResult,
        *,
        source_context_budget: int,
    ) -> list[SourceMaterial]:
        return [
            SourceMaterial(
                source_id=match.source_id,
                title=match.source_title,
                text=match.text,
                metadata={
                    "source_origin": "retrieved_knowledge",
                    "chunk_ids": [match.chunk_id],
                },
            )
            for match in result.matches
        ]

    monkeypatch.setattr(service, "_retrieve", retrieve)
    monkeypatch.setattr(
        "lecture_slm.generation.service.KnowledgeContextAssembler.assemble",
        assemble,
    )

    request = GenerationRequest(
        task=TaskType.EXPLANATION,
        instruction="Can you explain quantum computing?",
        previous_topics=["algorithms", "error correction"],
    )
    execution = service.generate(request, retrieval=options)

    assert router.expansion_callback is not None
    assert router.request is not None
    expanded_request = execution.request
    assert [query for query, _ in retrieval_calls] == [
        "quantum computing",
        "quantum computing algorithms",
        "quantum computing error correction",
    ]
    assert [top_k for _, top_k in retrieval_calls] == [None, 2, 2]
    assert [source.metadata["chunk_ids"][0] for source in expanded_request.source_material] == [
        "A",
        "B",
        "C",
        "D",
    ]
    diagnostics = expanded_request.metadata["knowledge_retrieval"]["diagnostics"]
    assert len(diagnostics["retrieval_rounds"]) == 2
    assert diagnostics["retrieval_rounds"][1]["retrieved_count"] == 4
    assert diagnostics["retrieval_rounds"][1]["unique_added_count"] == 2
    assert diagnostics["retrieval_rounds"][1]["merged_match_count"] == 4
    assert diagnostics["context_assembly"]["candidate_chunks"] == 4
    assert diagnostics["context_assembly"]["source_material_count"] == 4
    assert execution.result.final_output == "Generated output."


def test_explicit_sources_that_leave_no_room_for_retrieval_do_not_expand(
    monkeypatch: Any,
) -> None:
    router = FakeRouter(
        expand_scope=SourceScopeAssessment(
            status=SourceScopeStatus.PARTIAL,
            supported_topics=["manual source topic"],
            unsupported_requested_topics=["another topic"],
        )
    )
    service = _service(router)
    knowledge = KnowledgeConfig()
    monkeypatch.setattr(
        "lecture_slm.generation.service.load_knowledge_config",
        lambda _: knowledge,
    )
    monkeypatch.setattr(
        service,
        "_retrieve",
        lambda *args, **kwargs: RetrievalResult(
            query=kwargs["query"].canonical,
            matches=[_match("retrieved")],
        ),
    )
    monkeypatch.setattr(
        "lecture_slm.generation.service.estimate_tokens_from_characters",
        lambda _: 10_000,
    )
    request = GenerationRequest(
        task=TaskType.EXPLANATION,
        instruction="Explain the topic.",
        source_material=[
            SourceMaterial(
                source_id="manual",
                title="Manual source",
                text="This caller-supplied evidence is authoritative.",
            )
        ],
    )

    execution = service.generate(request, retrieval=RetrievalOptions(enabled=True))

    assert router.expansion_callback is None
    assert execution.request.source_material == request.source_material
