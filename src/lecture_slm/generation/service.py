"""Shared application orchestration for CLI and local API generation."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from lecture_slm.config.loader import ModelConfig
from lecture_slm.generation.context import estimate_tokens_from_characters
from lecture_slm.generation.models import (
    GenerationRequest,
    GenerationResult,
    ProgressEvent,
    SourceMaterial,
    SourceScopeAssessment,
)
from lecture_slm.generation.persistence import create_generation_run_directory, save_generation_run
from lecture_slm.generation.pipeline import RetrievalExpansionCallback
from lecture_slm.generation.profiles import GenerationProfiles
from lecture_slm.generation.router import GenerationRouter
from lecture_slm.knowledge.assembler import KnowledgeContextAssembler
from lecture_slm.knowledge.chunking import CHUNKING_VERSION
from lecture_slm.knowledge.config import KnowledgeConfig, load_knowledge_config
from lecture_slm.knowledge.embeddings import FastEmbedProvider, fastembed_provider_version
from lecture_slm.knowledge.models import (
    RetrievalMatch,
    RetrievalQuery,
    RetrievalRequest,
    RetrievalResult,
    RetrievalTimings,
)
from lecture_slm.knowledge.query import (
    build_retrieval_expansion_plan,
    canonicalize_retrieval_query,
)
from lecture_slm.knowledge.retrieval import KnowledgeRetriever
from lecture_slm.knowledge.storage import KnowledgeStore
from lecture_slm.knowledge.vault import validate_knowledge_paths

ProgressCallback = Callable[[ProgressEvent], None]
RouterFactory = Callable[..., GenerationRouter]
EXPANSION_RESULTS_PER_QUERY = 2


def _source_material_tokens(materials: list[SourceMaterial]) -> int:
    return sum(
        estimate_tokens_from_characters(
            f"{material.title}\n{material.section or ''}\n{material.text}"
        )
        for material in materials
    )


def _match_identity(match: RetrievalMatch) -> tuple[str, str]:
    return match.source_id, match.chunk_id


def _match_diagnostic(match: RetrievalMatch) -> dict[str, object]:
    return {
        "source_id": match.source_id,
        "chunk_id": match.chunk_id,
        "source_title": match.source_title,
        "source_path": match.source_path,
        "section": match.section,
        "fused_rank": match.fused_rank,
        "fused_score": match.fused_score,
        "neighbor_of": match.neighbor_of,
    }


def _diagnostic_int(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _diagnostic_strings(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _combined_assembly_diagnostics(
    initial: dict[str, object],
    expansion: dict[str, object],
    *,
    match_count: int,
    source_material_count: int,
) -> dict[str, object]:
    initial_tokens = _diagnostic_int(initial.get("estimated_tokens", 0))
    expansion_tokens = _diagnostic_int(expansion.get("estimated_tokens", 0))
    configured_budget = _diagnostic_int(initial.get("configured_budget", 0))
    excluded_by_role: dict[str, int] = {}
    for assembly in (initial, expansion):
        roles = assembly.get("excluded_by_role", {})
        if isinstance(roles, dict):
            for role, count in roles.items():
                excluded_by_role[str(role)] = excluded_by_role.get(str(role), 0) + (
                    _diagnostic_int(count)
                )
    selected_chunk_ids = _diagnostic_strings(initial.get("selected_chunk_ids", [])) + (
        _diagnostic_strings(expansion.get("selected_chunk_ids", []))
    )
    excluded_by_budget = _diagnostic_strings(initial.get("excluded_by_budget", [])) + (
        _diagnostic_strings(expansion.get("excluded_by_budget", []))
    )
    estimated_tokens = initial_tokens + expansion_tokens
    return {
        "candidate_chunks": match_count,
        "eligible_chunks": _diagnostic_int(initial.get("eligible_chunks", 0))
        + _diagnostic_int(expansion.get("eligible_chunks", 0)),
        "excluded_by_role": excluded_by_role,
        "selected_chunk_ids": selected_chunk_ids,
        "selected_chunks": len(selected_chunk_ids),
        "source_material_count": source_material_count,
        "estimated_tokens": estimated_tokens,
        "configured_budget": configured_budget,
        "unused_budget": max(0, configured_budget - estimated_tokens),
        "excluded_by_budget": excluded_by_budget,
    }


def _merge_retrieval_results(
    initial: RetrievalResult,
    expansions: list[tuple[str, RetrievalResult]],
) -> tuple[RetrievalResult, dict[str, object]]:
    """Preserve initial ranking order and append only new source/chunk identities."""

    matches = list(initial.matches)
    seen = {_match_identity(match) for match in matches}
    found_by_queries: dict[str, list[str]] = {}
    retrieved_count = 0
    unique_added_count = 0
    for query, result in expansions:
        retrieved_count += len(result.matches)
        for match in result.matches:
            identity = _match_identity(match)
            identity_key = f"{identity[0]}:{identity[1]}"
            found_by_queries.setdefault(identity_key, []).append(query)
            if identity in seen:
                continue
            seen.add(identity)
            matches.append(match)
            unique_added_count += 1

    all_results = [initial, *(result for _, result in expansions)]
    timing_values = {
        name: sum(getattr(result.timings, name) for result in all_results)
        for name in RetrievalTimings.model_fields
    }
    warnings = list(dict.fromkeys(warning for result in all_results for warning in result.warnings))
    merged = initial.model_copy(
        update={
            "matches": matches,
            "warnings": warnings,
            "timings": initial.timings.model_copy(update=timing_values),
        },
        deep=True,
    )
    merged.diagnostics["found_by_queries"] = found_by_queries
    return merged, {
        "retrieved_count": retrieved_count,
        "unique_added_count": unique_added_count,
        "merged_match_count": len(matches),
        "found_by_queries": found_by_queries,
    }


@dataclass(frozen=True)
class RetrievalOptions:
    enabled: bool = False
    top_k: int | None = None
    source_title: str | None = None
    section: str | None = None
    course: str | None = None
    tags: list[str] = field(default_factory=list)
    folder: str | None = None


@dataclass(frozen=True)
class GenerationExecution:
    request: GenerationRequest
    result: GenerationResult
    saved_run_directory: Path | None
    retrieved_count: int = 0


class GenerationService:
    """Prepare sources, execute the configured pipeline, and optionally persist its result."""

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        profiles: GenerationProfiles,
        knowledge_config_path: Path = Path("configs/knowledge/default.yaml"),
        artifact_root: Path = Path("artifacts/generations"),
        router_factory: RouterFactory = GenerationRouter,
    ) -> None:
        self.model_config = model_config
        self.profiles = profiles
        self.knowledge_config_path = knowledge_config_path
        self.artifact_root = artifact_root
        self.router_factory = router_factory

    def generate(
        self,
        request: GenerationRequest,
        *,
        retrieval: RetrievalOptions | None = None,
        save_run: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationExecution:
        active_request = request
        retrieval_result: RetrievalResult | None = None
        retrieval_expander: RetrievalExpansionCallback | None = None
        expanded_request: GenerationRequest | None = None
        if retrieval is not None and retrieval.enabled:
            knowledge_config = load_knowledge_config(self.knowledge_config_path)
            retrieval_query = canonicalize_retrieval_query(request.instruction)
            retrieval_started = time.perf_counter()
            retrieval_result = self._retrieve(
                request,
                retrieval,
                knowledge_config,
                query=retrieval_query,
            )
            initial_retrieval_seconds = time.perf_counter() - retrieval_started
            retrieval_result = retrieval_result.model_copy(
                update={
                    "original_query": retrieval_query.original,
                    "canonical_query": retrieval_query.canonical,
                }
            )
            retrieval_result.diagnostics["retrieval_rounds"] = [
                {
                    "round": 1,
                    "type": "initial",
                    "query": retrieval_query.canonical,
                    "retrieved_count": len(retrieval_result.matches),
                    "duration_seconds": initial_retrieval_seconds,
                    "results": [_match_diagnostic(match) for match in retrieval_result.matches],
                }
            ]
            budget = knowledge_config.retrieval.context_budgets[request.profile.value]
            explicit_material = list(request.source_material)
            remaining_budget = budget - _source_material_tokens(explicit_material)
            source_material = list(request.source_material)
            retrieved_material: list[SourceMaterial] = []
            if remaining_budget > 0:
                retrieved_material = KnowledgeContextAssembler().assemble(
                    retrieval_result,
                    source_context_budget=remaining_budget,
                )
                source_material.extend(retrieved_material)
            elif retrieval_result.matches:
                retrieval_result.warnings.append(
                    "Explicit source material used the configured source-context budget; "
                    "no retrieved material was added."
                )
            metadata = dict(request.metadata)
            metadata["knowledge_retrieval"] = retrieval_result.model_dump(mode="json")
            active_request = request.model_copy(
                update={"source_material": source_material, "metadata": metadata}
            )

            if retrieved_material and retrieval_result is not None:
                initial_retrieval_result = retrieval_result

                def expand_retrieval(
                    sourced_request: GenerationRequest,
                    source_scope: SourceScopeAssessment,
                ) -> GenerationRequest | None:
                    nonlocal expanded_request, retrieval_result
                    plan = build_retrieval_expansion_plan(
                        retrieval_query.canonical,
                        source_scope.unsupported_requested_topics,
                    )
                    initial_diagnostics = dict(initial_retrieval_result.diagnostics)
                    initial_diagnostics["expansion_plan"] = plan.model_dump(mode="json")
                    if not plan.expansion_queries:
                        initial_retrieval_result.diagnostics.update(initial_diagnostics)
                        initial_retrieval_result.diagnostics["retrieval_expanded"] = False
                        sourced_request.metadata["knowledge_retrieval"] = (
                            initial_retrieval_result.model_dump(mode="json")
                        )
                        expanded_request = sourced_request
                        return None

                    expansion_results: list[tuple[str, RetrievalResult]] = []
                    expansion_started = time.perf_counter()
                    initial_match_identities = {
                        _match_identity(match) for match in initial_retrieval_result.matches
                    }
                    for expansion_query in plan.expansion_queries:
                        result = self._retrieve(
                            request,
                            retrieval,
                            knowledge_config,
                            query=RetrievalQuery(
                                original=retrieval_query.canonical,
                                canonical=expansion_query,
                            ),
                            top_k=EXPANSION_RESULTS_PER_QUERY,
                        )
                        expansion_results.append((expansion_query, result))
                    expansion_seconds = time.perf_counter() - expansion_started

                    retrieval_result, round_diagnostics = _merge_retrieval_results(
                        initial_retrieval_result,
                        expansion_results,
                    )
                    retrieval_result.diagnostics.update(initial_diagnostics)
                    retrieval_result.diagnostics["retrieval_rounds"] = [
                        *initial_diagnostics["retrieval_rounds"],
                        {
                            "round": 2,
                            "type": "expansion",
                            "queries": [
                                {
                                    "query": query,
                                    "retrieved_count": len(result.matches),
                                    "duration_seconds": result.timings.total_seconds,
                                    "results": [
                                        _match_diagnostic(match) for match in result.matches
                                    ],
                                }
                                for query, result in expansion_results
                            ],
                            "retrieved_count": sum(
                                len(result.matches) for _, result in expansion_results
                            ),
                            "duration_seconds": expansion_seconds,
                            **round_diagnostics,
                        },
                    ]
                    retrieval_result.diagnostics["retrieval_expanded"] = True
                    retrieval_result.diagnostics["scope_reassessment"] = True
                    retrieval_result.diagnostics["expansion_retrieval_seconds"] = expansion_seconds

                    explicit_sources = [
                        material
                        for material in sourced_request.source_material
                        if material.metadata.get("source_origin") != "retrieved_knowledge"
                    ]
                    newly_retrieved_matches = [
                        match
                        for match in retrieval_result.matches
                        if _match_identity(match) not in initial_match_identities
                    ]
                    new_matches_result = retrieval_result.model_copy(
                        update={"matches": newly_retrieved_matches},
                        deep=True,
                    )
                    expansion_budget = budget - _source_material_tokens(
                        [*explicit_sources, *retrieved_material]
                    )
                    assembled_material = (
                        KnowledgeContextAssembler().assemble(
                            new_matches_result,
                            source_context_budget=expansion_budget,
                        )
                        if expansion_budget > 0
                        else []
                    )
                    initial_assembly = initial_diagnostics.get("context_assembly", {})
                    expansion_assembly = new_matches_result.diagnostics.get("context_assembly", {})
                    if isinstance(initial_assembly, dict) and isinstance(expansion_assembly, dict):
                        retrieval_result.diagnostics["context_assembly"] = (
                            _combined_assembly_diagnostics(
                                initial_assembly,
                                expansion_assembly,
                                match_count=len(retrieval_result.matches),
                                source_material_count=(
                                    len(retrieved_material) + len(assembled_material)
                                ),
                            )
                        )
                    retrieval_result.diagnostics["retrieval_rounds"][-1][
                        "assembled_source_count"
                    ] = len(assembled_material)
                    updated_metadata = dict(sourced_request.metadata)
                    updated_metadata["knowledge_retrieval"] = retrieval_result.model_dump(
                        mode="json"
                    )
                    updated_request = sourced_request.model_copy(
                        update={
                            "source_material": [
                                *explicit_sources,
                                *retrieved_material,
                                *assembled_material,
                            ],
                            "metadata": updated_metadata,
                        }
                    )
                    expanded_request = updated_request
                    return updated_request

                retrieval_expander = expand_retrieval

        router = self.router_factory(model_config=self.model_config, profiles=self.profiles)
        if retrieval_expander is None:
            result = router.route(active_request, on_progress=on_progress)
        else:
            result = router.route(
                active_request,
                on_progress=on_progress,
                expand_retrieval=retrieval_expander,
            )
            active_request = expanded_request or active_request
        run_directory = None
        if save_run:
            run_directory = create_generation_run_directory(
                self.artifact_root,
                active_request.request_id,
            )
            save_generation_run(run_directory, active_request, result)
        return GenerationExecution(
            active_request,
            result,
            run_directory,
            retrieved_count=0 if retrieval_result is None else len(retrieval_result.matches),
        )

    def knowledge_index_available(self) -> bool:
        knowledge_config = load_knowledge_config(self.knowledge_config_path)
        return (knowledge_config.data_dir / "knowledge.sqlite").is_file()

    def _retrieve(
        self,
        request: GenerationRequest,
        options: RetrievalOptions,
        knowledge: KnowledgeConfig,
        *,
        query: RetrievalQuery,
        top_k: int | None = None,
    ) -> RetrievalResult:
        database = knowledge.data_dir / "knowledge.sqlite"
        if not database.exists():
            raise FileNotFoundError(
                f"No knowledge index at {database}; run scripts/index_knowledge.py first."
            )
        validate_knowledge_paths(
            knowledge.data_dir,
            knowledge.embeddings.cache_dir or knowledge.data_dir / "models",
            KnowledgeStore.indexed_vault_root(knowledge.data_dir),
        )
        with KnowledgeStore(
            knowledge.data_dir,
            embedding_model=knowledge.embeddings.model,
            embedding_version=fastembed_provider_version(),
            chunking_version=CHUNKING_VERSION,
        ):
            pass
        embedding_provider = FastEmbedProvider(
            knowledge.embeddings.model,
            cache_dir=str(knowledge.embeddings.cache_dir or knowledge.data_dir / "models"),
        )
        with KnowledgeStore(
            knowledge.data_dir,
            embedding_model=embedding_provider.model_name,
            embedding_version=embedding_provider.provider_version,
            chunking_version=CHUNKING_VERSION,
        ) as store:
            return KnowledgeRetriever(
                store,
                embedding_provider,
                knowledge.retrieval,
            ).retrieve(
                RetrievalRequest(
                    query=query.canonical,
                    source_title=options.source_title,
                    section=options.section,
                    course=options.course,
                    tags=options.tags,
                    folder=options.folder,
                    top_k=top_k or options.top_k,
                )
            )
