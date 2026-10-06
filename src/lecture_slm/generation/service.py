"""Shared application orchestration for CLI and local API generation."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from lecture_slm.config.loader import ModelConfig
from lecture_slm.generation.context import estimate_tokens_from_characters
from lecture_slm.generation.models import (
    GenerationRequest,
    GenerationResult,
    ProgressEvent,
)
from lecture_slm.generation.persistence import create_generation_run_directory, save_generation_run
from lecture_slm.generation.profiles import GenerationProfiles
from lecture_slm.generation.router import GenerationRouter
from lecture_slm.knowledge.assembler import KnowledgeContextAssembler
from lecture_slm.knowledge.chunking import CHUNKING_VERSION
from lecture_slm.knowledge.config import KnowledgeConfig, load_knowledge_config
from lecture_slm.knowledge.embeddings import FastEmbedProvider, fastembed_provider_version
from lecture_slm.knowledge.models import RetrievalRequest, RetrievalResult
from lecture_slm.knowledge.retrieval import KnowledgeRetriever
from lecture_slm.knowledge.storage import KnowledgeStore
from lecture_slm.knowledge.vault import validate_knowledge_paths

ProgressCallback = Callable[[ProgressEvent], None]
RouterFactory = Callable[..., GenerationRouter]


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
        if retrieval is not None and retrieval.enabled:
            knowledge_config = load_knowledge_config(self.knowledge_config_path)
            retrieval_result = self._retrieve(request, retrieval, knowledge_config)
            budget = knowledge_config.retrieval.context_budgets[request.profile.value]
            explicit_tokens = sum(
                estimate_tokens_from_characters(
                    f"{material.title}\n{material.section or ''}\n{material.text}"
                )
                for material in request.source_material
            )
            remaining_budget = budget - explicit_tokens
            source_material = list(request.source_material)
            if remaining_budget > 0:
                source_material.extend(
                    KnowledgeContextAssembler().assemble(
                        retrieval_result,
                        source_context_budget=remaining_budget,
                    )
                )
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

        router = self.router_factory(model_config=self.model_config, profiles=self.profiles)
        result = router.route(active_request, on_progress=on_progress)
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
                    query=request.instruction,
                    source_title=options.source_title,
                    section=options.section,
                    course=options.course,
                    tags=options.tags,
                    folder=options.folder,
                    top_k=options.top_k,
                )
            )
