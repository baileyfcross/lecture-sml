"""Public request router entry point for generation workflows."""

from collections.abc import Callable

from lecture_slm.config.loader import ModelConfig
from lecture_slm.generation.models import GenerationRequest, GenerationResult
from lecture_slm.generation.pipeline import GenerationPipeline, ProgressCallback
from lecture_slm.generation.profiles import GenerationProfiles
from lecture_slm.generation.reviewer import GenerationReviewer
from lecture_slm.inference.ollama_client import OllamaClient


class GenerationRouter:
    """Route a typed request to Quick, Standard, or Deep configured stages."""

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        profiles: GenerationProfiles,
        client_factory: Callable[[str, float], OllamaClient] | None = None,
        reviewer: GenerationReviewer | None = None,
    ) -> None:
        self.pipeline = GenerationPipeline(
            model_config=model_config,
            profiles=profiles,
            client_factory=client_factory,
            reviewer=reviewer,
        )

    def route(
        self,
        request: GenerationRequest,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        """Execute the workflow selected by ``request.profile``."""

        return self.pipeline.generate(request, on_progress=on_progress)
