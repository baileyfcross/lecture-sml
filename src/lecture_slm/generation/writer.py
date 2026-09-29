"""Final educational artifact writing stage as a separate Ollama request."""

import time

from lecture_slm.config.loader import ModelConfig
from lecture_slm.generation.context import select_context_tier
from lecture_slm.generation.models import (
    GenerationRequest,
    GenerationStatus,
    StageRecord,
    StageTiming,
    TeachingPlan,
)
from lecture_slm.generation.profiles import GenerationProfiles, StageProfile
from lecture_slm.generation.prompts.base import PromptPackage, request_blocks
from lecture_slm.generation.prompts.writer import build_writer_prompt
from lecture_slm.inference.ollama_client import ChatResponse, OllamaClient, OllamaError


def _stage_timing(
    response: ChatResponse,
    *,
    elapsed: float,
    selected_context: int,
    estimated_input_tokens: int,
    budget: int,
) -> StageTiming:
    rate = None
    if response.completion_tokens is not None and response.eval_duration_ns:
        rate = response.completion_tokens * 1_000_000_000 / response.eval_duration_ns
    duration = (
        response.total_duration_ns / 1_000_000_000
        if response.total_duration_ns is not None
        else elapsed
    )
    return StageTiming(
        duration_seconds=duration,
        prompt_tokens=response.prompt_tokens,
        generated_tokens=response.completion_tokens,
        tokens_per_second=rate,
        stop_reason=response.completion_reason,
        selected_context=selected_context,
        estimated_input_tokens=estimated_input_tokens,
        output_budget=budget,
    )


class Writer:
    """Generate the final artifact from the authoritative request and TeachingPlan."""

    def __init__(
        self,
        *,
        client: OllamaClient,
        model: str,
        model_config: ModelConfig,
        profiles: GenerationProfiles,
    ) -> None:
        self.client = client
        self.model = model
        self.model_config = model_config
        self.profiles = profiles

    def write(
        self,
        request: GenerationRequest,
        settings: StageProfile,
        *,
        plan: TeachingPlan | None,
    ) -> StageRecord:
        if settings.think:
            return StageRecord(
                status=GenerationStatus.FAILED,
                error_type="InvalidWriterConfiguration",
                error_message="Writer requests must have thinking disabled",
            )
        prompt = (
            build_writer_prompt(request, plan) if plan is not None else self._quick_prompt(request)
        )
        assembled = f"{prompt.system_message}\n\n{prompt.user_message}"
        selection = select_context_tier(
            assembled,
            settings.context_tiers,
            safety_margin=self.profiles.context_safety_margin,
            output_reserve_tokens=settings.output_budget(request.task),
        )
        output_budget = settings.output_budget(request.task)
        started = time.perf_counter()
        try:
            response = self.client.chat(
                model=self.model,
                system_message=prompt.system_message,
                user_message=prompt.user_message,
                options={
                    "temperature": self.model_config.inference.temperature,
                    "top_p": self.model_config.inference.top_p,
                    "seed": self.model_config.inference.seed,
                    "num_ctx": selection.selected_context,
                    "num_predict": output_budget,
                },
                think=False,
                keep_alive=self.model_config.inference.keep_alive,
            )
        except OllamaError as error:
            return StageRecord(
                status=GenerationStatus.FAILED,
                timing=StageTiming(
                    duration_seconds=time.perf_counter() - started,
                    selected_context=selection.selected_context,
                    estimated_input_tokens=selection.estimated_input_tokens,
                    output_budget=output_budget,
                ),
                error_type=type(error).__name__,
                error_message=str(error).replace(
                    self.model_config.inference.host,
                    "[configured Ollama server]",
                ),
            )
        return StageRecord(
            status=GenerationStatus.COMPLETED,
            timing=_stage_timing(
                response,
                elapsed=time.perf_counter() - started,
                selected_context=selection.selected_context,
                estimated_input_tokens=selection.estimated_input_tokens,
                budget=output_budget,
            ),
            raw_response=response.content,
        )

    @staticmethod
    def _quick_prompt(request: GenerationRequest) -> PromptPackage:
        blocks = request_blocks(request)
        blocks.append(
            "## Writer task\nCreate the requested artifact directly. "
            "Do not provide a planning discussion."
        )
        return PromptPackage(
            version="writer-v1",
            system_message=(
                "You are the artifact writer for Lecture SLM. Follow the user's request, "
                "course and pedagogy constraints, and supplied context. "
                "Produce the artifact directly."
            ),
            user_message="\n\n".join(blocks),
        )
