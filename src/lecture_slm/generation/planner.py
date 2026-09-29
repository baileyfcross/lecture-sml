"""Structured planning stage implemented as its own Ollama request."""

import json
import time

from pydantic import ValidationError

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
from lecture_slm.generation.prompts.planner import build_planner_prompt
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


class Planner:
    """Create and validate a structured teaching plan, not a final artifact."""

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

    def plan(self, request: GenerationRequest, settings: StageProfile) -> StageRecord:
        prompt = build_planner_prompt(request)
        assembled = f"{prompt.system_message}\n\n{prompt.user_message}"
        selection = select_context_tier(
            assembled,
            settings.context_tiers,
            safety_margin=self.profiles.context_safety_margin,
            output_reserve_tokens=settings.output_budget(request.task),
        )
        last_raw: str | None = None
        last_error: Exception | None = None
        last_timing: StageTiming | None = None

        for _attempt in range(settings.retry_count + 1):
            started = time.perf_counter()
            attempt_timing: StageTiming | None = None
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
                        "num_predict": settings.output_budget(request.task),
                    },
                    think=settings.think,
                    keep_alive=self.model_config.inference.keep_alive,
                    format=TeachingPlan.model_json_schema(),
                )
                last_raw = response.content
                attempt_timing = _stage_timing(
                    response,
                    elapsed=time.perf_counter() - started,
                    selected_context=selection.selected_context,
                    estimated_input_tokens=selection.estimated_input_tokens,
                    budget=settings.output_budget(request.task),
                )
                last_timing = attempt_timing
                plan = TeachingPlan.model_validate_json(response.content)
                if plan.task is not request.task:
                    raise ValueError(
                        "Planner returned task "
                        f"'{plan.task.value}', expected '{request.task.value}'"
                    )
                return StageRecord(
                    status=GenerationStatus.COMPLETED,
                    timing=last_timing,
                    raw_response=last_raw,
                    plan=plan,
                )
            except (OllamaError, ValidationError, json.JSONDecodeError, ValueError) as error:
                last_error = error
                if attempt_timing is None:
                    last_timing = StageTiming(
                        duration_seconds=time.perf_counter() - started,
                        selected_context=selection.selected_context,
                        estimated_input_tokens=selection.estimated_input_tokens,
                        output_budget=settings.output_budget(request.task),
                    )

        return StageRecord(
            status=GenerationStatus.FAILED,
            timing=last_timing
            or StageTiming(
                duration_seconds=0.0,
                selected_context=selection.selected_context,
                estimated_input_tokens=selection.estimated_input_tokens,
                output_budget=settings.output_budget(request.task),
            ),
            raw_response=last_raw,
            error_type=type(last_error).__name__ if last_error else "PlannerError",
            error_message=str(last_error) if last_error else "Planner failed without details",
        )
