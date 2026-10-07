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
)
from lecture_slm.generation.plan_schemas import plan_schema_for_task, validate_plan_for_task
from lecture_slm.generation.profiles import GenerationProfiles, StageProfile
from lecture_slm.generation.prompts.planner import build_planner_prompt
from lecture_slm.generation.timing import detect_output_limit
from lecture_slm.inference.ollama_client import ChatResponse, OllamaClient, OllamaError


class PlannerThinkingBudgetExhaustedError(RuntimeError):
    """Thinking consumed the full generation allowance before a plan was emitted."""


class PlannerOutputTruncatedError(RuntimeError):
    """The generation allowance ended before the structured plan could be validated."""


def _stage_timing(
    response: ChatResponse,
    *,
    elapsed: float,
    selected_context: int,
    estimated_input_tokens: int,
    structured_output_budget: int,
    thinking_reserve_tokens: int,
    generation_budget: int,
    thinking_enabled: bool,
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
        output_limit_reached=detect_output_limit(
            stop_reason=response.completion_reason,
            generated_tokens=response.completion_tokens,
            output_budget=generation_budget,
        ),
        potentially_truncated=detect_output_limit(
            stop_reason=response.completion_reason,
            generated_tokens=response.completion_tokens,
            output_budget=generation_budget,
        ),
        thinking_enabled=thinking_enabled,
        thinking_characters=(
            len(response.thinking_content)
            if thinking_enabled and response.thinking_content is not None
            else None
        ),
        selected_context=selected_context,
        estimated_input_tokens=estimated_input_tokens,
        output_budget=structured_output_budget,
        thinking_reserve_tokens=thinking_reserve_tokens,
        generation_budget=generation_budget,
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
        prompt = build_planner_prompt(request, concise=not settings.think)
        assembled = f"{prompt.system_message}\n\n{prompt.user_message}"
        structured_output_budget = settings.output_budget(request.task)
        thinking_reserve_tokens = settings.thinking_reserve_tokens if settings.think else 0
        generation_budget = settings.generation_budget(request.task)
        selection = select_context_tier(
            assembled,
            settings.context_tiers,
            safety_margin=self.profiles.context_safety_margin,
            output_reserve_tokens=generation_budget,
        )
        last_raw: str | None = None
        last_error: Exception | None = None
        last_timing: StageTiming | None = None
        last_response: ChatResponse | None = None
        plan_schema = plan_schema_for_task(request.task)

        for _attempt in range(settings.retry_count + 1):
            started = time.perf_counter()
            attempt_timing: StageTiming | None = None
            try:
                response = self.client.chat(
                    model=self.model,
                    system_message=prompt.system_message,
                    user_message=prompt.user_message,
                    options={
                        "temperature": (
                            self.model_config.inference.temperature
                            if settings.temperature is None
                            else settings.temperature
                        ),
                        "top_p": self.model_config.inference.top_p,
                        "seed": self.model_config.inference.seed,
                        "num_ctx": selection.selected_context,
                        "num_predict": generation_budget,
                    },
                    think=settings.think,
                    keep_alive=self.model_config.inference.keep_alive,
                    format=plan_schema.model_json_schema(),
                )
                last_response = response
                last_raw = response.content
                attempt_timing = _stage_timing(
                    response,
                    elapsed=time.perf_counter() - started,
                    selected_context=selection.selected_context,
                    estimated_input_tokens=selection.estimated_input_tokens,
                    structured_output_budget=structured_output_budget,
                    thinking_reserve_tokens=thinking_reserve_tokens,
                    generation_budget=generation_budget,
                    thinking_enabled=settings.think,
                )
                last_timing = attempt_timing
                plan = validate_plan_for_task(
                    request.task,
                    response.content,
                    require_source_scope=bool(request.source_material),
                )
                return StageRecord(
                    status=GenerationStatus.COMPLETED,
                    timing=last_timing,
                    raw_response=last_raw,
                    plan=plan,
                    prompt_version=prompt.version,
                )
            except (OllamaError, ValidationError, json.JSONDecodeError, ValueError) as error:
                last_error = error
                if (
                    last_response is not None
                    and last_response.completion_reason == "length"
                    and last_response.completion_tokens is not None
                    and last_response.completion_tokens >= generation_budget
                ):
                    budget_details = (
                        f"Structured output budget: {structured_output_budget}; "
                        f"thinking reserve: {thinking_reserve_tokens}; "
                        f"total generation budget: {generation_budget}; "
                        f"generated tokens: {last_response.completion_tokens}; "
                        f"thinking characters: "
                        f"{len(last_response.thinking_content or '') if settings.think else 0}; "
                        f"stop reason: {last_response.completion_reason}."
                    )
                    if (
                        settings.think
                        and not last_response.content.strip()
                        and (last_response.thinking_content or "").strip()
                    ):
                        last_error = PlannerThinkingBudgetExhaustedError(
                            "Deep Planner exhausted its generation budget during reasoning "
                            f"before producing the structured plan. {budget_details}"
                        )
                    elif last_response.content.strip():
                        last_error = PlannerOutputTruncatedError(
                            "Planner output was truncated after reaching its generation "
                            "budget; the incomplete structured plan could not be validated. "
                            f"{budget_details} Validation error: {error}"
                        )
                if attempt_timing is None:
                    last_timing = StageTiming(
                        duration_seconds=time.perf_counter() - started,
                        selected_context=selection.selected_context,
                        estimated_input_tokens=selection.estimated_input_tokens,
                        output_budget=structured_output_budget,
                        thinking_reserve_tokens=thinking_reserve_tokens,
                        generation_budget=generation_budget,
                    )

        return StageRecord(
            status=GenerationStatus.FAILED,
            timing=last_timing
            or StageTiming(
                duration_seconds=0.0,
                selected_context=selection.selected_context,
                estimated_input_tokens=selection.estimated_input_tokens,
                output_budget=structured_output_budget,
                thinking_reserve_tokens=thinking_reserve_tokens,
                generation_budget=generation_budget,
            ),
            raw_response=last_raw,
            prompt_version=prompt.version,
            error_type=type(last_error).__name__ if last_error else "PlannerError",
            error_message=str(last_error) if last_error else "Planner failed without details",
        )
