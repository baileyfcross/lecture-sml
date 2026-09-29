"""Quick, Standard, and Deep generation orchestration."""

import hashlib
import time
from collections.abc import Callable

from lecture_slm.config.loader import ModelConfig
from lecture_slm.generation.context import select_context_tier
from lecture_slm.generation.models import (
    GenerationRequest,
    GenerationResult,
    GenerationStage,
    GenerationStatus,
    ProgressEvent,
    ReviewFeedback,
    StageRecord,
    StageTiming,
)
from lecture_slm.generation.planner import Planner
from lecture_slm.generation.profiles import GenerationProfile, GenerationProfiles
from lecture_slm.generation.prompts.planner import build_planner_prompt
from lecture_slm.generation.prompts.writer import (
    WRITER_PROMPT_VERSION,
    build_writer_prompt,
)
from lecture_slm.generation.reviewer import GenerationReviewer
from lecture_slm.generation.writer import Writer
from lecture_slm.inference.ollama_client import OllamaClient, OllamaError

ProgressCallback = Callable[[ProgressEvent], None]
ClientFactory = Callable[[str, float], OllamaClient]


class GenerationPipeline:
    """Route one request through Quick writer-only or Planner -> Writer stages."""

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        profiles: GenerationProfiles,
        client_factory: ClientFactory | None = None,
        reviewer: GenerationReviewer | None = None,
    ) -> None:
        if model_config.inference.provider != "ollama":
            raise ValueError("Generation currently supports only the Ollama provider")
        self.model_config = model_config
        self.profiles = profiles
        self.client_factory = client_factory or (
            lambda host, timeout: OllamaClient(host, timeout=timeout)
        )
        self.reviewer = reviewer

    def generate(
        self,
        request: GenerationRequest,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        started = time.perf_counter()
        profile = self.profiles.profiles[request.profile]
        errors: list[str] = []
        planner_record: StageRecord | None = None
        writer_record: StageRecord | None = None
        reviewer_feedback = None
        reviewer_error: str | None = None
        plan = None
        selected_contexts: dict[str, int] = {}
        estimated_inputs: dict[str, int] = {}
        progress = self._progress_emitter(started, on_progress)

        progress(
            GenerationStage.PREPARING,
            f"Preparing {request.profile.value} {request.task.value} request",
        )
        try:
            self.client_factory(self.model_config.inference.host, 30.0).ensure_model_available(
                self.model_config.ollama_name
            )
        except OllamaError as error:
            message = str(error).replace(
                self.model_config.inference.host, "[configured Ollama server]"
            )
            errors.append(message)
            progress(GenerationStage.FAILED, f"Model preflight failed: {type(error).__name__}")
            return self._result(
                request,
                profile,
                GenerationStatus.FAILED,
                started,
                errors=errors,
                selected_contexts=selected_contexts,
                estimated_inputs=estimated_inputs,
                planner_record=None,
                writer_record=None,
                reviewer_feedback=None,
                reviewer_error=None,
                final_output=None,
            )

        if profile.planner.enabled:
            planner_prompt = build_planner_prompt(request)
            planner_input = f"{planner_prompt.system_message}\n\n{planner_prompt.user_message}"
            try:
                planner_selection = select_context_tier(
                    planner_input,
                    profile.planner.context_tiers,
                    safety_margin=self.profiles.context_safety_margin,
                    output_reserve_tokens=profile.planner.output_budget(request.task),
                )
                selected_contexts["planner"] = planner_selection.selected_context
                estimated_inputs["planner"] = planner_selection.estimated_input_tokens
            except ValueError as error:
                planner_record = StageRecord(
                    status=GenerationStatus.FAILED,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            else:
                planning_message = (
                    "Deep planning: creating a structured teaching plan"
                    if request.profile.value == "deep"
                    else "Planning: creating a concise structured teaching plan"
                )
                self._emit_stage_start(
                    progress,
                    GenerationStage.PLANNING,
                    planning_message,
                    profile.planner.max_output_tokens,
                )
                planner = Planner(
                    client=self.client_factory(
                        self.model_config.inference.host,
                        profile.planner.timeout_seconds,
                    ),
                    model=self.model_config.ollama_name,
                    model_config=self.model_config,
                    profiles=self.profiles,
                )
                planner_record = planner.plan(request, profile.planner)
                plan = planner_record.plan
                selected = (
                    None
                    if planner_record.timing is None
                    else planner_record.timing.selected_context
                )
                if selected is not None:
                    selected_contexts["planner"] = selected
                if (
                    planner_record.timing
                    and planner_record.timing.estimated_input_tokens is not None
                ):
                    estimated_inputs["planner"] = planner_record.timing.estimated_input_tokens
                self._emit_stage_complete(progress, GenerationStage.PLANNING, planner_record)

            if planner_record.status is GenerationStatus.FAILED or plan is None:
                errors.append(
                    f"Planner failed: {planner_record.error_type}: {planner_record.error_message}"
                )
                progress(GenerationStage.FAILED, "Planner failed; writer was not started")
                return self._result(
                    request,
                    profile,
                    GenerationStatus.FAILED,
                    started,
                    errors=errors,
                    selected_contexts=selected_contexts,
                    estimated_inputs=estimated_inputs,
                    planner_record=planner_record,
                    writer_record=None,
                    reviewer_feedback=None,
                    reviewer_error=None,
                    final_output=None,
                )

        writer_prompt = build_writer_prompt(request, plan)
        writer_input = f"{writer_prompt.system_message}\n\n{writer_prompt.user_message}"
        try:
            writer_selection = select_context_tier(
                writer_input,
                profile.writer.context_tiers,
                safety_margin=self.profiles.context_safety_margin,
                output_reserve_tokens=profile.writer.output_budget(request.task),
            )
            selected_contexts["writer"] = writer_selection.selected_context
            estimated_inputs["writer"] = writer_selection.estimated_input_tokens
        except ValueError as error:
            writer_record = StageRecord(
                status=GenerationStatus.FAILED,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        else:
            writer_budget = profile.writer.output_budget(request.task)
            self._emit_stage_start(
                progress,
                GenerationStage.WRITING,
                "Writer is producing the requested artifact",
                writer_budget,
            )
            writer = Writer(
                client=self.client_factory(
                    self.model_config.inference.host,
                    profile.writer.timeout_seconds,
                ),
                model=self.model_config.ollama_name,
                model_config=self.model_config,
                profiles=self.profiles,
            )
            writer_record = writer.write(request, profile.writer, plan=plan)
            selected = (
                None if writer_record.timing is None else writer_record.timing.selected_context
            )
            if selected is not None:
                selected_contexts["writer"] = selected
            if writer_record.timing and writer_record.timing.estimated_input_tokens is not None:
                estimated_inputs["writer"] = writer_record.timing.estimated_input_tokens
            self._emit_stage_complete(progress, GenerationStage.WRITING, writer_record)

        final_output = (
            writer_record.raw_response
            if writer_record.status is GenerationStatus.COMPLETED
            else None
        )
        status = writer_record.status
        if status is GenerationStatus.FAILED:
            errors.append(
                f"Writer failed: {writer_record.error_type}: {writer_record.error_message}"
            )
            progress(GenerationStage.FAILED, "Writer failed; any completed plan was preserved")

        if status is GenerationStatus.COMPLETED and (
            request.enable_review or profile.review_enabled
        ):
            if self.reviewer is None:
                reviewer_error = (
                    "Reviewer was requested but no reviewer implementation is configured"
                )
                errors.append(reviewer_error)
                status = GenerationStatus.FAILED
                progress(GenerationStage.FAILED, reviewer_error)
            else:
                progress(GenerationStage.REVIEWING, "Reviewer is checking the final artifact")
                reviewer_feedback = self.reviewer.review(request, plan, final_output or "")

        if status is GenerationStatus.COMPLETED:
            progress(GenerationStage.COMPLETE, "Generation complete")
        return self._result(
            request,
            profile,
            status,
            started,
            errors=errors,
            selected_contexts=selected_contexts,
            estimated_inputs=estimated_inputs,
            planner_record=planner_record,
            writer_record=writer_record,
            reviewer_feedback=reviewer_feedback,
            reviewer_error=reviewer_error,
            final_output=final_output,
        )

    def _progress_emitter(
        self,
        started: float,
        callback: ProgressCallback | None,
    ) -> Callable[..., None]:
        def emit(
            stage: GenerationStage,
            message: str,
            *,
            generated_tokens: int | None = None,
            tokens_per_second: float | None = None,
            estimated_seconds_remaining: float | None = None,
        ) -> None:
            if callback is not None:
                callback(
                    ProgressEvent(
                        stage=stage,
                        message=message,
                        elapsed_seconds=time.perf_counter() - started,
                        generated_tokens=generated_tokens,
                        tokens_per_second=tokens_per_second,
                        estimate_seconds_remaining=estimated_seconds_remaining,
                        estimate_is_approximate=True,
                    )
                )

        return emit

    def _emit_stage_start(
        self,
        progress: Callable[..., None],
        stage: GenerationStage,
        message: str,
        output_budget: int,
    ) -> None:
        progress(
            stage,
            message,
            tokens_per_second=self.profiles.estimated_fallback_tokens_per_second,
            estimated_seconds_remaining=(
                output_budget / self.profiles.estimated_fallback_tokens_per_second
            ),
        )

    @staticmethod
    def _emit_stage_complete(
        progress: Callable[..., None],
        stage: GenerationStage,
        record: StageRecord,
    ) -> None:
        timing = record.timing
        progress(
            stage,
            "Stage complete" if record.status is GenerationStatus.COMPLETED else "Stage failed",
            generated_tokens=None if timing is None else timing.generated_tokens,
            tokens_per_second=None if timing is None else timing.tokens_per_second,
        )

    def _result(
        self,
        request: GenerationRequest,
        profile: GenerationProfile,
        status: GenerationStatus,
        started: float,
        *,
        errors: list[str],
        selected_contexts: dict[str, int],
        estimated_inputs: dict[str, int],
        planner_record: StageRecord | None,
        writer_record: StageRecord | None,
        reviewer_feedback: ReviewFeedback | None,
        reviewer_error: str | None,
        final_output: str | None,
    ) -> GenerationResult:
        planner_seconds = (
            0.0
            if planner_record is None or planner_record.timing is None
            else planner_record.timing.duration_seconds
        )
        writer_seconds = (
            0.0
            if writer_record is None or writer_record.timing is None
            else writer_record.timing.duration_seconds
        )
        model_defaults = self.model_config.model_dump(mode="json")
        model_defaults["inference"]["host"] = "[redacted]"
        return GenerationResult(
            request_id=request.request_id,
            task=request.task,
            profile=request.profile,
            model=self.model_config.ollama_name,
            status=status,
            planner_result=planner_record,
            writer_result=writer_record,
            reviewer_result=reviewer_feedback,
            reviewer_error=reviewer_error,
            final_output=final_output,
            model_defaults=model_defaults,
            profile_configuration=profile.model_dump(mode="json"),
            selected_contexts=selected_contexts,
            estimated_input_tokens=estimated_inputs,
            timing=StageTiming(duration_seconds=time.perf_counter() - started),
            errors=errors,
            metadata={
                "generation_profiles_id": self.profiles.id,
                "generation_profiles_version": self.profiles.version,
                "planner_prompt_version": (
                    None if planner_record is None else planner_record.prompt_version
                ),
                "writer_prompt_version": WRITER_PROMPT_VERSION,
                "planner_duration_seconds": planner_seconds,
                "writer_duration_seconds": writer_seconds,
                "profile_sha256": hashlib.sha256(
                    self.profiles.model_dump_json().encode("utf-8")
                ).hexdigest(),
            },
            planner_prompt_version=(
                None if planner_record is None else planner_record.prompt_version
            ),
            writer_prompt_version=WRITER_PROMPT_VERSION,
        )
