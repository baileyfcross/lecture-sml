"""Quick, Standard, and Deep generation orchestration."""

import hashlib
import math
import time
from collections.abc import Callable
from typing import Literal

from lecture_slm.config.loader import ModelConfig
from lecture_slm.generation.context import select_context_tier
from lecture_slm.generation.models import (
    GenerationRequest,
    GenerationResult,
    GenerationStage,
    GenerationStatus,
    GroundingDecision,
    GroundingReviewRecord,
    ProgressEvent,
    ReviewFeedback,
    SourceScopeAssessment,
    SourceScopeStatus,
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
from lecture_slm.generation.reviewer import (
    GenerationReviewer,
    GroundingStageRunner,
)
from lecture_slm.generation.writer import Writer
from lecture_slm.inference.ollama_client import OllamaClient, OllamaError

ProgressCallback = Callable[[ProgressEvent], None]
RetrievalExpansionCallback = Callable[
    [GenerationRequest, SourceScopeAssessment],
    GenerationRequest | None,
]
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
        expand_retrieval: RetrievalExpansionCallback | None = None,
    ) -> GenerationResult:
        started = time.perf_counter()
        profile = self.profiles.profiles[request.profile]
        errors: list[str] = []
        planner_record: StageRecord | None = None
        writer_record: StageRecord | None = None
        reviewer_feedback = None
        reviewer_error: str | None = None
        initial_grounding_review: GroundingReviewRecord | None = None
        revision_record: StageRecord | None = None
        final_grounding_review: GroundingReviewRecord | None = None
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

        def run_planner(target_request: GenerationRequest) -> StageRecord:
            planner_prompt = build_planner_prompt(target_request)
            planner_input = f"{planner_prompt.system_message}\n\n{planner_prompt.user_message}"
            try:
                planner_selection = select_context_tier(
                    planner_input,
                    profile.planner.context_tiers,
                    safety_margin=self.profiles.context_safety_margin,
                    output_reserve_tokens=profile.planner.generation_budget(target_request.task),
                )
                selected_contexts["planner"] = planner_selection.selected_context
                estimated_inputs["planner"] = planner_selection.estimated_input_tokens
            except ValueError as error:
                return StageRecord(
                    status=GenerationStatus.FAILED,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )

            planning_message = (
                "Deep planning: creating a structured teaching plan"
                if target_request.profile.value == "deep"
                else "Planning: creating a concise structured teaching plan"
            )
            self._emit_stage_start(
                progress,
                GenerationStage.PLANNING,
                planning_message,
                profile.planner.generation_budget(target_request.task),
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
            record = planner.plan(target_request, profile.planner)
            selected = None if record.timing is None else record.timing.selected_context
            if selected is not None:
                selected_contexts["planner"] = selected
            if record.timing and record.timing.estimated_input_tokens is not None:
                estimated_inputs["planner"] = record.timing.estimated_input_tokens
            self._emit_stage_complete(progress, GenerationStage.PLANNING, record)
            return record

        if profile.planner.enabled:
            planner_record = run_planner(request)
            plan = planner_record.plan

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

            initial_scope = plan.source_scope
            if (
                expand_retrieval is not None
                and initial_scope is not None
                and initial_scope.status is SourceScopeStatus.PARTIAL
            ):
                initial_planner_seconds = (
                    0.0 if planner_record.timing is None else planner_record.timing.duration_seconds
                )
                retrieval_diagnostics = request.metadata.get("knowledge_retrieval")
                expanded_request = expand_retrieval(request, initial_scope)
                if expanded_request is not None:
                    request = expanded_request
                    planner_record = run_planner(request)
                    plan = planner_record.plan
                    retrieval_diagnostics = request.metadata.get("knowledge_retrieval")
                if isinstance(retrieval_diagnostics, dict):
                    round_diagnostics = retrieval_diagnostics.setdefault("diagnostics", {})
                    if isinstance(round_diagnostics, dict):
                        round_diagnostics["source_scope_assessments"] = {
                            "initial": initial_scope.model_dump(mode="json"),
                            "final": (
                                None
                                if plan is None or plan.source_scope is None
                                else plan.source_scope.model_dump(mode="json")
                            ),
                        }
                        round_diagnostics["initial_planning_seconds"] = initial_planner_seconds
                        round_diagnostics["final_planning_seconds"] = (
                            initial_planner_seconds
                            if expanded_request is None
                            else (
                                0.0
                                if planner_record.timing is None
                                else planner_record.timing.duration_seconds
                            )
                        )

                if expanded_request is not None and (
                    planner_record.status is GenerationStatus.FAILED or plan is None
                ):
                    errors.append(
                        "Planner failed after retrieval expansion: "
                        f"{planner_record.error_type}: {planner_record.error_message}"
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
            elif (
                expand_retrieval is not None
                and initial_scope is not None
                and isinstance(request.metadata.get("knowledge_retrieval"), dict)
            ):
                retrieval_diagnostics = request.metadata["knowledge_retrieval"]
                round_diagnostics = retrieval_diagnostics.setdefault("diagnostics", {})
                if isinstance(round_diagnostics, dict):
                    round_diagnostics["source_scope_assessments"] = {
                        "initial": initial_scope.model_dump(mode="json"),
                        "final": initial_scope.model_dump(mode="json"),
                    }
                    round_diagnostics["initial_planning_seconds"] = (
                        0.0
                        if planner_record.timing is None
                        else planner_record.timing.duration_seconds
                    )
                    round_diagnostics["final_planning_seconds"] = round_diagnostics[
                        "initial_planning_seconds"
                    ]

        if (
            request.source_material
            and plan is not None
            and plan.source_scope is not None
            and plan.source_scope.status is SourceScopeStatus.INSUFFICIENT
        ):
            message = (
                "Supplied sources are insufficient to support a substantive factual artifact; "
                "the writer was not started."
            )
            errors.append(message)
            progress(GenerationStage.FAILED, message)
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
                previous_record=planner_record,
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

        grounding_review_enabled = bool(request.source_material) and request.profile.value in {
            "standard",
            "deep",
        }
        if grounding_review_enabled:
            final_output = None
        if status is GenerationStatus.COMPLETED and grounding_review_enabled:
            grounding_runner = GroundingStageRunner(
                client=self.client_factory(
                    self.model_config.inference.host,
                    profile.writer.timeout_seconds,
                ),
                model=self.model_config.ollama_name,
                model_config=self.model_config,
                profiles=self.profiles,
                settings=profile.writer,
            )
            initial_review_preparation = grounding_runner.prepare_review(
                request,
                writer_record.raw_response or "",
            )
            self._emit_stage_start(
                progress,
                GenerationStage.REVIEWING,
                "Grounding review: checking claims against supplied sources",
                initial_review_preparation.output_budget or 0,
                previous_record=writer_record,
            )
            initial_grounding_review = grounding_runner.review(
                request,
                writer_record.raw_response or "",
                preparation=initial_review_preparation,
            )
            self._emit_stage_complete(
                progress,
                GenerationStage.REVIEWING,
                self._review_as_stage(initial_grounding_review),
            )
            self._record_stage_selection(
                "grounding_review",
                initial_grounding_review.timing,
                selected_contexts,
                estimated_inputs,
            )

            if initial_grounding_review.status is GenerationStatus.FAILED:
                status = GenerationStatus.FAILED
                errors.append(
                    "Initial grounding review failed: "
                    f"{initial_grounding_review.error_type}: "
                    f"{initial_grounding_review.error_message}"
                )
                progress(
                    GenerationStage.FAILED,
                    "Grounding review failed; the candidate was not approved",
                )
            elif (
                initial_grounding_review.review is not None
                and initial_grounding_review.review.decision is GroundingDecision.PASS
            ):
                final_output = writer_record.raw_response
            else:
                review = initial_grounding_review.review
                if review is None:
                    status = GenerationStatus.FAILED
                    errors.append("Initial grounding review completed without a decision")
                    progress(GenerationStage.FAILED, "Grounding review returned no decision")
                else:
                    self._emit_stage_start(
                        progress,
                        GenerationStage.REVISING,
                        "Grounding revision: applying the bounded source-based corrections",
                        profile.writer.output_budget(request.task),
                        previous_record=self._review_as_stage(initial_grounding_review),
                    )
                    revision_record = grounding_runner.revise(
                        request,
                        writer_record.raw_response or "",
                        review,
                    )
                    self._emit_stage_complete(
                        progress,
                        GenerationStage.REVISING,
                        revision_record,
                    )
                    self._record_stage_selection(
                        "grounding_revision",
                        revision_record.timing,
                        selected_contexts,
                        estimated_inputs,
                    )
                    if revision_record.status is GenerationStatus.FAILED:
                        status = GenerationStatus.FAILED
                        errors.append(
                            "Grounding revision failed: "
                            f"{revision_record.error_type}: {revision_record.error_message}"
                        )
                        progress(
                            GenerationStage.FAILED,
                            "Grounding revision failed; the original candidate was retained",
                        )
                    else:
                        revised_output = revision_record.raw_response or ""
                        final_review_preparation = grounding_runner.prepare_review(
                            request,
                            revised_output,
                        )
                        self._emit_stage_start(
                            progress,
                            GenerationStage.REVIEWING,
                            "Final grounding review: validating the revised artifact",
                            final_review_preparation.output_budget or 0,
                            previous_record=revision_record,
                        )
                        final_grounding_review = grounding_runner.review(
                            request,
                            revised_output,
                            preparation=final_review_preparation,
                        )
                        self._emit_stage_complete(
                            progress,
                            GenerationStage.REVIEWING,
                            self._review_as_stage(final_grounding_review),
                        )
                        self._record_stage_selection(
                            "grounding_final_review",
                            final_grounding_review.timing,
                            selected_contexts,
                            estimated_inputs,
                        )
                        if final_grounding_review.status is GenerationStatus.FAILED:
                            status = GenerationStatus.FAILED
                            errors.append(
                                "Final grounding review failed: "
                                f"{final_grounding_review.error_type}: "
                                f"{final_grounding_review.error_message}"
                            )
                            progress(
                                GenerationStage.FAILED,
                                "Final grounding review failed; revised candidate was not approved",
                            )
                        elif (
                            final_grounding_review.review is not None
                            and final_grounding_review.review.decision is GroundingDecision.PASS
                        ):
                            final_output = revised_output
                        else:
                            status = GenerationStatus.FAILED
                            errors.append(
                                "Final grounding review still requires revision; "
                                "the one-revision limit was reached"
                            )
                            progress(
                                GenerationStage.FAILED,
                                "Final review still requires revision; revision limit reached",
                            )

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
            initial_grounding_review=initial_grounding_review,
            revision_record=revision_record,
            final_grounding_review=final_grounding_review,
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
            estimate_rate_source: Literal["fallback", "observed_previous_stage"] | None = None,
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
                        estimate_rate_source=estimate_rate_source,
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
        *,
        previous_record: StageRecord | None = None,
    ) -> None:
        observed_rate = self._usable_stage_rate(previous_record)
        if observed_rate is None:
            rate = self.profiles.estimated_fallback_tokens_per_second
            rate_source: Literal["fallback", "observed_previous_stage"] = "fallback"
        else:
            rate = observed_rate
            rate_source = "observed_previous_stage"
        progress(
            stage,
            message,
            tokens_per_second=rate,
            estimated_seconds_remaining=output_budget / rate,
            estimate_rate_source=rate_source,
        )

    @staticmethod
    def _review_as_stage(record: GroundingReviewRecord) -> StageRecord:
        return StageRecord(status=record.status, timing=record.timing)

    @staticmethod
    def _record_stage_selection(
        name: str,
        timing: StageTiming | None,
        selected_contexts: dict[str, int],
        estimated_inputs: dict[str, int],
    ) -> None:
        if timing is None:
            return
        if timing.selected_context is not None:
            selected_contexts[name] = timing.selected_context
        if timing.estimated_input_tokens is not None:
            estimated_inputs[name] = timing.estimated_input_tokens

    @staticmethod
    def _usable_stage_rate(record: StageRecord | None) -> float | None:
        if (
            record is None
            or record.status is not GenerationStatus.COMPLETED
            or record.timing is None
        ):
            return None
        rate = record.timing.tokens_per_second
        if rate is None or rate <= 0 or not math.isfinite(rate):
            return None
        return rate

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
        initial_grounding_review: GroundingReviewRecord | None = None,
        revision_record: StageRecord | None = None,
        final_grounding_review: GroundingReviewRecord | None = None,
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
            initial_grounding_review=initial_grounding_review,
            revision_result=revision_record,
            final_grounding_review=final_grounding_review,
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
                "grounding_review_enabled": (
                    bool(request.source_material) and request.profile.value in {"standard", "deep"}
                ),
                "grounding_review_prompt_version": (
                    None
                    if initial_grounding_review is None
                    else initial_grounding_review.prompt_version
                ),
                "grounding_revision_prompt_version": (
                    None if revision_record is None else revision_record.prompt_version
                ),
                "grounding_final_review_prompt_version": (
                    None
                    if final_grounding_review is None
                    else final_grounding_review.prompt_version
                ),
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
