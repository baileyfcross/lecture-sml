"""Sequential baseline evaluation execution, persistence, and resume support."""

import hashlib
import json
import logging
import re
import shutil
import subprocess
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from lecture_slm import __version__
from lecture_slm.config.loader import ModelConfig, load_course_config, load_pedagogy_config
from lecture_slm.evaluation.evaluator import (
    CompletionStatus,
    EvaluationResult,
    EvaluationRunSummary,
    EvaluationTiming,
    GenerationConfiguration,
    RunManifest,
)
from lecture_slm.evaluation.prompts import EvaluationPrompt, load_evaluation_prompts
from lecture_slm.inference.ollama_client import ChatResponse, OllamaClient, OllamaError
from lecture_slm.schemas.dataset import TaskType

LOGGER = logging.getLogger(__name__)
SYSTEM_PROMPT = (
    "You are an educational assistant helping an instructor prepare teaching materials. "
    "Follow the requested task and supplied course context. Respect source restrictions, "
    "and do not claim unsupported facts."
)


def generation_configuration(
    config: ModelConfig,
    *,
    task: TaskType | None = None,
    think_override: bool | None = None,
    max_output_tokens_override: int | None = None,
) -> GenerationConfiguration:
    """Capture every baseline sampling control sent with evaluation requests."""

    inference = config.inference
    return GenerationConfiguration(
        think=inference.think if think_override is None else think_override,
        keep_alive=inference.keep_alive,
        temperature=inference.temperature,
        top_p=inference.top_p,
        seed=inference.seed,
        num_ctx=inference.context_length,
        num_predict=(
            max_output_tokens_override
            if max_output_tokens_override is not None
            else inference.max_output_tokens
            if task is None
            else inference.output_tokens_for(task)
        ),
        request_timeout_seconds=inference.request_timeout_seconds,
    )


def create_run_directory(results_root: Path, run_id: str | None = None) -> Path:
    """Create a new run directory without overwriting an existing run."""

    safe_run_id = run_id or (
        datetime.now(UTC).strftime("baseline-%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    )
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", safe_run_id):
        raise ValueError("run_id may contain only letters, numbers, dots, underscores, and hyphens")
    run_dir = results_root / safe_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _server_fingerprint(host: str) -> str:
    parts = urlsplit(host)
    normalized = f"{parts.scheme}://{parts.hostname or ''}:{parts.port or ''}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _git_commit(project_root: Path) -> str | None:
    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 - fixed arguments, resolved executable, no shell
            [git_executable, "rev-parse", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def _profile_hashes(prompts: list[EvaluationPrompt], project_root: Path) -> dict[str, str]:
    references = {
        reference
        for prompt in prompts
        for reference in (prompt.course_profile, prompt.pedagogy_profile)
        if reference is not None
    }
    return {
        reference: hashlib.sha256(
            _resolve_profile(reference, project_root).read_bytes()
        ).hexdigest()
        for reference in sorted(references)
    }


def _resolve_profile(reference: str, project_root: Path) -> Path:
    path = Path(reference)
    return path if path.is_absolute() else project_root / path


def compose_evaluation_context(prompt: EvaluationPrompt, project_root: Path) -> str:
    sections: list[str] = []
    if prompt.context:
        sections.append(f"Context:\n{prompt.context}")
    if prompt.source_material:
        sections.append(f"Supplied source material:\n{prompt.source_material}")
    if prompt.course_profile:
        course = load_course_config(_resolve_profile(prompt.course_profile, project_root))
        sections.append(f"Course profile:\n{course.model_dump_json(indent=2)}")
    if prompt.pedagogy_profile:
        pedagogy = load_pedagogy_config(_resolve_profile(prompt.pedagogy_profile, project_root))
        sections.append(f"Pedagogy profile:\n{pedagogy.model_dump_json(indent=2)}")
    sections.append(f"Task instruction:\n{prompt.instruction}")
    return "\n\n".join(sections)


def validate_prompt_profiles(prompts: list[EvaluationPrompt], project_root: Path) -> None:
    """Validate every referenced course and pedagogy profile before generation."""

    for prompt in prompts:
        compose_evaluation_context(prompt, project_root)


def _build_manifest(
    *,
    run_id: str,
    model_config: ModelConfig,
    generation: GenerationConfiguration,
    task_generations: dict[str, GenerationConfiguration],
    prompts: list[EvaluationPrompt],
    prompts_path: Path,
    project_root: Path,
) -> RunManifest:
    sanitized_config: dict[str, Any] = model_config.model_dump(mode="json")
    sanitized_config["inference"]["host"] = "[redacted]"
    return RunManifest(
        run_id=run_id,
        model=model_config.ollama_name,
        server_fingerprint=_server_fingerprint(model_config.inference.host),
        model_configuration=sanitized_config,
        generation_configuration=generation,
        task_generation_configurations=task_generations,
        git_commit=_git_commit(project_root),
        evaluation_dataset_version=prompts[0].version,
        evaluation_dataset_sha256=hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        profile_config_sha256=_profile_hashes(prompts, project_root),
        prompt_count=len(prompts),
        random_seed=model_config.inference.seed,
        project_version=__version__,
    )


def _read_results(path: Path) -> list[EvaluationResult]:
    records: list[EvaluationResult] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(EvaluationResult.model_validate_json(line))
            except ValueError as error:
                raise ValueError(
                    f"Invalid response record on line {line_number}: {error}"
                ) from error
    return records


def _append_result(path: Path, result: EvaluationResult) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(result.model_dump_json() + "\n")
        file.flush()


def _attempt_numbers(results: list[EvaluationResult]) -> Counter[str]:
    attempts: Counter[str] = Counter()
    for result in results:
        attempts[result.prompt_id] = max(attempts[result.prompt_id], result.attempt)
    return attempts


def _completed_ids(results: list[EvaluationResult]) -> set[str]:
    return {
        result.prompt_id
        for result in results
        if result.completion_status is CompletionStatus.COMPLETED
    }


def pending_prompt_ids(
    prompts: list[EvaluationPrompt],
    results: list[EvaluationResult],
    *,
    rerun: bool = False,
) -> list[str]:
    """Select prompts not yet successful, or every prompt when explicitly rerunning."""

    completed = _completed_ids(results)
    return [prompt.id for prompt in prompts if rerun or prompt.id not in completed]


def _duration_seconds(duration_ns: int | None) -> float | None:
    return None if duration_ns is None else duration_ns / 1_000_000_000


def _timing(response: ChatResponse, elapsed_seconds: float) -> EvaluationTiming:
    return EvaluationTiming(
        total_duration_ns=response.total_duration_ns,
        load_duration_ns=response.load_duration_ns,
        prompt_eval_duration_ns=response.prompt_eval_duration_ns,
        eval_duration_ns=response.eval_duration_ns,
        total_duration_seconds=_duration_seconds(response.total_duration_ns),
        load_duration_seconds=_duration_seconds(response.load_duration_ns),
        prompt_eval_duration_seconds=_duration_seconds(response.prompt_eval_duration_ns),
        eval_duration_seconds=_duration_seconds(response.eval_duration_ns),
        elapsed_seconds=elapsed_seconds,
        prompt_tokens=response.prompt_tokens,
        generated_tokens=response.completion_tokens,
    )


def _safe_error_message(error: Exception, host: str) -> str:
    return str(error).replace(host, "[configured Ollama server]")


def _make_result(
    prompt: EvaluationPrompt,
    *,
    model: str,
    generation: GenerationConfiguration,
    attempt: int,
    response: ChatResponse | None = None,
    error: Exception | None = None,
    host: str,
    elapsed_seconds: float | None = None,
) -> EvaluationResult:
    if response is not None:
        return EvaluationResult(
            prompt_id=prompt.id,
            model=model,
            task=prompt.task,
            instruction=prompt.instruction,
            course_profile=prompt.course_profile,
            pedagogy_profile=prompt.pedagogy_profile,
            context=prompt.context,
            source_material=prompt.source_material,
            expected_characteristics=prompt.expected_characteristics,
            evaluation_dimensions=prompt.evaluation_dimensions,
            response=response.content,
            generation_configuration=generation,
            timing=_timing(response, elapsed_seconds or 0.0),
            completion_status=CompletionStatus.COMPLETED,
            attempt=attempt,
        )
    if error is None:
        raise ValueError("A failed evaluation result requires an error")
    return EvaluationResult(
        prompt_id=prompt.id,
        model=model,
        task=prompt.task,
        instruction=prompt.instruction,
        course_profile=prompt.course_profile,
        pedagogy_profile=prompt.pedagogy_profile,
        context=prompt.context,
        source_material=prompt.source_material,
        expected_characteristics=prompt.expected_characteristics,
        evaluation_dimensions=prompt.evaluation_dimensions,
        response=None,
        generation_configuration=generation,
        timing=EvaluationTiming(elapsed_seconds=elapsed_seconds),
        completion_status=CompletionStatus.FAILED,
        error_type=type(error).__name__,
        error_message=_safe_error_message(error, host),
        attempt=attempt,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _validate_or_write_manifest(path: Path, manifest: RunManifest) -> None:
    if path.exists():
        existing = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        immutable_fields = (
            "model",
            "server_fingerprint",
            "evaluation_dataset_version",
            "evaluation_dataset_sha256",
            "profile_config_sha256",
            "prompt_count",
            "generation_configuration",
        )
        for field_name in immutable_fields:
            if getattr(existing, field_name) != getattr(manifest, field_name):
                raise ValueError(f"Cannot resume run: manifest field '{field_name}' differs")
        return
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def _write_summary(
    run_dir: Path,
    *,
    run_id: str,
    prompts: list[EvaluationPrompt],
    results: list[EvaluationResult],
    skipped_count: int,
) -> EvaluationRunSummary:
    by_id: dict[str, EvaluationResult] = {}
    for result in results:
        if result.prompt_id not in by_id or result.attempt >= by_id[result.prompt_id].attempt:
            by_id[result.prompt_id] = result
    successful_ids = _completed_ids(results)
    summary = EvaluationRunSummary(
        run_id=run_id,
        prompt_count=len(prompts),
        completed_count=len(successful_ids),
        failed_count=sum(
            1
            for prompt_id, result in by_id.items()
            if result.completion_status is CompletionStatus.FAILED
            and prompt_id not in successful_ids
        ),
        skipped_count=skipped_count,
    )
    _write_json(run_dir / "summary.json", summary.model_dump(mode="json"))
    return summary


def execute_evaluation(
    *,
    prompts_path: Path,
    run_dir: Path,
    model_config: ModelConfig,
    project_root: Path,
    limit: int | None = None,
    rerun: bool = False,
    think_override: bool | None = None,
    max_output_tokens_override: int | None = None,
    client: OllamaClient | None = None,
) -> EvaluationRunSummary:
    """Validate, execute sequentially, append results, and summarize a baseline run."""

    if model_config.inference.provider != "ollama":
        raise ValueError("Baseline evaluation currently supports only the Ollama provider")
    if max_output_tokens_override is not None and max_output_tokens_override < 1:
        raise ValueError("max_output_tokens_override must be a positive integer")
    all_prompts = load_evaluation_prompts(prompts_path)
    all_contexts = {
        prompt.id: compose_evaluation_context(prompt, project_root) for prompt in all_prompts
    }
    prompts = all_prompts
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be a positive integer")
        prompts = prompts[:limit]
    contexts = {prompt.id: all_contexts[prompt.id] for prompt in prompts}
    generation = generation_configuration(
        model_config,
        think_override=think_override,
        max_output_tokens_override=max_output_tokens_override,
    )
    task_generations = {
        task.value: generation_configuration(
            model_config,
            task=task,
            think_override=think_override,
            max_output_tokens_override=max_output_tokens_override,
        )
        for task in {prompt.task for prompt in prompts}
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    responses_path = run_dir / "responses.jsonl"
    reviews_path = run_dir / "review.jsonl"
    responses_path.touch(exist_ok=True)
    reviews_path.touch(exist_ok=True)
    run_id = run_dir.name
    manifest = _build_manifest(
        run_id=run_id,
        model_config=model_config,
        generation=generation,
        task_generations=task_generations,
        prompts=prompts,
        prompts_path=prompts_path,
        project_root=project_root,
    )
    _validate_or_write_manifest(run_dir / "run.json", manifest)

    previous_results = _read_results(responses_path)
    pending_ids = pending_prompt_ids(prompts, previous_results, rerun=rerun)
    skipped_count = len(prompts) - len(pending_ids)
    attempts = _attempt_numbers(previous_results)
    for prompt_id in pending_ids:
        attempts[prompt_id] += 1
    pending_prompts = [prompt for prompt in prompts if prompt.id in pending_ids]
    active_client = client or OllamaClient(
        model_config.inference.host,
        timeout=generation.request_timeout_seconds,
    )

    preflight_error: OllamaError | None = None
    if pending_prompts:
        try:
            active_client.ensure_model_available(model_config.ollama_name)
        except OllamaError as error:
            preflight_error = error
            LOGGER.error(
                "Ollama preflight failed: %s",
                _safe_error_message(error, model_config.inference.host),
            )

    for prompt in pending_prompts:
        prompt_generation = task_generations[prompt.task.value]
        started = time.perf_counter()
        try:
            if preflight_error is not None:
                raise preflight_error
            response = active_client.chat(
                model=model_config.ollama_name,
                system_message=SYSTEM_PROMPT,
                user_message=contexts[prompt.id],
                options={
                    "temperature": prompt_generation.temperature,
                    "top_p": prompt_generation.top_p,
                    "seed": prompt_generation.seed,
                    "num_ctx": prompt_generation.num_ctx,
                    "num_predict": prompt_generation.num_predict,
                },
                think=prompt_generation.think,
                keep_alive=prompt_generation.keep_alive,
            )
            record = _make_result(
                prompt,
                model=model_config.ollama_name,
                generation=prompt_generation,
                attempt=attempts[prompt.id],
                response=response,
                host=model_config.inference.host,
                elapsed_seconds=time.perf_counter() - started,
            )
        except OllamaError as error:
            LOGGER.error(
                "Prompt %s failed: %s",
                prompt.id,
                _safe_error_message(error, model_config.inference.host),
            )
            record = _make_result(
                prompt,
                model=model_config.ollama_name,
                generation=prompt_generation,
                attempt=attempts[prompt.id],
                error=error,
                host=model_config.inference.host,
                elapsed_seconds=time.perf_counter() - started,
            )
        _append_result(responses_path, record)
        previous_results.append(record)

    summary = _write_summary(
        run_dir,
        run_id=run_id,
        prompts=prompts,
        results=previous_results,
        skipped_count=skipped_count,
    )
    LOGGER.info(
        "Run %s: %d completed, %d failed, %d skipped",
        run_id,
        summary.completed_count,
        summary.failed_count,
        summary.skipped_count,
    )
    return summary
