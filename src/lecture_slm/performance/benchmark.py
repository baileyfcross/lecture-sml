"""Controlled Ollama performance benchmarking, isolated from quality scoring."""

import hashlib
import json
import logging
import math
import shutil
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from lecture_slm import __version__
from lecture_slm.config.loader import ModelConfig
from lecture_slm.evaluation.prompts import load_evaluation_prompts
from lecture_slm.evaluation.runner import SYSTEM_PROMPT, compose_evaluation_context
from lecture_slm.inference.ollama_client import ChatResponse, OllamaClient, OllamaError

LOGGER = logging.getLogger(__name__)
BENCHMARK_PROMPT = (
    "Create a concise introductory explanation of DNS for freshman Computer "
    "Information Systems students. Include one concrete example and one short "
    "check-for-understanding question."
)
CONTEXT_SIZES = (4096, 8192, 16384, 32768)
BENCHMARK_OUTPUT_LIMIT = 256
DEFAULT_BENCHMARK_TIMEOUT_SECONDS = 300.0


class BenchmarkStopReason(StrEnum):
    NORMAL_STOP = "normal_stop"
    OUTPUT_LIMIT = "output_limit"
    TIMEOUT = "timeout"
    ERROR = "error"
    UNKNOWN = "unknown"


class BenchmarkStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class BenchmarkCase(BaseModel):
    """One controlled request configuration."""

    case_id: str = Field(min_length=1)
    comparison: Literal["context_size", "think_mode"]
    num_ctx: int = Field(gt=0)
    think: bool
    num_predict: int = Field(gt=0)


class BenchmarkTiming(BaseModel):
    """Raw Ollama timing data and derived throughput values."""

    total_duration_ns: int | None = Field(default=None, ge=0)
    load_duration_ns: int | None = Field(default=None, ge=0)
    prompt_eval_duration_ns: int | None = Field(default=None, ge=0)
    eval_duration_ns: int | None = Field(default=None, ge=0)
    total_duration_seconds: float | None = Field(default=None, ge=0.0)
    load_duration_seconds: float | None = Field(default=None, ge=0.0)
    prompt_eval_duration_seconds: float | None = Field(default=None, ge=0.0)
    eval_duration_seconds: float | None = Field(default=None, ge=0.0)
    elapsed_seconds: float | None = Field(default=None, ge=0.0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    generated_tokens: int | None = Field(default=None, ge=0)
    prompt_tokens_per_second: float | None = Field(default=None, ge=0.0)
    generated_tokens_per_second: float | None = Field(default=None, ge=0.0)


class PromptSizeDistribution(BaseModel):
    """Approximate assembled evaluation prompt sizes, not tokenizer counts."""

    method: str = "ceil(assembled character count / 4); approximate only"
    prompt_count: int = Field(ge=0)
    minimum_approx_tokens: int = Field(ge=0)
    median_approx_tokens: int = Field(ge=0)
    maximum_approx_tokens: int = Field(ge=0)
    maximum_characters: int = Field(ge=0)


class BenchmarkRunManifest(BaseModel):
    """Benchmark provenance without storing a raw server address or secrets."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str = Field(min_length=1)
    provider: str = "ollama"
    server_fingerprint: str = Field(min_length=1)
    model_configuration: dict[str, Any]
    fixed_prompt: str = Field(min_length=1)
    prompt_sha256: str = Field(min_length=64, max_length=64)
    context_sizes: list[int] = Field(min_length=1)
    context_sweep_think: bool
    think_comparison_context: int = Field(gt=0)
    output_limit: int = Field(gt=0)
    request_timeout_seconds: float = Field(gt=0.0)
    seed: int
    temperature: float
    top_p: float
    baseline_prompt_sizes: PromptSizeDistribution
    git_commit: str | None = None
    project_version: str = Field(min_length=1)


class BenchmarkResult(BaseModel):
    """One success or failure from a context or thinking comparison case."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    case: BenchmarkCase
    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    response: str | None = None
    thinking_output: str | None = None
    thinking_characters: int | None = Field(default=None, ge=0)
    thinking_token_count: int | None = Field(default=None, ge=0)
    raw_completion_reason: str | None = None
    stop_reason: BenchmarkStopReason
    stop_reason_detail: str
    output_limit_reached: bool | None = None
    potentially_truncated: bool | None = None
    status: BenchmarkStatus
    timing: BenchmarkTiming = Field(default_factory=BenchmarkTiming)
    error_type: str | None = None
    error_message: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BenchmarkCaseSummary(BaseModel):
    """Compact, non-aggregated summary for one benchmark case."""

    case_id: str
    comparison: str
    num_ctx: int
    think: bool
    status: BenchmarkStatus
    stop_reason: BenchmarkStopReason
    output_limit_reached: bool | None
    prompt_tokens_per_second: float | None
    generated_tokens_per_second: float | None
    total_duration_seconds: float | None
    generated_tokens: int | None


class BenchmarkRunSummary(BaseModel):
    """Run counts and per-case measurements; no score or winner is calculated."""

    run_id: str
    case_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    baseline_prompt_sizes: PromptSizeDistribution
    cases: list[BenchmarkCaseSummary]
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def tokens_per_second(token_count: int | None, duration_ns: int | None) -> float | None:
    """Calculate token throughput; return None when timing is absent or zero."""

    if token_count is None or duration_ns is None or duration_ns <= 0:
        return None
    return round(token_count * 1_000_000_000 / duration_ns, 3)


def detect_stop_reason(
    *,
    completion_reason: str | None,
    generated_tokens: int | None,
    num_predict: int,
    error_type: str | None = None,
    completed: bool = True,
) -> tuple[BenchmarkStopReason, bool | None, str]:
    """Classify stop reason using Ollama metadata with token-count fallback."""

    if error_type and "timeout" in error_type.lower():
        return BenchmarkStopReason.TIMEOUT, None, "request timed out"
    if error_type:
        return (
            BenchmarkStopReason.ERROR,
            None,
            "request failed before a completion reason was available",
        )
    reason = (completion_reason or "").strip().lower()
    if reason in {"length", "max_tokens", "max_token", "limit", "token_limit"}:
        return BenchmarkStopReason.OUTPUT_LIMIT, True, f"Ollama done_reason={reason}"
    if generated_tokens is not None and generated_tokens >= num_predict:
        return BenchmarkStopReason.OUTPUT_LIMIT, True, "generated token count reached num_predict"
    if reason in {"stop", "end_turn", "eos"}:
        return BenchmarkStopReason.NORMAL_STOP, False, f"Ollama done_reason={reason}"
    if completed:
        if reason:
            return BenchmarkStopReason.NORMAL_STOP, False, f"completed with done_reason={reason}"
        return (
            BenchmarkStopReason.NORMAL_STOP,
            False,
            "completed below token cap; done_reason unavailable",
        )
    return BenchmarkStopReason.UNKNOWN, None, "completion reason unavailable"


def build_context_sweep(
    context_sizes: Sequence[int] = CONTEXT_SIZES,
    *,
    think: bool = False,
    num_predict: int = BENCHMARK_OUTPUT_LIMIT,
) -> list[BenchmarkCase]:
    """Build comparable fixed-prompt cases that vary only context size."""

    return [
        BenchmarkCase(
            case_id=f"context-{context_size}-think-{str(think).lower()}",
            comparison="context_size",
            num_ctx=context_size,
            think=think,
            num_predict=num_predict,
        )
        for context_size in context_sizes
    ]


def build_think_comparison(
    context_size: int,
    *,
    num_predict: int = BENCHMARK_OUTPUT_LIMIT,
) -> list[BenchmarkCase]:
    """Build a think=false/true pair with all other request controls held constant."""

    return [
        BenchmarkCase(
            case_id=f"think-{str(think).lower()}-ctx-{context_size}",
            comparison="think_mode",
            num_ctx=context_size,
            think=think,
            num_predict=num_predict,
        )
        for think in (False, True)
    ]


def estimate_prompt_size_distribution(
    assembled_prompts: Sequence[str],
) -> PromptSizeDistribution:
    """Estimate token sizes from characters; this is not an Ollama tokenizer."""

    estimates = sorted(math.ceil(len(prompt) / 4) for prompt in assembled_prompts)
    if not estimates:
        return PromptSizeDistribution(
            prompt_count=0,
            minimum_approx_tokens=0,
            median_approx_tokens=0,
            maximum_approx_tokens=0,
            maximum_characters=0,
        )
    middle = len(estimates) // 2
    if len(estimates) % 2:
        median = estimates[middle]
    else:
        median = math.ceil((estimates[middle - 1] + estimates[middle]) / 2)
    return PromptSizeDistribution(
        prompt_count=len(estimates),
        minimum_approx_tokens=estimates[0],
        median_approx_tokens=median,
        maximum_approx_tokens=estimates[-1],
        maximum_characters=max(len(prompt) for prompt in assembled_prompts),
    )


def estimate_baseline_prompt_sizes(
    prompts_path: Path,
    project_root: Path,
) -> PromptSizeDistribution:
    """Estimate complete assembled baseline inputs, including configured profiles."""

    prompts = load_evaluation_prompts(prompts_path)
    assembled = [
        f"{SYSTEM_PROMPT}\n\n{compose_evaluation_context(prompt, project_root)}"
        for prompt in prompts
    ]
    return estimate_prompt_size_distribution(assembled)


def choose_reasonable_context(
    distribution: PromptSizeDistribution,
    context_sizes: Sequence[int] = CONTEXT_SIZES,
    *,
    headroom_multiplier: float = 1.5,
) -> int:
    """Choose the smallest measured window above max prompt estimate plus headroom."""

    target = math.ceil(distribution.maximum_approx_tokens * headroom_multiplier)
    for context_size in sorted(context_sizes):
        if context_size >= target:
            return context_size
    return max(context_sizes)


def _seconds(duration_ns: int | None) -> float | None:
    return None if duration_ns is None else duration_ns / 1_000_000_000


def _timing(response: ChatResponse, elapsed_seconds: float) -> BenchmarkTiming:
    return BenchmarkTiming(
        total_duration_ns=response.total_duration_ns,
        load_duration_ns=response.load_duration_ns,
        prompt_eval_duration_ns=response.prompt_eval_duration_ns,
        eval_duration_ns=response.eval_duration_ns,
        total_duration_seconds=_seconds(response.total_duration_ns),
        load_duration_seconds=_seconds(response.load_duration_ns),
        prompt_eval_duration_seconds=_seconds(response.prompt_eval_duration_ns),
        eval_duration_seconds=_seconds(response.eval_duration_ns),
        elapsed_seconds=elapsed_seconds,
        prompt_tokens=response.prompt_tokens,
        generated_tokens=response.completion_tokens,
        prompt_tokens_per_second=tokens_per_second(
            response.prompt_tokens,
            response.prompt_eval_duration_ns,
        ),
        generated_tokens_per_second=tokens_per_second(
            response.completion_tokens,
            response.eval_duration_ns,
        ),
    )


def _server_fingerprint(host: str) -> str:
    parts = urlsplit(host)
    normalized = f"{parts.scheme}://{parts.hostname or ''}:{parts.port or ''}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _git_commit(project_root: Path) -> str | None:
    executable = shutil.which("git")
    if executable is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 - fixed args, resolved executable, no shell
            [executable, "rev-parse", "HEAD"],
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


def _append_result(path: Path, result: BenchmarkResult) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(result.model_dump_json() + "\n")
        file.flush()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _make_success_result(
    *,
    run_id: str,
    case: BenchmarkCase,
    model: str,
    prompt: str,
    response: ChatResponse,
    elapsed_seconds: float,
) -> BenchmarkResult:
    reason, limit_reached, detail = detect_stop_reason(
        completion_reason=response.completion_reason,
        generated_tokens=response.completion_tokens,
        num_predict=case.num_predict,
    )
    thinking = response.thinking_content
    return BenchmarkResult(
        run_id=run_id,
        case=case,
        model=model,
        prompt=prompt,
        response=response.content,
        thinking_output=thinking,
        thinking_characters=None if thinking is None else len(thinking),
        raw_completion_reason=response.completion_reason,
        stop_reason=reason,
        stop_reason_detail=detail,
        output_limit_reached=limit_reached,
        potentially_truncated=limit_reached,
        status=BenchmarkStatus.COMPLETED,
        timing=_timing(response, elapsed_seconds),
    )


def _make_failure_result(
    *,
    run_id: str,
    case: BenchmarkCase,
    model: str,
    prompt: str,
    error: Exception,
    host: str,
    elapsed_seconds: float,
) -> BenchmarkResult:
    reason, limit_reached, detail = detect_stop_reason(
        completion_reason=None,
        generated_tokens=None,
        num_predict=case.num_predict,
        error_type=type(error).__name__,
        completed=False,
    )
    return BenchmarkResult(
        run_id=run_id,
        case=case,
        model=model,
        prompt=prompt,
        stop_reason=reason,
        stop_reason_detail=detail,
        output_limit_reached=limit_reached,
        potentially_truncated=limit_reached,
        status=BenchmarkStatus.FAILED,
        timing=BenchmarkTiming(elapsed_seconds=elapsed_seconds),
        error_type=type(error).__name__,
        error_message=str(error).replace(host, "[configured Ollama server]"),
    )


def _summarize_result(result: BenchmarkResult) -> BenchmarkCaseSummary:
    return BenchmarkCaseSummary(
        case_id=result.case.case_id,
        comparison=result.case.comparison,
        num_ctx=result.case.num_ctx,
        think=result.case.think,
        status=result.status,
        stop_reason=result.stop_reason,
        output_limit_reached=result.output_limit_reached,
        prompt_tokens_per_second=result.timing.prompt_tokens_per_second,
        generated_tokens_per_second=result.timing.generated_tokens_per_second,
        total_duration_seconds=result.timing.total_duration_seconds,
        generated_tokens=result.timing.generated_tokens,
    )


def _print_table(title: str, results: Sequence[BenchmarkResult]) -> None:
    print(title)
    print(
        "Context | Think | Prompt tok/s | Generation tok/s | Total sec | Generated | "
        "Stop reason | Cap reached"
    )
    for result in results:
        timing = result.timing
        print(
            f"{result.case.num_ctx:<7} | {str(result.case.think).lower():<5} | "
            f"{_display(timing.prompt_tokens_per_second):<12} | "
            f"{_display(timing.generated_tokens_per_second):<16} | "
            f"{_display(timing.total_duration_seconds):<9} | "
            f"{_display(timing.generated_tokens):<9} | {result.stop_reason.value} | "
            f"{result.output_limit_reached}"
        )


def _display(value: float | int | None) -> str:
    return "-" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)


def run_benchmark(
    *,
    run_dir: Path,
    model_config: ModelConfig,
    project_root: Path,
    baseline_prompt_sizes: PromptSizeDistribution,
    think_context_size: int,
    request_timeout_seconds: float = DEFAULT_BENCHMARK_TIMEOUT_SECONDS,
    context_sizes: Sequence[int] = CONTEXT_SIZES,
    num_predict: int = BENCHMARK_OUTPUT_LIMIT,
    section: Literal["all", "context", "think"] = "all",
    client: OllamaClient | None = None,
) -> list[BenchmarkResult]:
    """Run context and thinking comparisons sequentially and persist every case."""

    if model_config.inference.provider != "ollama":
        raise ValueError("Performance benchmark currently supports only Ollama")
    if request_timeout_seconds <= 0 or num_predict <= 0:
        raise ValueError("timeout and output-token limit must be positive")
    if section not in {"all", "context", "think"}:
        raise ValueError("section must be all, context, or think")
    run_dir.mkdir(parents=True, exist_ok=False)
    run_id = run_dir.name
    model = model_config.ollama_name
    cases: list[BenchmarkCase] = []
    if section in {"all", "context"}:
        cases.extend(build_context_sweep(context_sizes, think=False, num_predict=num_predict))
    if section in {"all", "think"}:
        cases.extend(build_think_comparison(think_context_size, num_predict=num_predict))
    if not cases:
        raise ValueError("benchmark must include at least one case")
    sanitized_config: dict[str, Any] = model_config.model_dump(mode="json")
    sanitized_config["inference"]["host"] = "[redacted]"
    manifest = BenchmarkRunManifest(
        run_id=run_id,
        model=model,
        server_fingerprint=_server_fingerprint(model_config.inference.host),
        model_configuration=sanitized_config,
        fixed_prompt=BENCHMARK_PROMPT,
        prompt_sha256=hashlib.sha256(BENCHMARK_PROMPT.encode("utf-8")).hexdigest(),
        context_sizes=list(context_sizes),
        context_sweep_think=False,
        think_comparison_context=think_context_size,
        output_limit=num_predict,
        request_timeout_seconds=request_timeout_seconds,
        seed=model_config.inference.seed,
        temperature=model_config.inference.temperature,
        top_p=model_config.inference.top_p,
        baseline_prompt_sizes=baseline_prompt_sizes,
        git_commit=_git_commit(project_root),
        project_version=__version__,
    )
    (run_dir / "run.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    results_path = run_dir / "results.jsonl"
    results_path.touch()
    active_client = client or OllamaClient(
        model_config.inference.host,
        timeout=request_timeout_seconds,
    )
    preflight_error: OllamaError | None = None
    try:
        active_client.ensure_model_available(model)
    except OllamaError as error:
        preflight_error = error
        LOGGER.error("Ollama benchmark preflight failed: %s", error)

    results: list[BenchmarkResult] = []
    for case_number, case in enumerate(cases, start=1):
        print(
            f"Running benchmark case {case_number}/{len(cases)}: {case.case_id}",
            flush=True,
        )
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        try:
            if preflight_error is not None:
                raise preflight_error
            response = active_client.chat(
                model=model,
                user_message=BENCHMARK_PROMPT,
                options={
                    "temperature": model_config.inference.temperature,
                    "top_p": model_config.inference.top_p,
                    "seed": model_config.inference.seed,
                    "num_ctx": case.num_ctx,
                    "num_predict": case.num_predict,
                },
                think=case.think,
                keep_alive=model_config.inference.keep_alive,
                allow_empty_content=True,
            )
            result = _make_success_result(
                run_id=run_id,
                case=case,
                model=model,
                prompt=BENCHMARK_PROMPT,
                response=response,
                elapsed_seconds=time.perf_counter() - started,
            ).model_copy(update={"started_at": started_at})
        except OllamaError as error:
            LOGGER.error("Benchmark case %s failed: %s", case.case_id, error)
            result = _make_failure_result(
                run_id=run_id,
                case=case,
                model=model,
                prompt=BENCHMARK_PROMPT,
                error=error,
                host=model_config.inference.host,
                elapsed_seconds=time.perf_counter() - started,
            ).model_copy(update={"started_at": started_at})
        _append_result(results_path, result)
        results.append(result)

    summary = BenchmarkRunSummary(
        run_id=run_id,
        case_count=len(results),
        completed_count=sum(result.status is BenchmarkStatus.COMPLETED for result in results),
        failed_count=sum(result.status is BenchmarkStatus.FAILED for result in results),
        baseline_prompt_sizes=baseline_prompt_sizes,
        cases=[_summarize_result(result) for result in results],
    )
    _write_json(run_dir / "summary.json", summary.model_dump(mode="json"))
    if section in {"all", "context"}:
        _print_table("Context-size comparison (think=false)", results[: len(context_sizes)])
    if section in {"all", "think"}:
        offset = len(context_sizes) if section == "all" else 0
        _print_table("Thinking comparison", results[offset:])
    print(f"Saved benchmark results to {run_dir}")
    return results
