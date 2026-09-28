import json
from pathlib import Path
from typing import Any

from lecture_slm.config.loader import ModelConfig, load_model_config
from lecture_slm.inference.ollama_client import ChatResponse
from lecture_slm.performance.benchmark import (
    BENCHMARK_OUTPUT_LIMIT,
    BenchmarkStatus,
    BenchmarkStopReason,
    PromptSizeDistribution,
    build_context_sweep,
    build_think_comparison,
    choose_reasonable_context,
    detect_stop_reason,
    estimate_prompt_size_distribution,
    run_benchmark,
    tokens_per_second,
)

ROOT = Path(__file__).parents[1]


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def ensure_model_available(self, model: str) -> None:
        return None

    def chat(
        self,
        *,
        model: str,
        user_message: str,
        system_message: str | None = None,
        context: str | None = None,
        options: dict[str, str | int | float | bool] | None = None,
        think: bool | None = None,
        keep_alive: str | int | None = None,
        allow_empty_content: bool = False,
    ) -> ChatResponse:
        self.calls.append(
            {
                "model": model,
                "prompt": user_message,
                "options": options,
                "think": think,
                "keep_alive": keep_alive,
            }
        )
        return ChatResponse(
            model=model,
            content="A short measured answer.",
            prompt_tokens=80,
            completion_tokens=20,
            total_duration_ns=2_000_000_000,
            load_duration_ns=100_000_000,
            prompt_eval_duration_ns=500_000_000,
            eval_duration_ns=1_000_000_000,
            completion_reason="stop",
            thinking_content="brief internal reasoning",
        )


def test_tokens_per_second_and_zero_duration() -> None:
    assert tokens_per_second(24, 2_000_000_000) == 12.0
    assert tokens_per_second(24, 0) is None
    assert tokens_per_second(None, 2_000_000_000) is None


def test_stop_reason_detects_normal_stop_and_output_limit() -> None:
    normal = detect_stop_reason(
        completion_reason="stop",
        generated_tokens=100,
        num_predict=256,
    )
    capped = detect_stop_reason(
        completion_reason="stop",
        generated_tokens=256,
        num_predict=256,
    )
    timed_out = detect_stop_reason(
        completion_reason=None,
        generated_tokens=None,
        num_predict=256,
        error_type="OllamaTimeoutError",
        completed=False,
    )
    assert normal[0] is BenchmarkStopReason.NORMAL_STOP
    assert normal[1] is False
    assert capped[0] is BenchmarkStopReason.OUTPUT_LIMIT
    assert capped[1] is True
    assert timed_out[0] is BenchmarkStopReason.TIMEOUT
    assert timed_out[1] is None


def test_context_sweep_and_think_comparison_hold_controls_constant() -> None:
    context_cases = build_context_sweep()
    assert [case.num_ctx for case in context_cases] == [4096, 8192, 16384, 32768]
    assert {case.think for case in context_cases} == {False}
    assert {case.num_predict for case in context_cases} == {BENCHMARK_OUTPUT_LIMIT}

    think_cases = build_think_comparison(8192)
    assert [case.think for case in think_cases] == [False, True]
    assert {case.num_ctx for case in think_cases} == {8192}
    assert {case.num_predict for case in think_cases} == {BENCHMARK_OUTPUT_LIMIT}


def test_prompt_size_distribution_and_context_selection() -> None:
    sizes = estimate_prompt_size_distribution(["a" * 40, "b" * 80, "c" * 120])
    assert sizes.minimum_approx_tokens == 10
    assert sizes.median_approx_tokens == 20
    assert sizes.maximum_approx_tokens == 30
    assert choose_reasonable_context(sizes) == 4096


def test_benchmark_config_overrides_are_request_local_and_persisted(tmp_path: Path) -> None:
    config: ModelConfig = load_model_config(
        ROOT / "configs/models/qwen35-9b.yaml",
        environ={"OLLAMA_HOST": "http://benchmark.example:11434"},
    )
    distribution = PromptSizeDistribution(
        prompt_count=26,
        minimum_approx_tokens=200,
        median_approx_tokens=400,
        maximum_approx_tokens=900,
        maximum_characters=3600,
    )
    fake_client = FakeClient()
    run_dir = tmp_path / "benchmark-run"
    results = run_benchmark(
        run_dir=run_dir,
        model_config=config,
        project_root=ROOT,
        baseline_prompt_sizes=distribution,
        think_context_size=4096,
        context_sizes=(4096, 8192),
        num_predict=128,
        request_timeout_seconds=123.0,
        client=fake_client,  # type: ignore[arg-type]
    )

    assert len(results) == 4
    assert config.inference.context_length == 32768
    assert config.inference.max_output_tokens == 2048
    assert config.inference.think is True
    assert [call["options"]["num_ctx"] for call in fake_client.calls] == [4096, 8192, 4096, 4096]
    assert [call["think"] for call in fake_client.calls] == [False, False, False, True]
    assert {call["options"]["num_predict"] for call in fake_client.calls} == {128}
    assert all(call["prompt"] == fake_client.calls[0]["prompt"] for call in fake_client.calls)

    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["model_configuration"]["inference"]["host"] == "[redacted]"
    assert manifest["model_configuration"]["inference"]["context_length"] == 32768
    assert manifest["request_timeout_seconds"] == 123.0
    records = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 4
    assert records[0]["timing"]["generated_tokens_per_second"] == 20.0
    assert records[0]["thinking_output"] == "brief internal reasoning"
    assert records[0]["raw_completion_reason"] == "stop"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_count"] == 4
    assert summary["failed_count"] == 0
    assert results[0].status is BenchmarkStatus.COMPLETED
