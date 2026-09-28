"""Characterize Ollama inference speed without scoring educational quality."""

import argparse
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from lecture_slm.config.loader import load_model_config
from lecture_slm.performance.benchmark import (
    BENCHMARK_OUTPUT_LIMIT,
    CONTEXT_SIZES,
    DEFAULT_BENCHMARK_TIMEOUT_SECONDS,
    choose_reasonable_context,
    estimate_baseline_prompt_sizes,
    run_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/models/qwen35-9b.yaml"))
    parser.add_argument("--prompts", type=Path, default=Path("evals/prompts/baseline.jsonl"))
    parser.add_argument("--results-root", type=Path, default=Path("evals/performance"))
    parser.add_argument("--run-id")
    parser.add_argument("--think-context", type=int)
    parser.add_argument("--section", choices=("all", "context", "think"), default="all")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_BENCHMARK_TIMEOUT_SECONDS)
    parser.add_argument("--num-predict", type=int, default=BENCHMARK_OUTPUT_LIMIT)
    args = parser.parse_args()
    try:
        config = load_model_config(args.config)
        prompt_sizes = estimate_baseline_prompt_sizes(args.prompts, Path.cwd())
        context = (
            args.think_context
            if args.think_context is not None
            else choose_reasonable_context(prompt_sizes)
        )
        if context not in CONTEXT_SIZES:
            raise ValueError(f"--think-context must be one of {CONTEXT_SIZES}")
        if args.timeout_seconds <= 0 or args.num_predict <= 0:
            raise ValueError("timeout and num-predict must be positive")
        args.results_root.mkdir(parents=True, exist_ok=True)
        run_id = args.run_id or (
            "ollama-benchmark-"
            + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            + f"-{uuid.uuid4().hex[:8]}"
        )
        run_dir = args.results_root / run_id
        results = run_benchmark(
            run_dir=run_dir,
            model_config=config,
            project_root=Path.cwd(),
            baseline_prompt_sizes=prompt_sizes,
            think_context_size=context,
            request_timeout_seconds=args.timeout_seconds,
            num_predict=args.num_predict,
            section=args.section,
        )
    except (OSError, ValueError) as error:
        print(f"Ollama benchmark failed: {error}", file=sys.stderr)
        return 1

    print(
        "Approximate assembled evaluation input sizes (characters / 4; not tokenizer counts): "
        f"min={prompt_sizes.minimum_approx_tokens}, "
        f"median={prompt_sizes.median_approx_tokens}, "
        f"max={prompt_sizes.maximum_approx_tokens} tokens across "
        f"{prompt_sizes.prompt_count} prompts."
    )
    if args.section in {"all", "think"}:
        print(
            f"Thinking comparison context: {context} tokens "
            "(smallest tested window above approximate maximum input plus 50% headroom)."
        )
    failed = sum(result.status.value == "failed" for result in results)
    print(f"Benchmark cases: {len(results) - failed} completed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
