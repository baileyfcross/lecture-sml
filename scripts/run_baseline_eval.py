"""Run a validated baseline evaluation and store resumable response records."""

import argparse
import sys
from pathlib import Path

from lecture_slm.config.loader import load_model_config
from lecture_slm.evaluation.runner import create_run_directory, execute_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/models/qwen35-9b.yaml"))
    parser.add_argument("--prompts", type=Path, default=Path("evals/prompts/baseline.jsonl"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Explicit per-run output cap; overrides the model config and is recorded",
    )
    parser.add_argument(
        "--think",
        choices=("configured", "enabled", "disabled"),
        default="configured",
        help="Explicit per-run thinking override; use configured for the model baseline",
    )
    parser.add_argument(
        "--rerun", action="store_true", help="Rerun successful prompts and append attempts"
    )
    args = parser.parse_args()
    try:
        config = load_model_config(args.config)
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be a positive integer")
        if args.max_output_tokens is not None and args.max_output_tokens < 1:
            raise ValueError("--max-output-tokens must be a positive integer")
        run_dir = args.run_dir or create_run_directory(Path("evals/results"))
        think_override = {
            "configured": None,
            "enabled": True,
            "disabled": False,
        }[args.think]
        summary = execute_evaluation(
            prompts_path=args.prompts,
            run_dir=run_dir,
            model_config=config,
            project_root=Path.cwd(),
            limit=args.limit,
            rerun=args.rerun,
            think_override=think_override,
            max_output_tokens_override=args.max_output_tokens,
        )
    except (OSError, ValueError) as error:
        print(f"Baseline evaluation failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Run {summary.run_id}: {summary.completed_count} completed, "
        f"{summary.failed_count} failed, {summary.skipped_count} skipped."
    )
    print(f"Results: {run_dir}")
    return 1 if summary.failed_count else 0


if __name__ == "__main__":
    sys.exit(main())
