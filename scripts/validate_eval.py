"""Validate the evaluation prompt dataset and referenced profiles."""

import argparse
import sys
from pathlib import Path

from lecture_slm.evaluation.prompts import load_evaluation_prompts
from lecture_slm.evaluation.rubric import load_evaluation_rubric
from lecture_slm.evaluation.runner import validate_prompt_profiles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=Path("evals/prompts/baseline.jsonl"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--rubric", type=Path, default=Path("evals/rubric.yaml"))
    args = parser.parse_args()
    try:
        prompts = load_evaluation_prompts(args.prompts)
        validate_prompt_profiles(prompts, args.project_root)
        rubric = load_evaluation_rubric(args.rubric)
        rubric_dimensions = {criterion.dimension for criterion in rubric.criteria}
        for prompt in prompts:
            if not set(prompt.evaluation_dimensions).issubset(rubric_dimensions):
                raise ValueError(f"Prompt '{prompt.id}' uses dimensions absent from the rubric")
    except (OSError, ValueError) as error:
        print(f"Evaluation validation failed: {error}", file=sys.stderr)
        return 1
    versions = {prompt.version for prompt in prompts}
    print(f"Validated {len(prompts)} evaluation prompts (version {versions.pop()}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
