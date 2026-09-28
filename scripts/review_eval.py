"""Review completed evaluation responses from a run directory in the terminal."""

import argparse
import sys
from pathlib import Path

from lecture_slm.evaluation.review import review_run
from lecture_slm.evaluation.rubric import load_evaluation_rubric


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--reviewer", default="reviewer")
    parser.add_argument("--rubric", type=Path, default=Path("evals/rubric.yaml"))
    args = parser.parse_args()
    try:
        rubric = load_evaluation_rubric(args.rubric)
        review_run(args.run_directory, reviewer=args.reviewer, rubric=rubric)
    except (OSError, ValueError) as error:
        print(f"Evaluation review failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
