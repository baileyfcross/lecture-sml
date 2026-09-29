"""Print compact statistics for pending candidates or approved exports."""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from lecture_slm.ingestion.review import CandidateStore
from lecture_slm.schemas.dataset import DatasetExample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/ingestion/candidates.json"))
    parser.add_argument("--approved", type=Path)
    args = parser.parse_args()
    try:
        if args.approved:
            records = [
                DatasetExample.model_validate(json.loads(line))
                for line in args.approved.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            tasks = Counter(record.task.value for record in records)
            courses = Counter(record.course for record in records)
            lengths = [len(record.response) for record in records]
            print(
                json.dumps(
                    {
                        "count": len(records),
                        "tasks": tasks,
                        "courses": courses,
                        "response_chars": _lengths(lengths),
                    },
                    indent=2,
                    default=dict,
                )
            )
        else:
            candidates = list(CandidateStore(args.candidates).candidates.values())
            print(
                json.dumps(
                    {
                        "count": len(candidates),
                        "statuses": dict(
                            Counter(candidate.review_status.value for candidate in candidates)
                        ),
                        "tasks": dict(
                            Counter(
                                candidate.task.value for candidate in candidates if candidate.task
                            )
                        ),
                        "courses": dict(
                            Counter(
                                candidate.course for candidate in candidates if candidate.course
                            )
                        ),
                        "quality_tiers": dict(
                            Counter(candidate.quality_tier.value for candidate in candidates)
                        ),
                        "authorship": dict(
                            Counter(candidate.authorship.value for candidate in candidates)
                        ),
                        "source_formats": dict(
                            Counter(
                                record.source_file.rsplit(".", 1)[-1]
                                for candidate in candidates
                                for record in candidate.provenance
                            )
                        ),
                        "response_chars": _lengths(
                            [len(candidate.expected_output) for candidate in candidates]
                        ),
                    },
                    indent=2,
                )
            )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Statistics failed: {error}", file=sys.stderr)
        return 1
    return 0


def _lengths(values: list[int]) -> dict[str, float | int | None]:
    return {
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "max": max(values) if values else None,
    }


if __name__ == "__main__":
    sys.exit(main())
