"""Validate the local candidate review queue and provenance references."""

import argparse
import json
import sys
from pathlib import Path

from lecture_slm.ingestion.manifest import ManifestStore
from lecture_slm.ingestion.models import ReviewStatus
from lecture_slm.ingestion.review import CandidateStore


def validate(path: Path, manifest_path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Candidate store must be a JSON array")
    ids = [item.get("candidate_id") for item in payload if isinstance(item, dict)]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate IDs detected")
    store = CandidateStore(path)
    manifest_ids = {record.source_id for record in ManifestStore(manifest_path).records}
    errors: list[str] = []
    for candidate in store.candidates.values():
        if not candidate.source_ids or not candidate.provenance:
            errors.append(f"{candidate.candidate_id}: missing source provenance")
        missing_sources = set(candidate.source_ids) - manifest_ids
        if missing_sources:
            errors.append(f"{candidate.candidate_id}: unknown source IDs {sorted(missing_sources)}")
        if (
            candidate.review_status is ReviewStatus.APPROVED
            and candidate.quality_tier.value not in {"A", "C"}
        ):
            errors.append(f"{candidate.candidate_id}: approved candidate has invalid quality tier")
        if any("evals" in record.source_file.lower() for record in candidate.provenance):
            errors.append(f"{candidate.candidate_id}: evaluation provenance is not allowed")
    if errors:
        raise ValueError("Candidate validation failed:\n" + "\n".join(errors))
    print(f"Validated {len(store.candidates)} candidates from {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("data/ingestion/candidates.json"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/ingestion/manifests/sources.json"),
    )
    args = parser.parse_args()
    try:
        return validate(args.path, args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Candidate validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
