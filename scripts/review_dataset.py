"""Review candidate teaching examples in a resumable terminal workflow."""

import argparse
import sys
from pathlib import Path

from lecture_slm.ingestion.models import CandidateTrainingExample, ReviewStatus
from lecture_slm.ingestion.review import CandidateStore
from lecture_slm.schemas.dataset import QualityTier, TaskType

ACTIONS = {
    "a": ReviewStatus.APPROVED,
    "r": ReviewStatus.REJECTED,
    "n": ReviewStatus.NEEDS_EDIT,
    "d": ReviewStatus.DEFERRED,
    "x": ReviewStatus.DUPLICATE,
}


def print_candidate(candidate: CandidateTrainingExample) -> None:
    data = candidate.model_dump(mode="json")
    output = str(data.get("expected_output", ""))
    preview = output if len(output) <= 1200 else output[:1200] + "\n...[preview truncated]"
    print(f"\nCandidate: {data['candidate_id']}  status={data['review_status']}")
    print(f"Task: {data.get('task')}  course={data.get('course')}  level={data.get('level')}")
    print(f"Instruction ({data.get('instruction_source')}): {data.get('instruction')}")
    print(f"Quality tier: {data.get('quality_tier')}  confidence={data.get('confidence')}")
    print(f"Sources: {', '.join(data.get('source_ids', []))}")
    print(f"Expected output preview:\n{preview}")


def apply_action(
    store: CandidateStore,
    candidate_id: str,
    action: str,
    reviewer: str,
    notes: str | None,
) -> None:
    if action == "e":
        field = input("Field to edit (task/instruction/course/level/quality_tier/notes): ").strip()
        if field not in {
            "task",
            "instruction",
            "course",
            "level",
            "quality_tier",
            "reviewer_notes",
        }:
            raise ValueError("Unsupported review field")
        if (
            field == "quality_tier"
            and store.get(candidate_id).review_status is ReviewStatus.APPROVED
        ):
            raise ValueError("Approved quality tier cannot be edited")
        value = input("New value: ")
        if field == "task":
            value = TaskType(value)
        elif field == "quality_tier":
            value = QualityTier(value)
        store.update_fields(candidate_id, **{field: value})
        return
    if action not in ACTIONS:
        raise ValueError("Unknown action")
    store.transition(candidate_id, ACTIONS[action], reviewer=reviewer, notes=notes)
    print(f"Saved {candidate_id}: {ACTIONS[action].value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("data/ingestion/candidates.json"))
    parser.add_argument(
        "--status", choices=[status.value for status in ReviewStatus], default="pending"
    )
    parser.add_argument("--candidate-id")
    parser.add_argument("--action", choices=[*ACTIONS, "e"])
    parser.add_argument("--notes")
    parser.add_argument("--reviewer", default="local-reviewer")
    parser.add_argument("--course")
    parser.add_argument("--task")
    args = parser.parse_args()
    try:
        store = CandidateStore(args.path)
        if args.candidate_id and args.action:
            apply_action(store, args.candidate_id, args.action, args.reviewer, args.notes)
            return 0
        candidates = store.list(
            status=ReviewStatus(args.status), task=args.task, course=args.course
        )
        for candidate in candidates:
            print_candidate(candidate)
            prompt = "Action [a approve, r reject, n needs-edit, d defer, "
            prompt += "x duplicate, e edit, s skip, q quit]: "
            action = input(prompt).strip().lower()
            if action == "q":
                break
            if action in ACTIONS or action == "e":
                apply_action(store, candidate.candidate_id, action, args.reviewer, args.notes)
        return 0
    except (OSError, KeyError, ValueError) as error:
        print(f"Review failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
