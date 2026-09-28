"""Resumable terminal-based human review of evaluation responses."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

from lecture_slm.evaluation.evaluator import (
    CompletionStatus,
    DimensionReview,
    EvaluationResult,
    HumanReview,
)
from lecture_slm.evaluation.rubric import EvaluationRubric

ReviewInput = Callable[[str], str]
ReviewOutput = Callable[[str], None]
ReviewValue = Literal[1, 2, 3, 4, 5, "N/A"]


def _load_jsonl(path: Path, model: type[EvaluationResult] | type[HumanReview]) -> list[object]:
    if not path.exists():
        return []
    items: list[object] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                items.append(model.model_validate_json(line))
            except ValueError as error:
                raise ValueError(
                    f"Invalid {path.name} record on line {line_number}: {error}"
                ) from error
    return items


def _completed_reviews(reviews: list[HumanReview]) -> set[str]:
    return {
        review.prompt_id
        for review in reviews
        if review.dimensions and all(score.score is not None for score in review.dimensions)
    }


def _latest_successes(results: list[EvaluationResult]) -> list[EvaluationResult]:
    latest: dict[str, EvaluationResult] = {}
    for result in results:
        previous = latest.get(result.prompt_id)
        if previous is None or result.attempt >= previous.attempt:
            latest[result.prompt_id] = result
    return [
        result
        for result in latest.values()
        if result.completion_status is CompletionStatus.COMPLETED
    ]


def _ask_score(
    input_fn: ReviewInput,
    output_fn: ReviewOutput,
    label: str,
) -> tuple[ReviewValue | None, bool]:
    while True:
        try:
            value = input_fn(f"{label} (1-5, N/A, or q to stop): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            output_fn("Review stopped. Completed prompt reviews have been saved.")
            return None, True
        if value == "q":
            return None, True
        if value == "n/a":
            return "N/A", False
        if value in {"1", "2", "3", "4", "5"}:
            return cast(ReviewValue, int(value)), False
        output_fn("Enter 1, 2, 3, 4, 5, N/A, or q.")


def review_run(
    run_dir: Path,
    *,
    reviewer: str,
    rubric: EvaluationRubric | None = None,
    input_fn: ReviewInput = input,
    output_fn: ReviewOutput = print,
) -> int:
    """Review completed responses; existing completed reviews are never replaced."""

    response_path = run_dir / "responses.jsonl"
    review_path = run_dir / "review.jsonl"
    raw_results = _load_jsonl(response_path, EvaluationResult)
    results = [item for item in raw_results if isinstance(item, EvaluationResult)]
    raw_reviews = _load_jsonl(review_path, HumanReview)
    reviews = [item for item in raw_reviews if isinstance(item, HumanReview)]
    completed_ids = _completed_reviews(reviews)
    reviewed_count = 0

    for result in _latest_successes(results):
        if result.prompt_id in completed_ids:
            continue
        output_fn(f"\n=== {result.prompt_id} ({result.task.value}) ===")
        output_fn(f"Instruction:\n{result.instruction}")
        if result.context:
            output_fn(f"Context:\n{result.context}")
        if result.source_material:
            output_fn(f"Source material:\n{result.source_material}")
        output_fn("Expected characteristics:")
        for characteristic in result.expected_characteristics:
            output_fn(f"- {characteristic.description}")
        output_fn(f"\nModel response:\n{result.response or ''}")
        if rubric is not None:
            anchors = ", ".join(
                f"{score}={label}" for score, label in sorted(rubric.score_anchors.items())
            )
            output_fn(f"Score anchors: {anchors}; N/A = not applicable")
        criteria = (
            {}
            if rubric is None
            else {criterion.dimension: criterion for criterion in rubric.criteria}
        )

        dimensions: list[DimensionReview] = []
        stopped = False
        for dimension in result.evaluation_dimensions:
            if criterion := criteria.get(dimension):
                output_fn(f"\n{dimension.value}: {criterion.description}")
            score, stopped = _ask_score(input_fn, output_fn, dimension.value)
            if stopped:
                break
            notes = input_fn("Notes (optional): ").strip() or None
            strengths = input_fn("Observed strengths (optional): ").strip()
            problems = input_fn("Observed problems (optional): ").strip()
            dimensions.append(
                DimensionReview(
                    dimension=dimension,
                    score=score,
                    notes=notes,
                    observed_strengths=[strengths] if strengths else [],
                    observed_problems=[problems] if problems else [],
                )
            )
        if stopped:
            output_fn("Review stopped. Completed prompt reviews have been saved.")
            break
        review = HumanReview(prompt_id=result.prompt_id, reviewer=reviewer, dimensions=dimensions)
        with review_path.open("a", encoding="utf-8") as file:
            file.write(review.model_dump_json() + "\n")
            file.flush()
        completed_ids.add(result.prompt_id)
        reviewed_count += 1
        output_fn(f"Saved review for {result.prompt_id}.")

    output_fn(f"Saved {reviewed_count} new prompt review(s).")
    return reviewed_count
