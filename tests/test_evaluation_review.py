from pathlib import Path

from lecture_slm.evaluation.evaluator import (
    CompletionStatus,
    EvaluationResult,
    GenerationConfiguration,
    HumanReview,
)
from lecture_slm.evaluation.prompts import ExpectedCharacteristic
from lecture_slm.evaluation.review import review_run
from lecture_slm.evaluation.rubric import EvaluationDimension
from lecture_slm.schemas.dataset import TaskType


def make_result(prompt_id: str) -> EvaluationResult:
    return EvaluationResult(
        prompt_id=prompt_id,
        model="qwen3.5:9b",
        task=TaskType.EXPLANATION,
        instruction=f"Explain the concept for {prompt_id}.",
        expected_characteristics=[ExpectedCharacteristic(description="Uses a clear example.")],
        evaluation_dimensions=[EvaluationDimension.CLARITY],
        response="A completed model response.",
        generation_configuration=GenerationConfiguration(
            think=True,
            keep_alive="10m",
            temperature=0.5,
            top_p=0.9,
            seed=3407,
            num_ctx=32768,
            num_predict=2048,
        ),
        completion_status=CompletionStatus.COMPLETED,
    )


def test_review_can_stop_resume_and_preserves_completed_reviews(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    response_path = run_dir / "responses.jsonl"
    response_path.write_text(
        make_result("prompt-one").model_dump_json()
        + "\n"
        + make_result("prompt-two").model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    review_path = run_dir / "review.jsonl"

    first_answers = iter(["4", "clear notes", "good example", "minor issue", "q"])
    first_count = review_run(
        run_dir,
        reviewer="instructor",
        input_fn=lambda prompt: next(first_answers),
        output_fn=lambda message: None,
    )
    assert first_count == 1
    first_review_line = review_path.read_text(encoding="utf-8").splitlines()[0]
    first_review = HumanReview.model_validate_json(first_review_line)
    assert first_review.prompt_id == "prompt-one"
    assert first_review.dimensions[0].score == 4

    second_answers = iter(["N/A", "not applicable", "", ""])
    second_count = review_run(
        run_dir,
        reviewer="instructor",
        input_fn=lambda prompt: next(second_answers),
        output_fn=lambda message: None,
    )
    assert second_count == 1
    reviews = [
        HumanReview.model_validate_json(line)
        for line in review_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [review.prompt_id for review in reviews] == ["prompt-one", "prompt-two"]
    assert reviews[0].model_dump_json() == first_review_line
    assert reviews[1].dimensions[0].score == "N/A"

    assert (
        review_run(
            run_dir,
            reviewer="other-reviewer",
            input_fn=lambda prompt: (_ for _ in ()).throw(AssertionError("already reviewed")),
            output_fn=lambda message: None,
        )
        == 0
    )
    assert len(review_path.read_text(encoding="utf-8").splitlines()) == 2
