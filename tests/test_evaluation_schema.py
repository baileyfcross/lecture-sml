from pathlib import Path

import pytest
from pydantic import ValidationError

from lecture_slm.evaluation.evaluator import (
    CompletionStatus,
    DimensionReview,
    EvaluationResult,
    EvaluationTiming,
    GenerationConfiguration,
    HumanReview,
    RunManifest,
)
from lecture_slm.evaluation.prompts import EvaluationPrompt, ExpectedCharacteristic
from lecture_slm.evaluation.rubric import (
    EvaluationDimension,
    EvaluationRubric,
    RubricCriterion,
    load_evaluation_rubric,
)
from lecture_slm.schemas.dataset import TaskType


def test_valid_evaluation_prompt() -> None:
    prompt = EvaluationPrompt(
        id="valid-prompt",
        task=TaskType.LECTURE,
        title="Introductory lecture",
        instruction="Create a lesson.",
        expected_characteristics=[ExpectedCharacteristic(description="Has objectives.")],
        evaluation_dimensions=[EvaluationDimension.CLARITY],
        tags=["baseline"],
        version="1.0",
    )
    assert prompt.task is TaskType.LECTURE


def test_invalid_evaluation_prompt_rejects_unknown_dimension() -> None:
    with pytest.raises(ValidationError):
        EvaluationPrompt.model_validate(
            {
                "id": "bad-prompt",
                "task": "lecture",
                "title": "Bad prompt",
                "instruction": "Do a task.",
                "expected_characteristics": [{"description": "Something."}],
                "evaluation_dimensions": ["vague_quality"],
                "tags": ["baseline"],
                "version": "1.0",
            }
        )


def test_rubric_preserves_individual_dimensions() -> None:
    rubric = EvaluationRubric(
        id="baseline",
        version="0.1",
        criteria=[
            RubricCriterion(
                dimension=EvaluationDimension.FACTUAL_ACCURACY,
                description="Claims are supported.",
            ),
            RubricCriterion(
                dimension=EvaluationDimension.SCAFFOLDING,
                description="Difficulty is scaffolded.",
            ),
        ],
    )
    assert len(rubric.criteria) == 2
    assert rubric.criteria[0].dimension is EvaluationDimension.FACTUAL_ACCURACY


def test_rubric_rejects_unknown_dimension() -> None:
    with pytest.raises(ValidationError):
        RubricCriterion(dimension="overall_vibe", description="Not a rubric dimension")


def test_versioned_rubric_contains_all_dimensions() -> None:
    root = Path(__file__).parents[1]
    rubric = load_evaluation_rubric(root / "evals/rubric.yaml")
    assert len(rubric.criteria) == 16
    assert rubric.score_anchors[1] == "Poor"
    assert rubric.score_anchors[5] == "Excellent"


def test_human_review_supports_not_applicable_score() -> None:
    review = HumanReview(
        prompt_id="prompt-1",
        reviewer="instructor",
        dimensions=[
            DimensionReview(
                dimension=EvaluationDimension.SLIDE_DENSITY,
                score="N/A",
                notes="This prompt did not request slides.",
            )
        ],
    )
    assert review.dimensions[0].score == "N/A"


def test_run_manifest_records_baseline_reproducibility_fields() -> None:
    manifest = RunManifest(
        run_id="baseline-test",
        model="qwen3.5:9b",
        server_fingerprint="0123456789abcdef",
        model_configuration={"inference": {"host": "[redacted]"}},
        generation_configuration=GenerationConfiguration(
            think=True,
            keep_alive="10m",
            temperature=0.5,
            top_p=0.9,
            seed=3407,
            num_ctx=32768,
            num_predict=2048,
        ),
        evaluation_dataset_version="1.0",
        evaluation_dataset_sha256="a" * 64,
        prompt_count=2,
        random_seed=3407,
        project_version="0.1.0",
    )
    assert manifest.generation_configuration.think is True
    assert manifest.git_commit is None


def test_evaluation_result_preserves_completed_output_and_timings() -> None:
    result = EvaluationResult(
        prompt_id="prompt-1",
        model="qwen3.5:9b",
        task=TaskType.LECTURE,
        instruction="Create an outline.",
        expected_characteristics=[ExpectedCharacteristic(description="Has objectives.")],
        evaluation_dimensions=[EvaluationDimension.CLARITY],
        response="Objectives, example, practice.",
        generation_configuration=GenerationConfiguration(
            think=True,
            keep_alive="10m",
            temperature=0.5,
            top_p=0.9,
            seed=3407,
            num_ctx=32768,
            num_predict=2048,
        ),
        timing=EvaluationTiming(total_duration_ns=1_000_000_000, total_duration_seconds=1.0),
        completion_status=CompletionStatus.COMPLETED,
    )
    assert result.timing.total_duration_ns == 1_000_000_000
    assert result.completion_status is CompletionStatus.COMPLETED


def test_failed_generation_result_is_preserved() -> None:
    result = EvaluationResult(
        prompt_id="prompt-failed",
        model="qwen3.5:9b",
        task=TaskType.EXPLANATION,
        instruction="Explain a concept.",
        expected_characteristics=[ExpectedCharacteristic(description="Uses an example.")],
        evaluation_dimensions=[EvaluationDimension.WORKED_EXAMPLES],
        generation_configuration=GenerationConfiguration(
            think=True,
            keep_alive="10m",
            temperature=0.5,
            top_p=0.9,
            seed=3407,
            num_ctx=32768,
            num_predict=2048,
        ),
        completion_status=CompletionStatus.FAILED,
        error_type="OllamaTimeoutError",
        error_message="Request timed out.",
    )
    assert result.response is None
    assert result.error_type == "OllamaTimeoutError"


def test_human_score_rejects_out_of_range_value() -> None:
    with pytest.raises(ValidationError):
        DimensionReview(dimension=EvaluationDimension.SCAFFOLDING, score=6)
