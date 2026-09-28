import pytest
from pydantic import ValidationError

from lecture_slm.evaluation.evaluator import DimensionScore, EvaluationResult
from lecture_slm.evaluation.rubric import (
    EvaluationDimension,
    EvaluationRubric,
    RubricCriterion,
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


def test_evaluation_result_accepts_dimension_scores() -> None:
    result = EvaluationResult(
        prompt_id="prompt-1",
        model_name="qwen3.5:9b",
        prompt="Create an outline.",
        response="Objectives, example, practice.",
        scores=[
            DimensionScore(
                dimension=EvaluationDimension.SCAFFOLDING,
                score=3,
                notes="Mostly gradual.",
            )
        ],
    )
    assert result.scores[0].score == 3


def test_evaluation_score_rejects_out_of_range_value() -> None:
    with pytest.raises(ValidationError):
        DimensionScore(dimension=EvaluationDimension.SCAFFOLDING, score=5)
