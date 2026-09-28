"""Evaluation result storage; automated judging is intentionally deferred."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from lecture_slm.evaluation.rubric import EvaluationDimension


class DimensionScore(BaseModel):
    """A human or future evaluator score for one rubric dimension."""

    dimension: EvaluationDimension
    score: int = Field(ge=0, le=4)
    notes: str | None = None


class EvaluationResult(BaseModel):
    """Stored output and optional dimension-level review for one prompt."""

    prompt_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)
    latency_seconds: float | None = Field(default=None, ge=0.0)
    configuration: dict[str, str | int | float | bool] = Field(default_factory=dict)
    scores: list[DimensionScore] = Field(default_factory=list)
