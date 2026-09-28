"""Evaluation dimensions and rubric structures."""

from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationDimension(StrEnum):
    FACTUAL_ACCURACY = "factual_accuracy"
    SOURCE_GROUNDING = "source_grounding"
    INSTRUCTION_FOLLOWING = "instruction_following"
    COURSE_LEVEL_APPROPRIATENESS = "course_level_appropriateness"
    PREREQUISITE_AWARENESS = "prerequisite_awareness"
    PEDAGOGICAL_SEQUENCE = "pedagogical_sequence"
    SCAFFOLDING = "scaffolding"
    CONNECTION_TO_PRIOR_KNOWLEDGE = "connection_to_prior_knowledge"
    WORKED_EXAMPLES = "worked_examples"
    PRACTICE_OPPORTUNITIES = "practice_opportunities"
    MISCONCEPTION_HANDLING = "misconception_handling"
    LEARNING_OBJECTIVE_ALIGNMENT = "learning_objective_alignment"
    SLIDE_DENSITY = "slide_density"
    CLARITY = "clarity"
    STRUCTURAL_COHERENCE = "structural_coherence"
    INSTRUCTOR_STYLE = "instructor_style"


class RubricCriterion(BaseModel):
    """Description and scale for one independently scored dimension."""

    model_config = ConfigDict(extra="forbid")

    dimension: EvaluationDimension
    description: str = Field(min_length=1)
    weight: float = Field(default=1.0, ge=0.0)
    scale_min: int = Field(default=1, ge=1, le=5)
    scale_max: int = Field(default=5, ge=1, le=5)

    @model_validator(mode="after")
    def scale_is_ordered(self) -> "RubricCriterion":
        if self.scale_min >= self.scale_max:
            raise ValueError("scale_min must be less than scale_max")
        return self


class EvaluationRubric(BaseModel):
    """Versioned rubric; dimensions remain separate in stored results."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    criteria: list[RubricCriterion] = Field(min_length=1)
    score_anchors: dict[int, str] = Field(
        default_factory=lambda: {
            1: "Poor",
            2: "Weak",
            3: "Acceptable",
            4: "Good",
            5: "Excellent",
        }
    )

    @model_validator(mode="after")
    def dimensions_are_unique(self) -> Self:
        dimensions = [criterion.dimension for criterion in self.criteria]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("rubric dimensions must be unique")
        if set(self.score_anchors) != {1, 2, 3, 4, 5}:
            raise ValueError("score_anchors must define labels for scores 1 through 5")
        return self


def load_evaluation_rubric(path: Path) -> EvaluationRubric:
    """Load and validate a rubric from YAML."""

    if not path.is_file():
        raise FileNotFoundError(f"Evaluation rubric not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            data: Any = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid rubric YAML in {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return EvaluationRubric.model_validate(data)
