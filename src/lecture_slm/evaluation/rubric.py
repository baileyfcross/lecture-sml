"""Evaluation dimensions and rubric structures."""

from enum import StrEnum

from pydantic import BaseModel, Field


class EvaluationDimension(StrEnum):
    FACTUAL_ACCURACY = "factual_accuracy"
    INSTRUCTION_FOLLOWING = "instruction_following"
    COURSE_LEVEL = "course_level_appropriateness"
    PEDAGOGICAL_SEQUENCE = "pedagogical_sequencing"
    SCAFFOLDING = "scaffolding"
    OBJECTIVE_ALIGNMENT = "learning_objective_alignment"
    SLIDE_TEXT_DENSITY = "slide_text_density"
    WORKED_EXAMPLES = "worked_examples"
    STUDENT_PRACTICE = "student_practice_opportunities"
    MISCONCEPTION_HANDLING = "misconception_handling"
    PRIOR_KNOWLEDGE = "connection_to_prior_knowledge"
    SOURCE_GROUNDING = "source_grounding"
    INSTRUCTOR_STYLE = "instructor_voice_style"
    STRUCTURAL_COHERENCE = "overall_structural_coherence"


class RubricCriterion(BaseModel):
    """Description and scale for one independently scored dimension."""

    dimension: EvaluationDimension
    description: str = Field(min_length=1)
    weight: float = Field(default=1.0, ge=0.0)
    scale_min: int = 0
    scale_max: int = 4


class EvaluationRubric(BaseModel):
    """Versioned rubric; dimensions remain separate in stored results."""

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    criteria: list[RubricCriterion] = Field(min_length=1)
