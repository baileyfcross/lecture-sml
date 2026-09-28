"""Training and validation dataset schemas."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskType(StrEnum):
    LECTURE = "lecture"
    SLIDES = "slides"
    LAB = "lab"
    ACTIVITY = "activity"
    INSTRUCTOR_GUIDE = "instructor_guide"
    HOMEWORK = "homework"
    ASSESSMENT = "assessment"
    EXPLANATION = "explanation"


class DatasetSplit(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"
    EVALUATION = "evaluation"


class QualityTier(StrEnum):
    """Provenance-aware quality tier for a training example."""

    A = "A"  # Instructor-created and explicitly approved.
    B = "B"  # Instructor-created but not recently reviewed.
    C = "C"  # AI-generated and human-reviewed.
    D = "D"  # Synthetic and not yet human-reviewed.


class Provenance(BaseModel):
    """Origin and review history for a dataset example."""

    model_config = ConfigDict(extra="forbid")

    sources: list[str] = Field(default_factory=list)
    human_created: bool = False
    existing_material_type: str | None = None
    instructor_revision: bool = False
    synthetic: bool = False
    teacher_model_generated: bool = False
    human_reviewed: bool = False
    approved: bool = False
    quality_tier: QualityTier


class QualityMetadata(BaseModel):
    """Non-automated quality metadata captured during review."""

    reviewer: str | None = None
    notes: str | None = None
    factual_reviewed: bool = False
    pedagogy_reviewed: bool = False


class DatasetExample(BaseModel):
    """A structured, provenance-preserving example for future SFT data."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    split: DatasetSplit
    task: TaskType
    course: str = Field(min_length=1)
    level: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    context: str | None = None
    source_material: str | None = None
    response: str = Field(min_length=1)
    pedagogy_tags: list[str] = Field(default_factory=list)
    provenance: Provenance
    quality: QualityMetadata = Field(default_factory=QualityMetadata)
    version: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("pedagogy_tags")
    @classmethod
    def tags_must_be_normalized(cls, tags: list[str]) -> list[str]:
        if any(not tag.strip() or tag != tag.lower() for tag in tags):
            raise ValueError("pedagogy_tags must be non-empty lowercase strings")
        return tags
