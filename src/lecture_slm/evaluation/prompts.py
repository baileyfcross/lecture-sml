"""Typed evaluation-prompt records and JSONL loading."""

import json
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lecture_slm.evaluation.rubric import EvaluationDimension
from lecture_slm.schemas.dataset import TaskType


class ExpectedCharacteristic(BaseModel):
    """Observable output characteristic for human review, not a target answer."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    dimension: EvaluationDimension | None = None


class EvaluationPrompt(BaseModel):
    """One isolated generative evaluation prompt and its review criteria."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    task: TaskType
    title: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    course_profile: str | None = None
    pedagogy_profile: str | None = None
    context: str | None = None
    source_material: str | None = None
    expected_characteristics: list[ExpectedCharacteristic] = Field(min_length=1)
    evaluation_dimensions: list[EvaluationDimension] = Field(min_length=1)
    tags: list[str] = Field(min_length=1)
    version: str = Field(min_length=1)

    @field_validator("evaluation_dimensions")
    @classmethod
    def dimensions_must_be_unique(
        cls, dimensions: list[EvaluationDimension]
    ) -> list[EvaluationDimension]:
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("evaluation_dimensions must not contain duplicates")
        return dimensions

    @field_validator("tags")
    @classmethod
    def tags_must_be_normalized(cls, tags: list[str]) -> list[str]:
        if any(not tag or tag != tag.lower().strip() for tag in tags):
            raise ValueError("tags must be non-empty lowercase strings without surrounding spaces")
        if len(tags) != len(set(tags)):
            raise ValueError("tags must not contain duplicates")
        return tags

    @model_validator(mode="after")
    def expected_dimensions_are_applicable(self) -> Self:
        dimensions = set(self.evaluation_dimensions)
        for characteristic in self.expected_characteristics:
            if characteristic.dimension is not None and characteristic.dimension not in dimensions:
                raise ValueError(
                    "expected-characteristic dimensions must appear in evaluation_dimensions"
                )
        return self


def load_evaluation_prompts(path: Path) -> list[EvaluationPrompt]:
    """Load and validate JSONL prompts, including uniqueness of prompt IDs."""

    prompts: list[EvaluationPrompt] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw: Any = json.loads(line)
                prompt = EvaluationPrompt.model_validate(raw)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"Invalid evaluation prompt on line {line_number}: {error}"
                ) from error
            if prompt.id in seen_ids:
                raise ValueError(f"Duplicate evaluation prompt id '{prompt.id}'")
            seen_ids.add(prompt.id)
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No evaluation prompts found in {path}")
    versions = {prompt.version for prompt in prompts}
    if len(versions) != 1:
        raise ValueError("All prompts in an evaluation dataset must use the same version")
    return prompts
