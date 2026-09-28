import pytest
from pydantic import ValidationError

from lecture_slm.schemas.dataset import DatasetExample, DatasetSplit, QualityTier, TaskType


def valid_record() -> dict[str, object]:
    return {
        "id": "example-1",
        "split": "training",
        "task": "lecture",
        "course": "example-course",
        "level": "freshman",
        "instruction": "Create an outline.",
        "response": "An outline.",
        "pedagogy_tags": ["scaffolding", "worked_examples"],
        "provenance": {"quality_tier": "A", "human_created": True, "approved": True},
        "version": "0.1",
    }


def test_valid_dataset_record() -> None:
    example = DatasetExample.model_validate(valid_record())
    assert example.split is DatasetSplit.TRAINING
    assert example.task is TaskType.LECTURE


def test_invalid_quality_tier() -> None:
    record = valid_record()
    record["provenance"] = {"quality_tier": "E"}
    with pytest.raises(ValidationError, match="quality_tier"):
        DatasetExample.model_validate(record)


def test_invalid_dataset_record() -> None:
    record = valid_record()
    del record["response"]
    with pytest.raises(ValidationError, match="response"):
        DatasetExample.model_validate(record)


def test_quality_tier_enum() -> None:
    assert {tier.value for tier in QualityTier} == {"A", "B", "C", "D"}


@pytest.mark.parametrize("split", list(DatasetSplit))
def test_dataset_split_handling(split: DatasetSplit) -> None:
    record = valid_record()
    record["split"] = split.value
    assert DatasetExample.model_validate(record).split is split
