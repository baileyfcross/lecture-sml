import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lecture_slm.evaluation.prompts import EvaluationPrompt, load_evaluation_prompts


def valid_prompt() -> dict[str, object]:
    return {
        "id": "sample-eval",
        "task": "lecture",
        "title": "Sample evaluation",
        "instruction": "Create a short lesson.",
        "expected_characteristics": [
            {"description": "Uses a concrete example.", "dimension": "worked_examples"}
        ],
        "evaluation_dimensions": ["clarity", "worked_examples"],
        "tags": ["baseline", "sample"],
        "version": "1.0",
    }


def test_valid_prompt_reuses_dataset_task_enum() -> None:
    prompt = EvaluationPrompt.model_validate(valid_prompt())
    assert prompt.task.value == "lecture"
    assert prompt.expected_characteristics[0].dimension.value == "worked_examples"


def test_invalid_prompt_rejects_malformed_expected_characteristic() -> None:
    record = valid_prompt()
    record["expected_characteristics"] = ["a free-form string"]
    with pytest.raises(ValidationError):
        EvaluationPrompt.model_validate(record)


def test_duplicate_prompt_ids_are_rejected(tmp_path: Path) -> None:
    prompt = json.dumps(valid_prompt())
    path = tmp_path / "duplicate.jsonl"
    path.write_text(f"{prompt}\n{prompt}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate evaluation prompt id"):
        load_evaluation_prompts(path)


def test_unknown_dimensions_and_tags_are_rejected() -> None:
    record = valid_prompt()
    record["evaluation_dimensions"] = ["vague_quality"]
    with pytest.raises(ValidationError):
        EvaluationPrompt.model_validate(record)
