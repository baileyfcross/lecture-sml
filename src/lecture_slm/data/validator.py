"""Validation helpers for JSON and JSONL dataset files."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from lecture_slm.schemas.dataset import DatasetExample


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array/object or one JSON object per line."""

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    raise ValueError("JSON dataset must contain an object or an array of objects")


def validate_file(path: Path) -> list[DatasetExample]:
    """Validate every record and raise a combined error with record indexes."""

    validated: list[DatasetExample] = []
    errors: list[str] = []
    for index, record in enumerate(load_records(path), start=1):
        try:
            validated.append(DatasetExample.model_validate(record))
        except ValidationError as error:
            errors.append(f"record {index}: {error}")
    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errors))
    return validated
