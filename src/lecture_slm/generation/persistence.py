"""Opt-in persistence for inspecting development generation runs."""

import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from lecture_slm.generation.models import GenerationRequest, GenerationResult


def create_generation_run_directory(root: Path, request_id: str) -> Path:
    """Create a non-overwriting artifact directory for an explicitly saved run."""

    safe_request_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", request_id).strip("-.") or "request"
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{safe_request_id}-{uuid4().hex[:6]}"
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def save_generation_run(
    directory: Path,
    request: GenerationRequest,
    result: GenerationResult,
) -> None:
    """Persist the explicit request, plan, complete result, and final markdown."""

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "request.json").write_text(
        request.model_dump_json(indent=2),
        encoding="utf-8",
    )
    planner_record = result.planner_result
    if planner_record is not None and planner_record.plan is not None:
        (directory / "plan.json").write_text(
            planner_record.plan.model_dump_json(indent=2),
            encoding="utf-8",
        )
    (directory / "result.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    if result.final_output is not None:
        (directory / "output.md").write_text(result.final_output, encoding="utf-8")
