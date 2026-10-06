"""Opt-in persistence for inspecting development generation runs."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import TypeAdapter

from lecture_slm.generation.models import (
    GenerationRequest,
    GenerationResult,
    GroundingReviewRecord,
    StageRecord,
)
from lecture_slm.generation.prompts.grounding import (
    build_grounding_review_prompt,
    build_grounding_revision_prompt,
    grounding_source_ref_map,
)
from lecture_slm.generation.prompts.planner import (
    STANDARD_PLANNER_PROMPT_VERSION,
    build_planner_prompt,
)
from lecture_slm.generation.prompts.writer import build_writer_prompt


def create_generation_run_directory(root: Path, request_id: str) -> Path:
    """Create a non-overwriting artifact directory for an explicitly saved run."""

    safe_request_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", request_id).strip("-.") or "request"
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{safe_request_id}-{uuid4().hex[:6]}"
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _write_json(path: Path, value: Any) -> None:
    serialized = TypeAdapter(Any).dump_python(value, mode="json")
    path.write_text(
        json.dumps(serialized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _source_markdown(request: GenerationRequest) -> str:
    if not request.source_material:
        return "# Assembled Source Material\n\nNo source material was supplied.\n"

    sections = ["# Assembled Source Material"]
    for index, source in enumerate(request.source_material, start=1):
        sections.extend(
            [
                "",
                f"## Source {index}: {source.title}",
                f"- **Title:** {source.title}",
                f"- **Source ID:** {source.source_id}",
            ]
        )
        if source.section is not None:
            sections.append(f"- **Section:** {source.section}")
        sections.extend(
            [
                "- **Relevant metadata:**",
                "```json",
                json.dumps(source.metadata, ensure_ascii=False, indent=2),
                "```",
                "",
                "### Exact text supplied to generation",
            ]
        )
        longest_backtick_run = max(
            (len(run) for run in re.findall(r"`+", source.text)),
            default=0,
        )
        fence = "`" * max(3, longest_backtick_run + 1)
        sections.extend([fence, source.text, fence])
    return "\n".join(sections) + "\n"


def _stage_diagnostic(name: str, record: StageRecord | None, *, used: bool) -> list[str]:
    lines = [f"## {name}", f"- Used: {'yes' if used else 'no'}"]
    timing = None if record is None else record.timing
    values = [
        ("Context", None if timing is None else timing.selected_context),
        ("Estimated input tokens", None if timing is None else timing.estimated_input_tokens),
        ("Output budget", None if timing is None else timing.output_budget),
        ("Generated tokens", None if timing is None else timing.generated_tokens),
        ("Duration", None if timing is None else timing.duration_seconds),
        ("Tokens/sec", None if timing is None else timing.tokens_per_second),
        ("Stop reason", None if timing is None else timing.stop_reason),
        ("Output limit reached", None if timing is None else timing.output_limit_reached),
    ]
    lines.extend(f"- {label}: {value if value is not None else 'n/a'}" for label, value in values)
    return lines


def _grounding_diagnostic(name: str, record: GroundingReviewRecord | None) -> list[str]:
    lines = [f"## {name}", f"- Used: {'yes' if record is not None else 'no'}"]
    if record is None:
        return lines
    lines.extend(
        [
            f"- Status: {record.status.value}",
            f"- Decision: {record.review.decision.value if record.review is not None else 'n/a'}",
            f"- Prompt version: {record.prompt_version or 'n/a'}",
            f"- Duration: {record.timing.duration_seconds if record.timing is not None else 'n/a'}",
        ]
    )
    if record.review is not None:
        lines.append(f"- Flagged claims: {len(record.review.issues)}")
        lines.extend(
            f"  - [{issue.category.value}] {issue.excerpt}: {issue.reason}"
            for issue in record.review.issues
        )
    if record.error_message is not None:
        lines.append(f"- Error: {record.error_type}: {record.error_message}")
    return lines


def _write_diagnostics(
    directory: Path,
    request: GenerationRequest,
    result: GenerationResult,
    retrieval: Any,
) -> None:
    retrieval_mapping = retrieval if isinstance(retrieval, dict) else {}
    matches = retrieval_mapping.get("matches")
    warnings = retrieval_mapping.get("warnings")
    planner_record = result.planner_result
    writer_record = result.writer_result
    planner_used = planner_record is not None and planner_record.prompt_version is not None
    writer_used = (
        writer_record is not None
        and writer_record.timing is not None
        and writer_record.timing.selected_context is not None
    )
    lines = [
        "# Generation Diagnostics",
        "",
        "## Request",
        f"- Request ID: {request.request_id}",
        f"- Task: {request.task.value}",
        f"- Profile: {request.profile.value}",
        f"- Model: {result.model}",
        f"- Status: {result.status.value}",
        "",
        "## Retrieval",
        f"- Retrieval enabled: {'yes' if retrieval is not None else 'no'}",
        f"- Number of retrieved matches: {len(matches) if isinstance(matches, list) else 0}",
        f"- Number of assembled source materials: {len(request.source_material)}",
        "- Retrieval warnings: "
        + (
            "; ".join(str(warning) for warning in warnings)
            if isinstance(warnings, list) and warnings
            else "none"
        ),
        "",
        *_stage_diagnostic("Planner", planner_record, used=planner_used),
        "",
        *_stage_diagnostic("Writer", writer_record, used=writer_used),
        "",
        *_grounding_diagnostic("Initial grounding review", result.initial_grounding_review),
        "",
        *_stage_diagnostic(
            "Grounding revision",
            result.revision_result,
            used=result.revision_result is not None,
        ),
        "",
        *_grounding_diagnostic("Final grounding review", result.final_grounding_review),
        "",
        "## Files",
    ]
    lines.extend(f"- {path.name}" for path in sorted(directory.iterdir()) if path.is_file())
    (directory / "diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_generation_run(
    directory: Path,
    request: GenerationRequest,
    result: GenerationResult,
) -> None:
    """Persist the explicit request, prompt snapshots, diagnostics, and final output."""

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "request.json").write_text(
        request.model_dump_json(indent=2),
        encoding="utf-8",
    )

    retrieval = request.metadata.get("knowledge_retrieval")
    if retrieval is not None:
        _write_json(directory / "retrieval.json", retrieval)

    _write_json(
        directory / "sources.json",
        [source.model_dump(mode="json") for source in request.source_material],
    )
    (directory / "sources.md").write_text(_source_markdown(request), encoding="utf-8")

    planner_record = result.planner_result
    if planner_record is not None and planner_record.plan is not None:
        (directory / "plan.json").write_text(
            planner_record.plan.model_dump_json(indent=2),
            encoding="utf-8",
        )
    if planner_record is not None and planner_record.prompt_version is not None:
        planner_prompt = build_planner_prompt(
            request,
            concise=planner_record.prompt_version == STANDARD_PLANNER_PROMPT_VERSION,
        )
        timing = planner_record.timing
        _write_json(
            directory / "planner_prompt.json",
            {
                "prompt_version": planner_prompt.version,
                "system_message": planner_prompt.system_message,
                "user_message": planner_prompt.user_message,
                "selected_context": None if timing is None else timing.selected_context,
                "estimated_input_tokens": (
                    None if timing is None else timing.estimated_input_tokens
                ),
                "output_budget": None if timing is None else timing.output_budget,
            },
        )

    writer_record = result.writer_result
    if (
        writer_record is not None
        and writer_record.timing is not None
        and writer_record.timing.selected_context is not None
    ):
        writer_prompt = build_writer_prompt(
            request,
            None if planner_record is None else planner_record.plan,
        )
        timing = writer_record.timing
        _write_json(
            directory / "writer_prompt.json",
            {
                "prompt_version": writer_prompt.version,
                "system_message": writer_prompt.system_message,
                "user_message": writer_prompt.user_message,
                "selected_context": timing.selected_context,
                "estimated_input_tokens": timing.estimated_input_tokens,
                "output_budget": timing.output_budget,
            },
        )
    if (
        result.initial_grounding_review is not None
        and writer_record is not None
        and writer_record.raw_response is not None
    ):
        (directory / "writer_output.md").write_text(writer_record.raw_response, encoding="utf-8")

    initial_review = result.initial_grounding_review
    if initial_review is not None:
        _write_json(directory / "grounding_review_initial.json", initial_review)
        review_prompt = build_grounding_review_prompt(
            request,
            (writer_record.raw_response if writer_record is not None else None) or "",
        )
        timing = initial_review.timing
        _write_json(
            directory / "grounding_review_initial_prompt.json",
            {
                "prompt_version": review_prompt.version,
                "system_message": review_prompt.system_message,
                "user_message": review_prompt.user_message,
                "source_ref_map": grounding_source_ref_map(request),
                "selected_context": None if timing is None else timing.selected_context,
                "estimated_input_tokens": (
                    None if timing is None else timing.estimated_input_tokens
                ),
                "output_budget": None if timing is None else timing.output_budget,
            },
        )

    revision_record = result.revision_result
    if revision_record is not None:
        _write_json(directory / "grounding_revision.json", revision_record)
        if revision_record.raw_response is not None:
            (directory / "revised_output.md").write_text(
                revision_record.raw_response,
                encoding="utf-8",
            )
        if initial_review is not None and initial_review.review is not None:
            revision_prompt = build_grounding_revision_prompt(
                request,
                (writer_record.raw_response if writer_record is not None else None) or "",
                initial_review.review,
            )
            timing = revision_record.timing
            _write_json(
                directory / "grounding_revision_prompt.json",
                {
                    "prompt_version": revision_prompt.version,
                    "system_message": revision_prompt.system_message,
                    "user_message": revision_prompt.user_message,
                    "source_ref_map": grounding_source_ref_map(request),
                    "selected_context": None if timing is None else timing.selected_context,
                    "estimated_input_tokens": (
                        None if timing is None else timing.estimated_input_tokens
                    ),
                    "output_budget": None if timing is None else timing.output_budget,
                },
            )

    final_review = result.final_grounding_review
    if final_review is not None:
        _write_json(directory / "grounding_review_final.json", final_review)
        final_candidate = (
            revision_record.raw_response
            if revision_record is not None
            else writer_record.raw_response
            if writer_record is not None
            else ""
        )
        review_prompt = build_grounding_review_prompt(request, final_candidate or "")
        timing = final_review.timing
        _write_json(
            directory / "grounding_review_final_prompt.json",
            {
                "prompt_version": review_prompt.version,
                "system_message": review_prompt.system_message,
                "user_message": review_prompt.user_message,
                "source_ref_map": grounding_source_ref_map(request),
                "selected_context": None if timing is None else timing.selected_context,
                "estimated_input_tokens": (
                    None if timing is None else timing.estimated_input_tokens
                ),
                "output_budget": None if timing is None else timing.output_budget,
            },
        )

    (directory / "result.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    if result.final_output is not None:
        (directory / "output.md").write_text(result.final_output, encoding="utf-8")

    _write_diagnostics(directory, request, result, retrieval)
