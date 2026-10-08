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
    GroundingClaimClassification,
    GroundingReviewRecord,
    StageRecord,
)
from lecture_slm.generation.prompts.base import PromptPackage
from lecture_slm.generation.prompts.grounding import (
    build_grounding_review_prompt,
    build_grounding_revision_prompt,
    grounding_source_ref_map,
)
from lecture_slm.generation.prompts.planner import (
    PLANNER_REASSESSMENT_PROMPT_VERSION,
    STANDARD_PLANNER_PROMPT_VERSION,
    build_planner_prompt,
    build_planner_reassessment_prompt,
)
from lecture_slm.generation.prompts.writer import build_writer_prompt
from lecture_slm.generation.reviewer import prepare_grounding_claims


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


def _planner_prompt_snapshot(
    prompt: PromptPackage,
    record: StageRecord,
    *,
    temperature: float | None,
) -> dict[str, Any]:
    timing = record.timing
    return {
        "prompt_version": prompt.version,
        "temperature": temperature,
        "system_message": prompt.system_message,
        "user_message": prompt.user_message,
        "selected_context": None if timing is None else timing.selected_context,
        "estimated_input_tokens": (None if timing is None else timing.estimated_input_tokens),
        "output_budget": None if timing is None else timing.output_budget,
        "thinking_reserve_tokens": (None if timing is None else timing.thinking_reserve_tokens),
        "generation_budget": None if timing is None else timing.generation_budget,
        "thinking_enabled": None if timing is None else timing.thinking_enabled,
        "duration_seconds": None if timing is None else timing.duration_seconds,
        "generated_tokens": None if timing is None else timing.generated_tokens,
        "stop_reason": None if timing is None else timing.stop_reason,
        "output_limit_reached": None if timing is None else timing.output_limit_reached,
    }


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
    values: list[tuple[str, object | None]] = [
        ("Prompt version", None if record is None else record.prompt_version),
        ("Status", None if record is None else record.status.value),
        ("Context", None if timing is None else timing.selected_context),
        ("Estimated input tokens", None if timing is None else timing.estimated_input_tokens),
        ("Thinking enabled", None if timing is None else timing.thinking_enabled),
        ("Structured output budget", None if timing is None else timing.output_budget),
        ("Thinking reserve", None if timing is None else timing.thinking_reserve_tokens),
        ("Total generation budget", None if timing is None else timing.generation_budget),
        ("Generated tokens", None if timing is None else timing.generated_tokens),
        ("Thinking characters", None if timing is None else timing.thinking_characters),
    ]
    values.extend(
        [
            ("Duration", None if timing is None else timing.duration_seconds),
            ("Tokens/sec", None if timing is None else timing.tokens_per_second),
            ("Stop reason", None if timing is None else timing.stop_reason),
            ("Output limit reached", None if timing is None else timing.output_limit_reached),
            ("Error type", None if record is None else record.error_type),
            ("Error", None if record is None else record.error_message),
        ]
    )
    lines.extend(f"- {label}: {value if value is not None else 'n/a'}" for label, value in values)
    return lines


def _grounding_diagnostic(name: str, record: GroundingReviewRecord | None) -> list[str]:
    lines = [f"## {name}", f"- Used: {'yes' if record is not None else 'no'}"]
    if record is None:
        return lines
    timing = record.timing
    claims_extracted = record.claims_extracted
    direct_count = record.direct_supported_claim_count
    unresolved_count = record.unresolved_claim_count
    output_budget = record.review_output_budget
    selected_context = None if timing is None else timing.selected_context
    generated_tokens = None if timing is None else timing.generated_tokens
    stop_reason = None if timing is None else timing.stop_reason
    output_limit_reached = None if timing is None else timing.output_limit_reached
    lines.extend(
        [
            f"- Status: {record.status.value}",
            f"- Decision: {record.review.decision.value if record.review is not None else 'n/a'}",
            f"- Prompt version: {record.prompt_version or 'n/a'}",
            f"- Duration: {timing.duration_seconds if timing is not None else 'n/a'}",
            f"- Claims extracted: {claims_extracted if claims_extracted is not None else 'n/a'}",
            f"- Direct-supported claims: {direct_count if direct_count is not None else 'n/a'}",
            f"- Unresolved reviewer claims: "
            f"{unresolved_count if unresolved_count is not None else 'n/a'}",
            f"- Review output budget: {output_budget if output_budget is not None else 'n/a'}",
            f"- Selected context: {selected_context if selected_context is not None else 'n/a'}",
            f"- Generated tokens: {generated_tokens if generated_tokens is not None else 'n/a'}",
            f"- Stop reason: {stop_reason if stop_reason is not None else 'n/a'}",
            (
                "- Output limit reached: "
                f"{output_limit_reached if output_limit_reached is not None else 'n/a'}"
            ),
        ]
    )
    if record.review is not None:
        assessments = record.review.claim_assessments
        counts = {
            classification: sum(item.classification is classification for item in assessments)
            for classification in GroundingClaimClassification
        }
        evidence_failures = record.review.evidence_validation_failures
        lines.extend(
            [
                f"- Evidence ledger spans: {len(record.review.evidence_ledger)}",
                f"- Reviewer-supported claims: {counts[GroundingClaimClassification.SUPPORTED]}",
                (
                    "- Continuity-supported claims: "
                    f"{counts[GroundingClaimClassification.CONTINUITY_SUPPORTED]}"
                ),
                f"- Pedagogical claims: {counts[GroundingClaimClassification.PEDAGOGICAL]}",
                f"- Unsupported claims: {counts[GroundingClaimClassification.UNSUPPORTED]}",
                (f"- Evidence validation failures / unknown evidence IDs: {evidence_failures}"),
                (
                    "- Continuity validation failures / not-reviewable-or-invalid IDs: "
                    f"{record.review.continuity_validation_failures}"
                ),
                f"- Coverage complete: {'yes' if record.review.coverage_complete else 'no'}",
                f"- Decision: {record.review.decision.value}",
            ]
        )
        continuity_selection = record.review.continuity_selection
        if isinstance(continuity_selection, dict):
            lines.extend(
                [
                    "- Continuity eligible claims: "
                    f"{continuity_selection.get('eligible_claim_count', 'n/a')}",
                    "- Continuity gate-eligible claim IDs: "
                    + ", ".join(
                        str(item)
                        for item in continuity_selection.get("gate_eligible_claim_ids", [])
                    ),
                    "- Continuity reviewable claim IDs: "
                    + ", ".join(
                        str(item) for item in continuity_selection.get("reviewable_claim_ids", [])
                    ),
                    "- Continuity available items: "
                    f"{continuity_selection.get('available_item_count', 'n/a')}",
                    "- Continuity available spans: "
                    f"{continuity_selection.get('available_span_count', 'n/a')}",
                    "- Continuity selected spans: "
                    f"{continuity_selection.get('selected_span_count', 'n/a')}",
                    "- Continuity selected spans by claim: "
                    + json.dumps(
                        continuity_selection.get("selected_by_claim", {}),
                        ensure_ascii=False,
                    ),
                    "- Continuity selected candidate details: "
                    + json.dumps(
                        continuity_selection.get("selected_candidate_details", []),
                        ensure_ascii=False,
                    ),
                    "- Continuity rejected structural-only candidates: "
                    f"{continuity_selection.get('rejected_structural_only_candidates', 0)}",
                    "- Continuity cap per claim: "
                    f"{continuity_selection.get('max_spans_per_claim', 'n/a')}",
                    f"- Continuity cap total: {continuity_selection.get('max_total_spans', 'n/a')}",
                ]
            )
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
    retrieval_query_lines: list[str] = []
    if retrieval_mapping:
        original_query = retrieval_mapping.get("original_query", request.instruction)
        canonical_query = retrieval_mapping.get(
            "canonical_query",
            retrieval_mapping.get("query", "n/a"),
        )
        retrieval_query_lines = [
            f"- Original instruction: {original_query}",
            f"- Canonical query: {canonical_query}",
        ]
    retrieval_diagnostics = retrieval_mapping.get("diagnostics", {})
    retrieval_rounds = (
        retrieval_diagnostics.get("retrieval_rounds", [])
        if isinstance(retrieval_diagnostics, dict)
        else []
    )
    retrieval_round_lines = [
        f"- Retrieval rounds: {len(retrieval_rounds) if isinstance(retrieval_rounds, list) else 0}"
    ]
    if isinstance(retrieval_diagnostics, dict):
        retrieval_round_lines.append(
            f"- Retrieval expanded: "
            f"{'yes' if retrieval_diagnostics.get('retrieval_expanded') else 'no'}"
        )
        if retrieval_diagnostics.get("expansion_retrieval_seconds") is not None:
            retrieval_round_lines.append(
                "- Expansion retrieval duration: "
                f"{retrieval_diagnostics['expansion_retrieval_seconds']} seconds"
            )
        for label, key in (
            ("Initial planning duration", "initial_planning_seconds"),
            ("Final planning duration", "final_planning_seconds"),
        ):
            if retrieval_diagnostics.get(key) is not None:
                retrieval_round_lines.append(f"- {label}: {retrieval_diagnostics[key]} seconds")
        assessments = retrieval_diagnostics.get("source_scope_assessments")
        if isinstance(assessments, dict):
            scope_labels = (
                ("Initial source scope", "initial"),
                ("Final source scope", "final"),
            )
            for label, key in scope_labels:
                assessment = assessments.get(key)
                if isinstance(assessment, dict):
                    retrieval_round_lines.append(f"- {label}: {assessment.get('status', 'n/a')}")
                    for topic_label, topic_key in (
                        ("supported", "supported_topics"),
                        ("unsupported", "unsupported_requested_topics"),
                    ):
                        topics = assessment.get(topic_key)
                        if isinstance(topics, list):
                            topic_text = ", ".join(str(topic) for topic in topics) or "none"
                            retrieval_round_lines.append(
                                f"  - {topic_label.title()} topics: {topic_text}"
                            )
        expansion_plan = retrieval_diagnostics.get("expansion_plan")
        if isinstance(expansion_plan, dict):
            queries = expansion_plan.get("expansion_queries")
            if isinstance(queries, list):
                retrieval_round_lines.append(
                    "- Expansion queries: " + (", ".join(str(query) for query in queries) or "none")
                )
        expansion_round = (
            next(
                (
                    item
                    for item in retrieval_rounds
                    if isinstance(item, dict) and item.get("type") == "expansion"
                ),
                None,
            )
            if isinstance(retrieval_rounds, list)
            else None
        )
        if isinstance(expansion_round, dict):
            retrieval_round_lines.extend(
                [
                    f"- Unique chunks added: {expansion_round.get('unique_added_count', 0)}",
                    f"- Merged retrieved matches: {expansion_round.get('merged_match_count', 0)}",
                ]
            )
    if isinstance(retrieval_rounds, list):
        for round_diagnostic in retrieval_rounds:
            if isinstance(round_diagnostic, dict):
                retrieval_round_lines.append(
                    f"- Round {round_diagnostic.get('round', 'n/a')} "
                    f"({round_diagnostic.get('type', 'unknown')}): "
                    f"{round_diagnostic.get('retrieved_count', 0)} matches in "
                    f"{round_diagnostic.get('duration_seconds', 0)} seconds"
                )
    source_scope_lines = ["## Source scope"]
    source_scope = (
        None
        if planner_record is None or planner_record.plan is None
        else planner_record.plan.source_scope
    )
    if source_scope is None:
        source_scope_lines.append("- Assessment: not available")
    else:
        source_scope_lines.append(f"- Status: {source_scope.status.value}")
        source_scope_lines.append("- Supported topics:")
        source_scope_lines.extend(f"  - {topic}" for topic in source_scope.supported_topics)
        if not source_scope.supported_topics:
            source_scope_lines.append("  - none identified")
        source_scope_lines.append("- Unsupported requested topics:")
        source_scope_lines.extend(
            f"  - {topic}" for topic in source_scope.unsupported_requested_topics
        )
        if not source_scope.unsupported_requested_topics:
            source_scope_lines.append("  - none identified")
        if source_scope.scope_note:
            source_scope_lines.append(f"- Scope note: {source_scope.scope_note}")
    if isinstance(retrieval_diagnostics, dict):
        diagnostics = retrieval_diagnostics.get("diagnostics")
        normalization = (
            diagnostics.get("source_scope_normalization") if isinstance(diagnostics, dict) else None
        )
        if isinstance(normalization, dict):
            raw = normalization.get("raw_unsupported_requested_topics", [])
            normalized = normalization.get("normalized_unsupported_requested_topics", [])
            removed = normalization.get("removed_unsupported_topics", [])
            source_scope_lines.append("- Raw unsupported requested topics:")
            if isinstance(raw, list) and raw:
                source_scope_lines.extend(f"  - {topic}" for topic in raw)
            else:
                source_scope_lines.append("  - none identified")
            source_scope_lines.append("- Normalized unsupported requested topics:")
            if isinstance(normalized, list) and normalized:
                source_scope_lines.extend(f"  - {topic}" for topic in normalized)
            else:
                source_scope_lines.append("  - none identified")
            source_scope_lines.append("- Removed unsupported topics:")
            if isinstance(removed, list) and removed:
                source_scope_lines.extend(
                    f"  - {entry.get('topic', 'n/a')}: {entry.get('reason', 'n/a')}"
                    for entry in removed
                    if isinstance(entry, dict)
                )
            else:
                source_scope_lines.append("  - none identified")
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
        *retrieval_query_lines,
        *retrieval_round_lines,
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
        *_stage_diagnostic(
            "Planner initial",
            result.planner_initial_result,
            used=result.planner_initial_result is not None,
        ),
        "",
        *_stage_diagnostic(
            "Planner reassessment",
            result.planner_reassessment_result,
            used=result.planner_reassessment_result is not None,
        ),
        "",
        *source_scope_lines,
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

    workspace_diagnostics = request.metadata.get("workspace")
    if request.workspace_context is not None and workspace_diagnostics is not None:
        workspace_references = [
            source.model_dump(mode="json")
            for source in request.source_material
            if source.metadata.get("source_origin") == "workspace_reference"
        ]
        _write_json(
            directory / "workspace_context.json",
            {
                "workspace_id": request.workspace_id,
                "continuity": request.workspace_context.model_dump(mode="json"),
                "selection": workspace_diagnostics,
                "references": workspace_references,
            },
        )

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
        initial_record = result.planner_initial_result
        reassessment_record = result.planner_reassessment_result
        initial_request = result.planner_initial_request or request
        if initial_record is not None and initial_record.prompt_version is not None:
            initial_prompt = build_planner_prompt(
                initial_request,
                concise=initial_record.prompt_version == STANDARD_PLANNER_PROMPT_VERSION,
            )
            if reassessment_record is not None:
                configured_planner = result.profile_configuration.get("planner", {})
                _write_json(
                    directory / "planner_initial_prompt.json",
                    _planner_prompt_snapshot(
                        initial_prompt,
                        initial_record,
                        temperature=(
                            configured_planner.get("temperature")
                            if isinstance(configured_planner, dict)
                            else None
                        ),
                    ),
                )
                if initial_record.plan is not None:
                    (directory / "planner_initial_plan.json").write_text(
                        initial_record.plan.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
                if initial_record.raw_response is not None:
                    (directory / "planner_initial_raw_response.txt").write_text(
                        initial_record.raw_response,
                        encoding="utf-8",
                    )

        if (
            reassessment_record is not None
            and initial_record is not None
            and initial_record.plan is not None
        ):
            reassessment_prompt = build_planner_reassessment_prompt(
                request,
                initial_record.plan,
            )
            _write_json(
                directory / "planner_reassessment_prompt.json",
                _planner_prompt_snapshot(
                    reassessment_prompt,
                    reassessment_record,
                    temperature=0,
                ),
            )
            if reassessment_record.plan is not None:
                (directory / "planner_reassessment_plan.json").write_text(
                    reassessment_record.plan.model_dump_json(indent=2),
                    encoding="utf-8",
                )
            if reassessment_record.raw_response is not None:
                (directory / "planner_reassessment_raw_response.txt").write_text(
                    reassessment_record.raw_response,
                    encoding="utf-8",
                )
            _write_json(
                directory / "planner_reassessment_error.json",
                {
                    "status": reassessment_record.status.value,
                    "prompt_version": reassessment_record.prompt_version,
                    "error_type": reassessment_record.error_type,
                    "error_message": reassessment_record.error_message,
                    "timing": (
                        None
                        if reassessment_record.timing is None
                        else reassessment_record.timing.model_dump(mode="json")
                    ),
                },
            )

            effective_prompt = reassessment_prompt
            effective_record = reassessment_record
        else:
            effective_prompt = (
                initial_prompt
                if initial_record is not None
                else build_planner_prompt(
                    request,
                    concise=planner_record.prompt_version == STANDARD_PLANNER_PROMPT_VERSION,
                )
            )
            effective_record = planner_record
        _write_json(
            directory / "planner_prompt.json",
            _planner_prompt_snapshot(
                effective_prompt,
                effective_record,
                temperature=(
                    0
                    if effective_prompt.version == PLANNER_REASSESSMENT_PROMPT_VERSION
                    else (
                        result.profile_configuration.get("planner", {}).get("temperature")
                        if isinstance(result.profile_configuration.get("planner"), dict)
                        else None
                    )
                ),
            ),
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
        initial_candidate = (
            writer_record.raw_response if writer_record is not None else None
        ) or ""
        _, _, unresolved_claims = prepare_grounding_claims(request, initial_candidate)
        review_prompt = build_grounding_review_prompt(request, unresolved_claims)
        _write_json(directory / "grounding_review.json", initial_review)
        review_timing = initial_review.timing
        _write_json(
            directory / "grounding_review_initial_prompt.json",
            {
                "prompt_version": review_prompt.version,
                "system_message": review_prompt.system_message,
                "user_message": review_prompt.user_message,
                "source_ref_map": grounding_source_ref_map(request),
                "selected_context": (
                    None if review_timing is None else review_timing.selected_context
                ),
                "estimated_input_tokens": (
                    None if review_timing is None else review_timing.estimated_input_tokens
                ),
                "output_budget": initial_review.review_output_budget,
                "claims_extracted": initial_review.claims_extracted,
                "direct_supported_claim_count": initial_review.direct_supported_claim_count,
                "unresolved_claim_count": initial_review.unresolved_claim_count,
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
            revision_timing = revision_record.timing
            _write_json(
                directory / "grounding_revision_prompt.json",
                {
                    "prompt_version": revision_prompt.version,
                    "system_message": revision_prompt.system_message,
                    "user_message": revision_prompt.user_message,
                    "source_ref_map": grounding_source_ref_map(request),
                    "selected_context": (
                        None if revision_timing is None else revision_timing.selected_context
                    ),
                    "estimated_input_tokens": (
                        None if revision_timing is None else revision_timing.estimated_input_tokens
                    ),
                    "output_budget": (
                        None if revision_timing is None else revision_timing.output_budget
                    ),
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
        _, _, unresolved_claims = prepare_grounding_claims(request, final_candidate or "")
        review_prompt = build_grounding_review_prompt(request, unresolved_claims)
        final_timing = final_review.timing
        _write_json(
            directory / "grounding_review_final_prompt.json",
            {
                "prompt_version": review_prompt.version,
                "system_message": review_prompt.system_message,
                "user_message": review_prompt.user_message,
                "source_ref_map": grounding_source_ref_map(request),
                "selected_context": (
                    None if final_timing is None else final_timing.selected_context
                ),
                "estimated_input_tokens": (
                    None if final_timing is None else final_timing.estimated_input_tokens
                ),
                "output_budget": final_review.review_output_budget,
                "claims_extracted": final_review.claims_extracted,
                "direct_supported_claim_count": final_review.direct_supported_claim_count,
                "unresolved_claim_count": final_review.unresolved_claim_count,
            },
        )

    (directory / "result.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    if result.final_output is not None:
        (directory / "output.md").write_text(result.final_output, encoding="utf-8")

    _write_diagnostics(directory, request, result, retrieval)
