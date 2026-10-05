"""Ground-truth retrieval and resolution metrics."""

import re
from collections.abc import Sequence
from typing import Any

from lecture_slm.knowledge.evaluation.models import RetrievalEvalCase
from lecture_slm.knowledge.models import RetrievalResult


def classify_resolution_failures(result: RetrievalResult) -> list[str]:
    """Return only failure categories that can be determined from resolver diagnostics."""

    warnings = " ".join(result.warnings).casefold()
    categories: list[str] = []
    if "ambiguous" in warnings:
        categories.append("source_ambiguous")
    if "no indexed source matched" in warnings or "source id is not indexed" in warnings:
        categories.append("source_not_found")
    if "no extracted heading matched" in warnings:
        categories.append("section_not_found")
    return categories


def case_metrics(case: RetrievalEvalCase, result: RetrievalResult, *, top_k: int) -> dict[str, Any]:
    expected_ids = set(case.expected_source_ids)
    expected_titles = {_normalize(title) for title in case.expected_source_titles}
    expected_sections = {_normalize(section) for section in case.expected_sections}
    expected_pages = set(case.expected_pages)

    relevant_ranks: list[int] = []
    for match in result.matches:
        if match.neighbor_of is not None:
            continue
        rank = match.rerank_rank or match.fused_rank
        source_relevant = (
            (not expected_ids and not expected_titles)
            or match.source_id in expected_ids
            or _normalize(match.source_title) in expected_titles
        )
        section_relevant = (
            not expected_sections
            or _normalize(match.section or "") in expected_sections
            or any(
                _normalize(section) in {_normalize(part) for part in match.section_path}
                for section in case.expected_sections
            )
        )
        page_relevant = not expected_pages or match.page_number in expected_pages
        if source_relevant and section_relevant and page_relevant:
            relevant_ranks.append(rank)

    expected_source_known = bool(expected_ids or expected_titles)
    passage_truth_known = bool(expected_source_known or expected_sections or expected_pages)
    resolved_source = result.resolved_source or {}
    resolved_id = str(resolved_source.get("source_id", ""))
    resolved_title = _normalize(str(resolved_source.get("title", "")))
    source_resolution_accuracy = (
        bool(
            (expected_ids and resolved_id in expected_ids)
            or (expected_titles and resolved_title in expected_titles)
        )
        if expected_source_known
        else None
    )
    resolved_section = _normalize(result.resolved_section or "")
    section_resolution_accuracy = (
        any(
            resolved_section == _normalize(expected)
            or resolved_section.startswith(_normalize(expected) + " ")
            for expected in expected_sections
        )
        if expected_sections
        else None
    )
    missing_expected = case.category.casefold() in {"missing_source", "missing source"}
    ambiguous_expected = case.category.casefold() in {
        "ambiguous_source",
        "ambiguous source",
    }
    fail_closed_accuracy = (
        result.resolved_source is None and not result.matches if missing_expected else None
    )
    ambiguity_detection_accuracy = (
        result.resolved_source is None
        and any("ambiguous" in warning.casefold() for warning in result.warnings)
        if ambiguous_expected
        else None
    )
    resolution_reason = str(resolved_source.get("resolution_reason", "unresolved"))
    if not result.resolved_source:
        warning_text = " ".join(result.warnings).casefold()
        if "ambiguous" in warning_text:
            resolution_reason = "ambiguous"
        elif "no indexed source matched" in warning_text or "not indexed" in warning_text:
            resolution_reason = "missing"
    requested_section = _normalize(case.section or "")
    section_method = "unresolved"
    if resolved_section:
        if requested_section and resolved_section == requested_section:
            section_method = "exact"
        elif requested_section and resolved_section.startswith(requested_section + " "):
            section_method = "prefix"
        else:
            section_method = "heading_match"
    return {
        "hit_at_1": bool(any(rank <= 1 for rank in relevant_ranks))
        if passage_truth_known
        else None,
        "hit_at_3": bool(any(rank <= 3 for rank in relevant_ranks))
        if passage_truth_known
        else None,
        "hit_at_5": bool(any(rank <= 5 for rank in relevant_ranks))
        if passage_truth_known
        else None,
        "hit_at_k": bool(any(rank <= top_k for rank in relevant_ranks))
        if passage_truth_known
        else None,
        "mrr": (1.0 / relevant_ranks[0] if relevant_ranks else 0.0)
        if passage_truth_known
        else None,
        "relevant_ranks": relevant_ranks,
        "source_resolution_accuracy": source_resolution_accuracy,
        "section_resolution_accuracy": section_resolution_accuracy,
        "fail_closed_accuracy": fail_closed_accuracy,
        "ambiguity_detection_accuracy": ambiguity_detection_accuracy,
        "source_resolution_method": resolution_reason,
        "section_resolution_method": section_method,
    }


def aggregate_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_k",
        "mrr",
        "source_resolution_accuracy",
        "section_resolution_accuracy",
        "fail_closed_accuracy",
        "ambiguity_detection_accuracy",
    )
    summary: dict[str, Any] = {"case_count": len(records)}
    for key in keys:
        values = [
            float(record["metrics"][key])
            for record in records
            if record["metrics"].get(key) is not None
        ]
        summary[key] = sum(values) / len(values) if values else None
        summary[f"{key}_case_count"] = len(values)
    for key in ("source_resolution_method", "section_resolution_method"):
        counts: dict[str, int] = {}
        for record in records:
            value = str(record["metrics"].get(key, "unresolved"))
            counts[value] = counts.get(value, 0) + 1
        summary[key + "_counts"] = counts
    return summary


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
