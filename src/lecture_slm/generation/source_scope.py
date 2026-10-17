"""Deterministic source-scope normalization and requested-scope provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypeGuard

from lecture_slm.generation.models import (
    GenerationRequest,
    SourceScopeAssessment,
    SourceScopeStatus,
    TaskTeachingPlan,
)
from lecture_slm.workspaces.models import WorkspaceItemRole

_SCOPE_STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "this",
    "that",
    "is",
    "are",
    "from",
    "into",
    "using",
}

_TOKEN_ALIASES = {
    "async": "asynchronous",
    "await": "asynchronous",
    "taskbased": "task",
    "task-based": "task",
    "tasks": "task",
    "classes": "class",
    "methods": "method",
    "parameters": "parameter",
}

_GENERIC_SCOPE_PATTERNS = (
    r"\brequested topic\b",
    r"\brequested topics\b",
    r"\bbroader requested topic\b",
    r"\bbroader topic\b",
    r"\bcurrent topic\b",
    r"\bcourse topic\b",
    r"\bcourse scope\b",
    r"\bsequence\b",
    r"\broadmap\b",
    r"\bprevious topic\b",
    r"\bnext topic\b",
)

_PLAN_SCOPE_LOW_INFORMATION_TERMS = {
    "class",
    "type",
    "code",
    "example",
    "method",
    "syntax",
    "usage",
    "topic",
    "content",
    "section",
}

_DEFERRAL_MARKERS = (
    r"\boutside\b.*\bsource scope\b",
    r"\boutside\b.*\bsupported scope\b",
    r"\bwill not be covered\b",
    r"\bnot covered\b",
    r"\bout of scope\b",
    r"\bdefer(?:red)?\b",
    r"\bdeferred\b",
)

_GENERIC_CLASS_SCOPE_MARKERS = (
    r"\bgeneric\s+class(?:es)?\b",
    r"\btype\s+parameter(?:s)?\b",
    r"\bgeneric\s+constraint(?:s)?\b",
    r"\bconstraint(?:s)?\s+and\s+inheritance\b",
    r"\bgeneric\s+method(?:s)?\s+vs\s+generic\s+class(?:es)?\b",
    r"\bdeclaring\b.{0,40}\bclass(?:es)?\b",
)


@dataclass(frozen=True)
class RequestedScopeContext:
    terms: set[str]
    phrases: set[str]


@dataclass(frozen=True)
class SourceScopePlanConflict:
    unsupported_topic: str
    plan_location: str
    plan_text: str
    overlap_terms: list[str]


@dataclass(frozen=True)
class SourceScopePlanValidation:
    valid: bool
    conflicts: list[SourceScopePlanConflict]
    core_scope_unavailable: bool
    raw_status: SourceScopeStatus | None
    normalized_status: SourceScopeStatus | None
    effective_status: SourceScopeStatus | None
    core_unsupported_topics: list[str]
    optional_unsupported_topics: list[str]

    def diagnostics(self) -> dict[str, object]:
        return {
            "raw_status": None if self.raw_status is None else self.raw_status.value,
            "normalized_status": (
                None if self.normalized_status is None else self.normalized_status.value
            ),
            "effective_status": (
                None if self.effective_status is None else self.effective_status.value
            ),
            "valid": self.valid,
            "core_scope_unavailable": self.core_scope_unavailable,
            "core_unsupported_topics": self.core_unsupported_topics,
            "optional_unsupported_topics": self.optional_unsupported_topics,
            "conflicts": [
                {
                    "unsupported_topic": conflict.unsupported_topic,
                    "plan_location": conflict.plan_location,
                    "plan_text": conflict.plan_text,
                    "overlap_terms": conflict.overlap_terms,
                }
                for conflict in self.conflicts
            ],
        }


_SOURCE_SCOPE_TOPIC_LISTS = (
    "unsupported_requested_topics",
    "core_unsupported_topics",
    "optional_unsupported_topics",
)


def _is_string_list(value: object) -> TypeGuard[list[str]]:
    return isinstance(value, list) and all(isinstance(topic, str) for topic in value)


def _source_topic_key(topic: str) -> str:
    return " ".join(topic.split()).casefold()


def _canonicalize_topic_list(
    field: str,
    topics: list[str],
) -> tuple[list[str], list[dict[str, str]]]:
    canonical: list[str] = []
    repairs: list[dict[str, str]] = []
    seen: set[str] = set()
    for topic in topics:
        trimmed = topic.strip()
        if not trimmed:
            repairs.append({"list": field, "topic": topic, "action": "removed_blank"})
            continue
        if trimmed != topic:
            repairs.append({"list": field, "topic": topic, "action": "trimmed"})
        key = _source_topic_key(trimmed)
        if key in seen:
            repairs.append({"list": field, "topic": topic, "action": "deduplicated"})
            continue
        seen.add(key)
        canonical.append(trimmed)
    return canonical, repairs


def canonicalize_source_scope_payload(
    payload: object,
) -> tuple[object, dict[str, object]]:
    """Repair only hierarchy and list-shape conflicts in a Planner JSON payload."""
    if not isinstance(payload, dict):
        return payload, {}
    source_scope = payload.get("source_scope")
    if not isinstance(source_scope, dict):
        return payload, {}

    raw_topics = {field: source_scope.get(field, []) for field in _SOURCE_SCOPE_TOPIC_LISTS}
    if not all(_is_string_list(topics) for topics in raw_topics.values()):
        return payload, {
            "applied": False,
            "raw_topics": raw_topics,
            "canonical_topics": raw_topics,
            "repairs": [],
        }

    canonical_lists: dict[str, list[str]] = {}
    repairs: list[dict[str, str]] = []
    for field in _SOURCE_SCOPE_TOPIC_LISTS:
        topics = raw_topics[field]
        if _is_string_list(topics):
            canonical_lists[field], field_repairs = _canonicalize_topic_list(field, topics)
            repairs.extend(field_repairs)

    canonical_requested = canonical_lists["unsupported_requested_topics"]
    requested_keys = {_source_topic_key(topic) for topic in canonical_requested}
    for topic in canonical_lists["core_unsupported_topics"]:
        key = _source_topic_key(topic)
        if key not in requested_keys:
            canonical_requested.append(topic)
            requested_keys.add(key)
            repairs.append(
                {
                    "list": "unsupported_requested_topics",
                    "topic": topic,
                    "action": "added_core_topic",
                }
            )

    canonical_optional: list[str] = []
    for topic in canonical_lists["optional_unsupported_topics"]:
        if _source_topic_key(topic) in requested_keys:
            repairs.append(
                {
                    "list": "optional_unsupported_topics",
                    "topic": topic,
                    "action": "removed_higher_priority_topic",
                }
            )
            continue
        canonical_optional.append(topic)
    canonical_lists["optional_unsupported_topics"] = canonical_optional

    canonical_scope = dict(source_scope)
    for field, topics in canonical_lists.items():
        if field in source_scope or topics:
            canonical_scope[field] = topics
    canonical_payload = dict(payload)
    canonical_payload["source_scope"] = canonical_scope
    return canonical_payload, {
        "applied": bool(repairs),
        "raw_topics": raw_topics,
        "canonical_topics": canonical_lists,
        "repairs": repairs,
    }


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"\[[^\]]*\]\([^)]+\)", " ", text)
    cleaned = re.sub(r"[*_`#>\-]+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s/]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip().casefold()


def _normalized_tokens(text: str) -> set[str]:
    tokens = set()
    for token in _normalize_text(text).replace("/", " ").split():
        canonical = _TOKEN_ALIASES.get(token, token)
        if len(canonical) < 3 or canonical in _SCOPE_STOP_WORDS:
            continue
        tokens.add(canonical)
    return tokens


def _compact_content_lines(content: str, *, limit: int) -> list[str]:
    lines = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def build_requested_scope_terms(request: GenerationRequest) -> RequestedScopeContext:
    phrases: set[str] = set()
    terms: set[str] = set()

    def add_phrase(text: str) -> None:
        normalized = _normalize_text(text)
        if normalized:
            phrases.add(normalized)
            terms.update(_normalized_tokens(normalized))

    add_phrase(request.instruction)
    if request.course is not None:
        add_phrase(request.course.name)
        add_phrase(request.course.id)
    for topic in request.previous_topics:
        add_phrase(topic)
    if request.previous_course_context is not None:
        for topic in request.previous_course_context.recently_taught_topics:
            add_phrase(topic)
        for term in request.previous_course_context.known_terminology:
            add_phrase(term)
        for topic in request.previous_course_context.not_yet_taught:
            add_phrase(topic)
        if request.previous_course_context.previous_lecture_summary is not None:
            add_phrase(request.previous_course_context.previous_lecture_summary)
        if request.previous_course_context.previous_lab_summary is not None:
            add_phrase(request.previous_course_context.previous_lab_summary)
        if request.previous_course_context.additional_context is not None:
            add_phrase(request.previous_course_context.additional_context)
    if request.workspace_context is not None:
        for item in request.workspace_context.items:
            add_phrase(item.title)
            if item.role is WorkspaceItemRole.CONTEXT:
                for line in _compact_content_lines(item.content, limit=6):
                    add_phrase(line)
            elif item.role is WorkspaceItemRole.HISTORY:
                for line in _compact_content_lines(item.content, limit=2):
                    add_phrase(line)
    return RequestedScopeContext(terms=terms, phrases=phrases)


def _topic_matches_requested_scope(topic: str, scope: RequestedScopeContext) -> bool:
    normalized_topic = _normalize_text(topic)
    if not normalized_topic:
        return False
    if any(
        re.search(pattern, normalized_topic, flags=re.IGNORECASE)
        for pattern in _GENERIC_SCOPE_PATTERNS
    ):
        return True
    if normalized_topic in scope.phrases:
        return True
    topic_terms = _normalized_tokens(normalized_topic)
    if not topic_terms:
        return False
    overlap = topic_terms & scope.terms
    if overlap:
        return True
    if len(topic_terms) == 1:
        return False
    words = normalized_topic.split()
    return any(" ".join(pair) in scope.phrases for pair in zip(words, words[1:], strict=False))


def normalize_source_scope_with_diagnostics(
    request: GenerationRequest,
    source_scope: SourceScopeAssessment | None,
) -> tuple[SourceScopeAssessment | None, dict[str, object]]:
    if source_scope is None:
        return None, {
            "raw_supported_topics": [],
            "raw_unsupported_requested_topics": [],
            "raw_core_unsupported_topics": [],
            "raw_optional_unsupported_topics": [],
            "normalized_unsupported_requested_topics": [],
            "normalized_core_unsupported_topics": [],
            "optional_unsupported_topics": [],
            "removed_unsupported_topics": [],
            "preserved_core_topics_without_lexical_overlap": [],
        }

    scope = build_requested_scope_terms(request)
    raw_topics = list(source_scope.unsupported_requested_topics)
    raw_core_topics = list(source_scope.core_unsupported_topics)
    core_keys = {topic.strip().casefold() for topic in raw_core_topics}
    kept_topics: list[str] = []
    removed_topics: list[dict[str, str]] = []
    preserved_core_topics: list[str] = []
    for topic in raw_topics:
        is_core = topic.strip().casefold() in core_keys
        if is_core:
            if not _topic_matches_requested_scope(topic, scope):
                preserved_core_topics.append(topic)
            kept_topics.append(topic)
        elif _topic_matches_requested_scope(topic, scope):
            kept_topics.append(topic)
        else:
            removed_topics.append(
                {
                    "topic": topic,
                    "reason": "No overlap with authoritative requested/course scope.",
                }
            )
    kept_keys = {topic.strip().casefold() for topic in kept_topics}
    kept_core_topics = [topic for topic in raw_core_topics if topic.strip().casefold() in kept_keys]
    status = source_scope.status
    if not kept_topics and source_scope.optional_unsupported_topics:
        status = SourceScopeStatus.SUFFICIENT
    normalized = source_scope.model_copy(
        update={
            "status": status,
            "unsupported_requested_topics": kept_topics,
            "core_unsupported_topics": kept_core_topics,
        }
    )
    return normalized, {
        "raw_supported_topics": list(source_scope.supported_topics),
        "raw_unsupported_requested_topics": raw_topics,
        "raw_core_unsupported_topics": raw_core_topics,
        "raw_optional_unsupported_topics": list(source_scope.optional_unsupported_topics),
        "normalized_unsupported_requested_topics": kept_topics,
        "normalized_core_unsupported_topics": kept_core_topics,
        "optional_unsupported_topics": list(source_scope.optional_unsupported_topics),
        "removed_unsupported_topics": removed_topics,
        "preserved_core_topics_without_lexical_overlap": preserved_core_topics,
    }


def normalize_source_scope(
    request: GenerationRequest,
    source_scope: SourceScopeAssessment | None,
) -> SourceScopeAssessment | None:
    normalized, _ = normalize_source_scope_with_diagnostics(request, source_scope)
    return normalized


def _iter_plan_text_fragments(value: object, path: str = "plan") -> list[tuple[str, str]]:
    if isinstance(value, str):
        text = " ".join(value.split())
        return [(path, text)] if text else []
    if isinstance(value, list):
        fragments: list[tuple[str, str]] = []
        for index, item in enumerate(value):
            fragments.extend(_iter_plan_text_fragments(item, f"{path}[{index}]"))
        return fragments
    if isinstance(value, dict):
        fragments = []
        for key, item in value.items():
            if key == "source_scope":
                continue
            fragments.extend(_iter_plan_text_fragments(item, f"{path}.{key}"))
        return fragments
    return []


def _is_deferral_fragment(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in _DEFERRAL_MARKERS)


def _meaningful_terms(tokens: set[str]) -> set[str]:
    return tokens - _PLAN_SCOPE_LOW_INFORMATION_TERMS


def _generic_class_scope_conflict(topic_terms: set[str], fragment_text: str) -> list[str]:
    if not {"generic", "class"}.issubset(topic_terms):
        return []
    if not any(
        re.search(pattern, fragment_text, flags=re.IGNORECASE)
        for pattern in _GENERIC_CLASS_SCOPE_MARKERS
    ):
        return []
    marker_terms = {
        token
        for token in _normalized_tokens(fragment_text)
        if token
        in {
            "generic",
            "class",
            "type",
            "parameter",
            "constraint",
            "inheritance",
            "method",
            "syntax",
            "usage",
        }
    }
    return sorted(marker_terms)


def _topic_plan_overlap(topic: str, fragment_text: str) -> list[str]:
    topic_terms = _normalized_tokens(topic)
    fragment_terms = _normalized_tokens(fragment_text)
    if not topic_terms or not fragment_terms:
        return []
    normalized_topic = _normalize_text(topic)
    normalized_fragment = _normalize_text(fragment_text)
    if normalized_topic and normalized_topic in normalized_fragment:
        overlap = _meaningful_terms(topic_terms & fragment_terms)
        return sorted(overlap or (topic_terms & fragment_terms))

    overlap = _meaningful_terms(topic_terms) & _meaningful_terms(fragment_terms)
    generic_class_overlap = _generic_class_scope_conflict(topic_terms, fragment_text)
    if generic_class_overlap:
        return generic_class_overlap
    meaningful_topic_terms = _meaningful_terms(topic_terms)
    if len(meaningful_topic_terms) == 1:
        return sorted(overlap)
    if len(overlap) >= 2:
        return sorted(overlap)
    return []


def validate_plan_against_source_scope(
    request: GenerationRequest,
    plan: TaskTeachingPlan,
    *,
    raw_scope: SourceScopeAssessment | None = None,
) -> SourceScopePlanValidation:
    raw_status = None if raw_scope is None else raw_scope.status
    normalized_scope = plan.source_scope
    normalized_status = None if normalized_scope is None else normalized_scope.status
    if normalized_scope is None:
        effective_status = normalized_status
        return SourceScopePlanValidation(
            valid=True,
            conflicts=[],
            core_scope_unavailable=False,
            raw_status=raw_status,
            normalized_status=normalized_status,
            effective_status=effective_status,
            core_unsupported_topics=[],
            optional_unsupported_topics=[],
        )

    core_scope_unavailable = bool(normalized_scope.core_unsupported_topics)
    conflicts: list[SourceScopePlanConflict] = []
    seen_conflicts: set[tuple[str, str, str]] = set()
    if (
        normalized_scope.status is SourceScopeStatus.PARTIAL
        and normalized_scope.unsupported_requested_topics
    ):
        fragments = _iter_plan_text_fragments(plan.model_dump(mode="json"))
        for location, fragment_text in fragments:
            if _is_deferral_fragment(fragment_text):
                continue
            for topic in normalized_scope.unsupported_requested_topics:
                overlap_terms = _topic_plan_overlap(topic, fragment_text)
                if not overlap_terms:
                    continue
                key = (topic, location, fragment_text.casefold())
                if key in seen_conflicts:
                    continue
                seen_conflicts.add(key)
                conflicts.append(
                    SourceScopePlanConflict(
                        unsupported_topic=topic,
                        plan_location=location,
                        plan_text=fragment_text,
                        overlap_terms=overlap_terms,
                    )
                )

    effective_status = (
        SourceScopeStatus.INSUFFICIENT if core_scope_unavailable else normalized_scope.status
    )
    return SourceScopePlanValidation(
        valid=not conflicts and not core_scope_unavailable,
        conflicts=conflicts,
        core_scope_unavailable=core_scope_unavailable,
        raw_status=raw_status,
        normalized_status=normalized_status,
        effective_status=effective_status,
        core_unsupported_topics=list(normalized_scope.core_unsupported_topics),
        optional_unsupported_topics=list(normalized_scope.optional_unsupported_topics),
    )
