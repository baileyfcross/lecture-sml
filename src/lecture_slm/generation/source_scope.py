"""Deterministic source-scope normalization and requested-scope provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass

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
            "raw_unsupported_requested_topics": [],
            "normalized_unsupported_requested_topics": [],
            "removed_unsupported_topics": [],
        }

    scope = build_requested_scope_terms(request)
    raw_topics = list(source_scope.unsupported_requested_topics)
    kept_topics: list[str] = []
    removed_topics: list[dict[str, str]] = []
    for topic in raw_topics:
        if _topic_matches_requested_scope(topic, scope):
            kept_topics.append(topic)
            continue
        removed_topics.append(
            {
                "topic": topic,
                "reason": "No overlap with authoritative requested/course scope.",
            }
        )
    normalized = source_scope.model_copy(update={"unsupported_requested_topics": kept_topics})
    return normalized, {
        "raw_unsupported_requested_topics": raw_topics,
        "normalized_unsupported_requested_topics": kept_topics,
        "removed_unsupported_topics": removed_topics,
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
    if overlap:
        return sorted(overlap)

    generic_class_overlap = _generic_class_scope_conflict(topic_terms, fragment_text)
    if generic_class_overlap:
        return generic_class_overlap
    return []


def _is_core_scope_unavailable(
    request: GenerationRequest,
    plan: TaskTeachingPlan,
    conflicts: list[SourceScopePlanConflict],
) -> bool:
    if not conflicts:
        return False
    instruction_terms = _meaningful_terms(_normalized_tokens(request.instruction))
    conflicted_topics = {conflict.unsupported_topic for conflict in conflicts}
    for topic in conflicted_topics:
        topic_terms = _meaningful_terms(_normalized_tokens(topic))
        if topic_terms and topic_terms & instruction_terms:
            return True

    substantial_fragments = [
        fragment
        for path, fragment in _iter_plan_text_fragments(plan.model_dump(mode="json"))
        if path.startswith("plan.artifact_structure")
        or path.startswith("plan.objectives")
        or path.startswith("plan.concept_sequence")
        or path.startswith("plan.slide_sequence")
        or path.startswith("plan.steps")
        or path.startswith("plan.questions")
    ]
    if not substantial_fragments:
        substantial_fragments = [
            fragment for _, fragment in _iter_plan_text_fragments(plan.model_dump(mode="json"))
        ]
    if not substantial_fragments:
        return False
    conflict_ratio = len(conflicts) / len(substantial_fragments)
    return len(conflicts) >= 2 and conflict_ratio >= 0.3


def validate_plan_against_source_scope(
    request: GenerationRequest,
    plan: TaskTeachingPlan,
    *,
    raw_scope: SourceScopeAssessment | None = None,
) -> SourceScopePlanValidation:
    raw_status = None if raw_scope is None else raw_scope.status
    normalized_scope = plan.source_scope
    normalized_status = None if normalized_scope is None else normalized_scope.status
    if (
        normalized_scope is None
        or normalized_scope.status is not SourceScopeStatus.PARTIAL
        or not normalized_scope.unsupported_requested_topics
    ):
        effective_status = normalized_status
        return SourceScopePlanValidation(
            valid=True,
            conflicts=[],
            core_scope_unavailable=False,
            raw_status=raw_status,
            normalized_status=normalized_status,
            effective_status=effective_status,
        )

    fragments = _iter_plan_text_fragments(plan.model_dump(mode="json"))
    conflicts: list[SourceScopePlanConflict] = []
    seen_conflicts: set[tuple[str, str, str]] = set()
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

    core_scope_unavailable = _is_core_scope_unavailable(request, plan, conflicts)
    effective_status = (
        SourceScopeStatus.INSUFFICIENT if core_scope_unavailable else normalized_scope.status
    )
    return SourceScopePlanValidation(
        valid=not conflicts,
        conflicts=conflicts,
        core_scope_unavailable=core_scope_unavailable,
        raw_status=raw_status,
        normalized_status=normalized_status,
        effective_status=effective_status,
    )
