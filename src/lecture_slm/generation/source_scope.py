"""Deterministic source-scope normalization and requested-scope provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass

from lecture_slm.generation.models import GenerationRequest, SourceScopeAssessment
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


@dataclass(frozen=True)
class RequestedScopeContext:
    terms: set[str]
    phrases: set[str]


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
