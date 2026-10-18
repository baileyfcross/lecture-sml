"""Deterministic normalization of generation requests into factual search queries."""

import re

from lecture_slm.knowledge.models import RetrievalExpansionPlan, RetrievalQuery

_INTRODUCTION_REQUEST_PREFIX = re.compile(
    r"^(?:create|generate|make|write(?:\s+out)?|build|prepare|produce|develop|give\s+me)\s+"
    r"(?:(?:an\s+)?introduction\s+to|(?:(?:the|a|an)\s+)?"
    r"(?:(?:next|following|previous|last|current)\s+)?"
    r"(?:lecture|slides?|slide\s+deck|lab|activity|assignment|homework|"
    r"instructor\s+guide|assessment)\s+introducing)\s+",
    re.IGNORECASE,
)
_LEADING_REQUEST_PHRASES = (
    re.compile(r"^(?:can|could|would)\s+you(?:\s+please)?\s+", re.IGNORECASE),
    re.compile(r"^please\s+", re.IGNORECASE),
    _INTRODUCTION_REQUEST_PREFIX,
    re.compile(
        r"^(?:create|generate|make|write(?:\s+out)?|build|prepare|produce|develop)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:explain|teach(?:\s+me\s+about)?|tell\s+me\s+about|what\s+is|what\s+are|"
        r"give\s+me\s+(?:an?\s+)?explanation\s+of)\s+",
        re.IGNORECASE,
    ),
)
_ARTIFACT_PREFIX = re.compile(
    r"^(?:(?:the|a|an)\s+)?(?:(?:next|following|previous|last|current)\s+|"
    r"\d+(?:st|nd|rd|th)?\s+)?"
    r"(?:lecture|slides?|slide\s+deck|lab|activity|assignment|homework|"
    r"instructor\s+guide|assessment)\b(?:\s+\d+(?:st|nd|rd|th)?)?",
    re.IGNORECASE,
)
_LEADING_CONNECTORS = re.compile(r"^(?:(?:about|on|for)\s+)+(?:the\s+)?", re.IGNORECASE)
_TRAILING_COURTESY = re.compile(r"\s+for\s+(?:me|students|learners)\s*$", re.IGNORECASE)
_TRAILING_PUNCTUATION = " \t\r\n.!?;:,"
_COURSE_REFERENCE = re.compile(
    r"\b(?:(?:next|following|previous|last|current)\s+"
    r"(?:lecture|lesson|class|unit|topic)|"
    r"\d+(?:st|nd|rd|th)?\s+(?:lecture|lesson|class|unit)|"
    r"(?:lecture|lesson|class|unit)\s+\d+(?:st|nd|rd|th)?|"
    r"continue\s+(?:the\s+)?course)\b",
    re.IGNORECASE,
)
_ARTIFACT_WORDS = {
    "lecture",
    "lectures",
    "slide",
    "slides",
    "deck",
    "lab",
    "activity",
    "assignment",
    "homework",
    "guide",
    "assessment",
    "section",
    "requested",
    "unsupported",
    "topic",
    "topics",
    "explain",
    "teach",
    "students",
    "student",
    "for",
    "and",
    "the",
}
_DETAIL_TERMS = {
    "syntax",
    "usage",
    "examples",
    "example",
    "basics",
    "fundamentals",
    "behavior",
    "details",
    "implementation",
    "introduction",
}
_AMBIGUOUS_SINGLE_TOPICS = {
    "class",
    "classes",
    "function",
    "functions",
    "interface",
    "interfaces",
    "method",
    "methods",
    "type",
    "types",
}
_ANCHOR_SUFFIXES = (" class", " course", " workspace")
MAX_RETRIEVAL_EXPANSION_QUERIES = 3
_MAX_HISTORY_TITLES = 1


def canonicalize_retrieval_query(instruction: str) -> RetrievalQuery:
    """Remove recognized request scaffolding while preserving subject wording."""

    query, _ = build_factual_retrieval_query(instruction)
    return query


def build_factual_retrieval_query(
    instruction: str,
    *,
    workspace_name: str | None = None,
    history_titles: list[str] | None = None,
) -> tuple[RetrievalQuery, dict[str, object]]:
    """Normalize an instruction and apply a Workspace domain anchor only when relevant."""

    original = instruction.strip()
    if not original:
        raise ValueError("Cannot canonicalize an empty retrieval instruction")
    raw_topic = _strip_request_scaffolding(original)
    anchor = _domain_anchor(workspace_name or "")
    considered = bool(workspace_name and workspace_name.strip())
    course_relative = _COURSE_REFERENCE.search(original) is not None

    self_contained_topic_terms = _normalized_terms(raw_topic)
    ambiguous_topic = (
        len(self_contained_topic_terms) == 1
        and self_contained_topic_terms[0].casefold() in _AMBIGUOUS_SINGLE_TOPICS
    )

    applied = False
    reason = "No Workspace identity supplied."
    if considered:
        if course_relative:
            applied = True
            reason = "Request depends on the active course sequence."
        elif not self_contained_topic_terms or ambiguous_topic:
            applied = True
            reason = (
                "Request has no self-contained factual topic."
                if not self_contained_topic_terms
                else "Request topic is ambiguous without the active course domain."
            )
        else:
            reason = "Request contains a self-contained factual topic."

    topic_for_query = raw_topic
    if applied:
        topic_for_query = _remove_phrase(raw_topic, workspace_name or "")
        if course_relative and anchor:
            topic_for_query = _remove_course_label(topic_for_query, anchor)
    topic_terms = _normalized_terms(topic_for_query)

    parts: list[str] = []
    if applied and anchor:
        parts.append(anchor)
    history_topics: list[str] = []
    if applied and course_relative and not topic_terms:
        for title in history_titles or []:
            history_topic = _history_topic_phrase(title)
            if history_topic:
                history_topics.append(history_topic)
            if len(history_topics) >= _MAX_HISTORY_TITLES:
                break
        parts.extend(history_topics)
    if not applied:
        canonical = _clean_query(topic_for_query)
    else:
        if topic_for_query.strip():
            parts.append(topic_for_query)
        canonical = _join_unique_terms(parts)
    if not canonical:
        canonical = _join_unique_terms([anchor] if applied else []) or original

    query = RetrievalQuery(original=original, canonical=canonical)
    diagnostics: dict[str, object] = {
        "raw_instruction": original,
        "canonical_factual_query": canonical,
        "query_terms": [term.casefold() for term in _normalized_terms(canonical)],
        "workspace_history_topics_used": history_topics,
        "workspace_anchor": {
            "considered": considered,
            "applied": applied,
            "value": anchor if applied else None,
            "reason": reason,
        },
    }
    return query, diagnostics


def build_retrieval_expansion_plan(
    canonical_query: str,
    missing_topics: list[str],
    *,
    domain_anchor: str | None = None,
    core_topics: list[str] | None = None,
) -> RetrievalExpansionPlan:
    """Build up to three compact factual variants from normalized unsupported topics."""

    canonical = _clean_query(canonical_query)
    if not canonical:
        raise ValueError("Cannot expand an empty canonical retrieval query")
    anchor = _clean_query(domain_anchor or "") or canonical
    topics: list[str] = []
    seen_topics: set[str] = set()
    for topic in missing_topics:
        cleaned = _clean_topic(topic)
        key = cleaned.casefold()
        if cleaned and key not in seen_topics:
            topics.append(cleaned)
            seen_topics.add(key)
    core_keys = {
        _clean_topic(topic).casefold() for topic in core_topics or [] if _clean_topic(topic)
    }
    topics.sort(key=lambda topic: topic.casefold() not in core_keys)

    candidates: list[str] = []
    for topic in topics:
        terms = _normalized_terms(topic)
        core_terms = [term for term in terms if term.casefold() not in _DETAIL_TERMS]
        if not core_terms:
            core_terms = terms
        candidates.append(_join_unique_terms([anchor, " ".join(core_terms)]))

    for topic in topics:
        terms = _normalized_terms(topic)
        core_terms = [term for term in terms if term.casefold() not in _DETAIL_TERMS]
        detail_terms = [term for term in terms if term.casefold() in _DETAIL_TERMS]
        if not core_terms:
            continue
        for detail in detail_terms:
            candidates.append(_join_unique_terms([anchor, " ".join(core_terms), detail]))

    queries: list[str] = []
    for candidate in candidates:
        cleaned = _clean_query(candidate)
        if (
            cleaned
            and cleaned.casefold() != canonical.casefold()
            and cleaned.casefold() not in {item.casefold() for item in queries}
        ):
            queries.append(cleaned)
            if len(queries) == MAX_RETRIEVAL_EXPANSION_QUERIES:
                break

    return RetrievalExpansionPlan(
        original_query=canonical,
        domain_anchor=anchor if domain_anchor else None,
        expansion_queries=queries,
        missing_topics=topics,
    )


def _strip_request_scaffolding(text: str) -> str:
    value = re.sub(r"\s+", " ", text.strip())
    for _ in range(4):
        previous = value
        for pattern in _LEADING_REQUEST_PHRASES:
            value = pattern.sub("", value, count=1)
            if value != previous:
                break
        value = _ARTIFACT_PREFIX.sub("", value, count=1)
        value = _LEADING_CONNECTORS.sub("", value, count=1)
        value = _TRAILING_COURTESY.sub("", value)
        value = re.sub(r"\s+", " ", value).strip(_TRAILING_PUNCTUATION).strip()
        if value == previous:
            break
    return value


def _domain_anchor(workspace_name: str) -> str:
    value = re.sub(r"\s+", " ", workspace_name.strip())
    lowered = value.casefold()
    for suffix in _ANCHOR_SUFFIXES:
        if lowered.endswith(suffix):
            value = value[: -len(suffix)].strip()
            break
    return value.strip(_TRAILING_PUNCTUATION)


def _remove_phrase(text: str, phrase: str) -> str:
    normalized_phrase = re.sub(r"\s+", " ", phrase.strip())
    if not normalized_phrase:
        return text
    return re.sub(
        rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)",
        " ",
        text,
        flags=re.IGNORECASE,
    ).strip()


def _remove_course_label(text: str, anchor: str) -> str:
    value = _remove_phrase(text, anchor)
    value = re.sub(r"\b(?:class|course|workspace)\b", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip(_TRAILING_PUNCTUATION).strip()


def _history_topic_phrase(history_title: str) -> str:
    for line in history_title.splitlines():
        candidate = re.sub(r"^\s*#+\s*", "", line).strip()
        match = re.search(
            r"\b(?:lecture|lesson)\s+\d+(?:st|nd|rd|th)?\s*[-:–—]\s*(.+)$",
            candidate,
            flags=re.IGNORECASE,
        )
        if match:
            topic = re.sub(r"\s*\(\d+\s+minutes?\)\s*$", "", match.group(1), flags=re.IGNORECASE)
            return _strip_request_scaffolding(topic)
    return ""


def _normalized_terms(text: str) -> list[str]:
    return re.findall(r"C#|[^\W_]+(?:[#/+][^\W_]+)*", text, flags=re.IGNORECASE | re.UNICODE)


def _join_unique_terms(parts: list[str]) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for term in _normalized_terms(part):
            key = term.casefold()
            if key not in seen:
                terms.append(term)
                seen.add(key)
    return " ".join(terms)


def _clean_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip()).strip(_TRAILING_PUNCTUATION).strip()


def _clean_topic(topic: str) -> str:
    terms = [
        term if term.casefold() == "c#" or term.isupper() else term.lower()
        for term in _normalized_terms(topic)
        if term.casefold() not in _ARTIFACT_WORDS
    ]
    return _join_unique_terms([" ".join(terms)])
