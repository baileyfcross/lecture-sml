"""Conservative canonicalization of conversational instructions for retrieval."""

import re

from lecture_slm.knowledge.models import RetrievalExpansionPlan, RetrievalQuery

_REQUEST_PREFIXES = (
    re.compile(r"^(?:can|could)\s+you\s+explain\b\s*", re.IGNORECASE),
    re.compile(r"^please\s+explain\b\s*", re.IGNORECASE),
    re.compile(r"^explain\b\s*", re.IGNORECASE),
    re.compile(r"^tell\s+me\s+about\b\s*", re.IGNORECASE),
    re.compile(r"^what\s+is\b\s*", re.IGNORECASE),
    re.compile(r"^give\s+me\s+(?:an?\s+)?explanation\s+of\b\s*", re.IGNORECASE),
    re.compile(
        r"^(?:please\s+)?(?:create|make)\s+(?:a\s+)?(?:lecture|slides?|lab)\s+"
        r"(?:about|on)\b\s*",
        re.IGNORECASE,
    ),
)
_TRAILING_PUNCTUATION = " \t\r\n.!?;:,"
MAX_RETRIEVAL_EXPANSION_QUERIES = 3


def canonicalize_retrieval_query(instruction: str) -> RetrievalQuery:
    """Remove only recognized leading request scaffolding; retain topical wording."""

    original = instruction.strip()
    if not original:
        raise ValueError("Cannot canonicalize an empty retrieval instruction")
    canonical = original
    for prefix in _REQUEST_PREFIXES:
        canonical = prefix.sub("", canonical, count=1)
        if canonical != original:
            break
    canonical = re.sub(r"\s+", " ", canonical.strip()).strip(_TRAILING_PUNCTUATION).strip()
    if not canonical or not re.search(r"\w", canonical, flags=re.UNICODE):
        canonical = original
    return RetrievalQuery(original=original, canonical=canonical)


def build_retrieval_expansion_plan(
    canonical_query: str,
    missing_topics: list[str],
) -> RetrievalExpansionPlan:
    """Anchor up to three unique missing topics to the original retrieval query."""

    canonical = re.sub(r"\s+", " ", canonical_query.strip())
    canonical = canonical.strip(_TRAILING_PUNCTUATION)
    if not canonical or not re.search(r"\w", canonical, flags=re.UNICODE):
        raise ValueError("Cannot expand an empty canonical retrieval query")

    queries: list[str] = []
    topics: list[str] = []
    seen_topics: set[str] = set()
    canonical_key = canonical.casefold()
    canonical_terms = {
        re.sub(r"^[^\w]+|[^\w]+$", "", token, flags=re.UNICODE).casefold()
        for token in canonical.split()
    }
    for topic in missing_topics:
        normalized_topic = re.sub(r"\s+", " ", topic.strip()).strip(_TRAILING_PUNCTUATION)
        topic_key = normalized_topic.casefold()
        if (
            not normalized_topic
            or not re.search(r"\w", normalized_topic, flags=re.UNICODE)
            or topic_key in seen_topics
        ):
            continue
        seen_topics.add(topic_key)

        if _contains_phrase(normalized_topic, canonical):
            query = normalized_topic
        else:
            topic_terms = [
                token
                for token in normalized_topic.split()
                if re.sub(r"^[^\w]+|[^\w]+$", "", token, flags=re.UNICODE).casefold()
                not in canonical_terms
            ]
            query = f"{canonical} {' '.join(topic_terms)}"
        query = re.sub(r"\s+", " ", query).strip()
        if query.casefold() == canonical_key or query.casefold() in {
            existing.casefold() for existing in queries
        }:
            continue
        topics.append(normalized_topic)
        queries.append(query)
        if len(queries) == MAX_RETRIEVAL_EXPANSION_QUERIES:
            break

    return RetrievalExpansionPlan(
        original_query=canonical,
        expansion_queries=queries,
        missing_topics=topics,
    )


def _contains_phrase(text: str, phrase: str) -> bool:
    return (
        re.search(
            rf"(?<!\w){re.escape(phrase)}(?!\w)",
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )
