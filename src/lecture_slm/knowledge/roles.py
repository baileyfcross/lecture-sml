"""Deterministic roles for separating factual prose from vault scaffolding."""

import re
from enum import StrEnum


class ChunkRole(StrEnum):
    CONTENT = "content"
    REFERENCE = "reference"
    METADATA = "metadata"
    NAVIGATION = "navigation"


_REFERENCE_HEADINGS = {
    "reference",
    "references",
    "source",
    "sources",
    "bibliography",
    "further reading",
    "reading",
}
_NAVIGATION_HEADINGS = {
    "linked full notes",
    "directly linked full notes",
    "related notes",
    "backlinks",
    "index",
    "navigation",
}
_ADMIN_LINE = re.compile(
    r"^\s*(?:"
    r"(?:created|creation date|updated|modified|date|status|tags?|aliases|course|source|"
    r"author|type|category|file|title|id)\s*:"
    r"|(?:\d{4}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2})?\s*$"
    r")",
    re.IGNORECASE,
)
_WIKILINK = re.compile(r"!?\[\[([^\]]+)\]\]")
_QUERY_BLOCK = re.compile(r"```(?:query|dataview|dataviewjs)\b", re.IGNORECASE)
_SOURCE_FILE_LINK = re.compile(r"\.(?:pdf|docx?|pptx?|epub|html?)$", re.IGNORECASE)


def classify_chunk_role(
    section: str | None,
    text: str,
    *,
    document_type: str = "markdown",
) -> ChunkRole:
    """Classify only strong structural signals; ambiguous prose remains factual content."""

    normalized_section = _normalize(section or "")
    if normalized_section in _NAVIGATION_HEADINGS or any(
        normalized_section.startswith(heading + " ") for heading in _NAVIGATION_HEADINGS
    ):
        return ChunkRole.NAVIGATION
    if normalized_section in _REFERENCE_HEADINGS or any(
        normalized_section.startswith(heading + " ") for heading in _REFERENCE_HEADINGS
    ):
        return ChunkRole.REFERENCE
    if document_type.casefold() != "markdown":
        return ChunkRole.CONTENT

    if _QUERY_BLOCK.search(text):
        return ChunkRole.NAVIGATION

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and _is_administrative_block(lines):
        return ChunkRole.METADATA

    links = list(_WIKILINK.finditer(text))
    visible_text = _WIKILINK.sub(" ", text)
    visible_words = re.findall(r"[a-z0-9]+", visible_text.casefold())
    linked_targets = [link.group(1).split("|", maxsplit=1)[0].strip() for link in links]
    link_only = bool(links) and len(visible_words) <= max(3, len(links))
    source_file_links = bool(linked_targets) and all(
        _SOURCE_FILE_LINK.search(target) for target in linked_targets
    )
    if link_only and source_file_links:
        return ChunkRole.REFERENCE
    if link_only:
        return ChunkRole.NAVIGATION

    return ChunkRole.CONTENT


def _is_administrative_block(lines: list[str]) -> bool:
    admin_count = sum(bool(_ADMIN_LINE.match(line)) for line in lines)
    if admin_count == len(lines):
        return True
    if admin_count >= 2 and admin_count / len(lines) >= 0.6:
        return True
    return (
        len(lines) <= 2
        and admin_count == len(lines)
        and any(
            re.match(r"^(?:status|tags?|aliases|created|updated|modified)\s*:", line, re.IGNORECASE)
            for line in lines
        )
    )


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))
