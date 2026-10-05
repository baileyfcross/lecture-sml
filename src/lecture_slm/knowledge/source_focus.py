"""Deterministic source-category hints used as small retrieval tie-breakers."""

from enum import StrEnum


class SourceFocusKind(StrEnum):
    FOCUSED_NOTE = "focused_note"
    OVERVIEW_TAG = "overview_tag"
    NEUTRAL = "neutral"


def classify_source_focus(
    relative_path: str,
    *,
    focused_note_directories: list[str],
    overview_tag_directories: list[str],
) -> SourceFocusKind:
    """Classify a source using directory components of its relative path."""

    directories = {
        part.strip().casefold()
        for part in relative_path.replace("\\", "/").split("/")[:-1]
        if part.strip()
    }
    if directories.intersection(name.strip().casefold() for name in focused_note_directories):
        return SourceFocusKind.FOCUSED_NOTE
    if directories.intersection(name.strip().casefold() for name in overview_tag_directories):
        return SourceFocusKind.OVERVIEW_TAG
    return SourceFocusKind.NEUTRAL
