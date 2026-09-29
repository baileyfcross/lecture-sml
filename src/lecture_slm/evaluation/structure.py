"""Heuristic checks for obvious response truncation or malformed structure."""

import re

from lecture_slm.schemas.dataset import TaskType


def inspect_response_structure(
    *,
    task: TaskType,
    instruction: str,
    response: str,
    potentially_truncated: bool,
) -> tuple[bool, list[str]]:
    """Return a structural-completeness flag and explicit heuristic observations."""

    observations: list[str] = []
    text = response.strip()
    lowered = text.lower()
    if not text:
        observations.append("empty final answer")
    if potentially_truncated:
        observations.append("generation reached the configured output limit")
    if text and (
        text.endswith((":", ",", ";", " -"))
        or re.search(r"\b(and|or|with|to|that|such as)$", text, flags=re.IGNORECASE)
    ):
        observations.append("response ends with a possible unfinished sentence or list item")

    requested = instruction.lower()
    if task is TaskType.SLIDES and "slide" in requested:
        slide_sections = re.findall(
            r"(?im)^\s*(?:#{1,6}\s*)?slide\s+(?:\d+|[ivxlcdm]+|[a-z])\b",
            text,
        )
        if len(slide_sections) < 2:
            observations.append("fewer than two recognizable slide headings")
    if task is TaskType.LAB:
        for requested_term, evidence_terms in (
            ("checkpoint", ("checkpoint", "check for understanding")),
            ("extension", ("extension", "challenge task")),
            ("expected observation", ("expected observation", "expected result")),
        ):
            if requested_term in requested and not any(term in lowered for term in evidence_terms):
                observations.append(f"requested lab element not found: {requested_term}")

    return not observations, observations
