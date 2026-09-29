"""Conservative deterministic artifact and optional course classification."""

import re
from dataclasses import dataclass
from pathlib import Path

from lecture_slm.ingestion.models import DocumentType, NormalizedDocument
from lecture_slm.schemas.dataset import TaskType


@dataclass(frozen=True)
class Classification:
    task: TaskType | None
    confidence: float
    reason: str


def classify_artifact(path: Path, document: NormalizedDocument) -> Classification:
    """Classify only strong filename/title signals; otherwise preserve unknown."""

    haystack = " ".join(
        [
            path.stem.lower(),
            document.title or "",
            *(section.title or "" for section in document.sections),
        ]
    )
    rules: tuple[tuple[TaskType, tuple[str, ...]], ...] = (
        (TaskType.LAB, ("lab", "laboratory")),
        (TaskType.INSTRUCTOR_GUIDE, ("instructor guide", "teacher guide", "answer key")),
        (TaskType.ASSESSMENT, ("assessment", "exam", "quiz", "test")),
        (TaskType.HOMEWORK, ("homework", "assignment")),
        (TaskType.ACTIVITY, ("activity", "exercise", "worksheet")),
        (TaskType.LECTURE, ("lecture", "lesson", "module")),
        (TaskType.EXPLANATION, ("explanation", "concept note")),
    )
    if document.document_type is DocumentType.PPTX:
        return Classification(TaskType.SLIDES, 0.98, "PowerPoint source")
    for task, terms in rules:
        if any(re.search(rf"\b{re.escape(term)}\b", haystack) for term in terms):
            return Classification(task, 0.9, "strong filename or heading signal")
    return Classification(None, 0.0, "no strong deterministic artifact signal")
