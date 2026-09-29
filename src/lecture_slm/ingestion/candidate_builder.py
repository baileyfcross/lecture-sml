"""Conservative candidate construction from normalized source documents."""

from pathlib import Path

from lecture_slm.ingestion.classify import classify_artifact
from lecture_slm.ingestion.models import (
    Authorship,
    CandidateTrainingExample,
    InstructionSource,
    NormalizedDocument,
    SourceLocation,
)


def format_source(document: NormalizedDocument) -> str:
    """Create a stable semantic representation without rewriting source text."""

    chunks: list[str] = []
    for section in sorted(document.sections, key=lambda item: item.order):
        if section.slide_number is not None:
            heading = f"Slide {section.slide_number}"
            if section.title:
                heading += f": {section.title}"
            chunks.append(f"{heading}\n{section.text}".strip())
        elif section.title:
            chunks.append(f"{section.title}\n{section.text}".strip())
        else:
            chunks.append(section.text.strip())
    return "\n\n".join(chunk for chunk in chunks if chunk)


def build_candidate_from_path(
    document: NormalizedDocument,
    source_path: str,
    *,
    course: str | None = None,
    level: str | None = None,
) -> CandidateTrainingExample | None:
    classification = classify_artifact(Path(source_path), document)
    if classification.task is None or not document.text:
        return None
    locations = [
        SourceLocation(
            section_id=section.section_id,
            source_location=section.source_location,
            page_number=section.page_number,
            slide_number=section.slide_number,
        )
        for section in document.sections
    ]
    instruction = (
        f"Create a {classification.task.value} artifact from the supplied instructor material."
    )
    return CandidateTrainingExample(
        candidate_id=f"candidate-{document.source_id}-artifact",
        task=classification.task,
        source_ids=[document.source_id],
        source_locations=locations,
        course=course,
        level=level,
        instruction=instruction,
        instruction_source=InstructionSource.RECONSTRUCTED,
        input_material="",
        expected_output=format_source(document),
        provenance=[document.provenance],
        confidence=classification.confidence,
        authorship=Authorship.UNKNOWN,
        metadata={"classification_reason": classification.reason},
    )
