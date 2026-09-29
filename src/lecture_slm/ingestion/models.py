"""Schemas for source manifests, normalized documents, and review candidates."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lecture_slm.schemas.dataset import QualityTier, TaskType


class ExtractionStatus(StrEnum):
    SUCCESS = "success"
    SUCCESS_WITH_WARNINGS = "success_with_warnings"
    PARTIAL = "partial"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    UNCHANGED = "unchanged"
    DUPLICATE = "duplicate"


class DocumentType(StrEnum):
    PPTX = "pptx"
    DOCX = "docx"
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    UNKNOWN = "unknown"


class Authorship(StrEnum):
    INSTRUCTOR_CREATED = "instructor_created"
    INSTRUCTOR_EDITED = "instructor_edited"
    AI_ASSISTED_HUMAN_REVIEWED = "ai_assisted_human_reviewed"
    SYNTHETIC_UNREVIEWED = "synthetic_unreviewed"
    UNKNOWN = "unknown"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_EDIT = "needs_edit"
    DUPLICATE = "duplicate"
    DEFERRED = "deferred"
    NEEDS_INSTRUCTION_REVIEW = "needs_instruction_review"


class InstructionSource(StrEnum):
    KNOWN_ORIGINAL = "known_original_instruction"
    RECOVERED = "recovered_instruction"
    RECONSTRUCTED = "reconstructed_instruction"
    MANUAL = "manually_entered_instruction"
    UNKNOWN = "unknown"


class SourceLocation(BaseModel):
    """Location of extracted text in its original document."""

    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    slide_number: int | None = Field(default=None, ge=1)


class NormalizedSection(BaseModel):
    """A semantically structured portion of an extracted source."""

    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(min_length=1)
    section_type: str = Field(min_length=1)
    title: str | None = None
    order: int = Field(ge=0)
    text: str = ""
    hierarchy_level: int = Field(default=0, ge=0)
    page_number: int | None = Field(default=None, ge=1)
    slide_number: int | None = Field(default=None, ge=1)
    source_location: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProvenanceRecord(BaseModel):
    """Ingestion provenance retained by documents and candidate examples."""

    model_config = ConfigDict(extra="forbid")

    source_file: str = Field(min_length=1)
    source_hash: str = Field(min_length=64, max_length=64)
    source_version: int = Field(default=1, ge=1)
    extraction_method: str = Field(min_length=1)
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_locations: list[SourceLocation] = Field(default_factory=list)
    authorship: Authorship = Authorship.UNKNOWN
    human_reviewed: bool = False
    review_status: ReviewStatus = ReviewStatus.PENDING


class NormalizedDocument(BaseModel):
    """Common representation returned by every supported extractor."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    title: str | None = None
    document_type: DocumentType
    sections: list[NormalizedSection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    extraction_warnings: list[str] = Field(default_factory=list)
    extraction_quality: ExtractionStatus
    provenance: ProvenanceRecord

    @property
    def text(self) -> str:
        """Return source text in original section order."""

        return "\n\n".join(section.text for section in self.sections if section.text.strip())


class SourceManifest(BaseModel):
    """One source/version record in the persistent ingestion manifest."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    extension: str = Field(min_length=1)
    file_size: int = Field(ge=0)
    modified_time: datetime
    sha256: str = Field(min_length=64, max_length=64)
    extractor: str = Field(min_length=1)
    extractor_version: str = Field(min_length=1)
    extraction_status: ExtractionStatus
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    imported_at: datetime | None = None
    document_type: DocumentType = DocumentType.UNKNOWN
    course: str | None = None
    provenance: ProvenanceRecord | None = None
    notes: list[str] = Field(default_factory=list)
    normalized_path: str | None = None
    previous_source_id: str | None = None
    canonical_source_id: str | None = None


class ReviewEvent(BaseModel):
    """Append-only status transition metadata for a candidate."""

    model_config = ConfigDict(extra="forbid")

    status: ReviewStatus
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reviewer: str | None = None
    notes: str | None = None


class CandidateTrainingExample(BaseModel):
    """A reviewable example that is not training data until explicitly approved."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    task: TaskType | None = None
    source_ids: list[str] = Field(min_length=1)
    source_locations: list[SourceLocation] = Field(default_factory=list)
    course: str | None = None
    level: str | None = None
    instruction: str | None = None
    instruction_source: InstructionSource = InstructionSource.UNKNOWN
    context: str | None = None
    input_material: str = ""
    expected_output: str = ""
    provenance: list[ProvenanceRecord] = Field(min_length=1)
    quality_tier: QualityTier = QualityTier.B
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewer_notes: str | None = None
    authorship: Authorship = Authorship.UNKNOWN
    pedagogy_tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status_history: list[ReviewEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def transition(
        self,
        status: ReviewStatus,
        *,
        reviewer: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Apply an explicit review transition while preserving its history."""

        if status is ReviewStatus.APPROVED:
            self.quality_tier = (
                QualityTier.C
                if self.authorship is Authorship.AI_ASSISTED_HUMAN_REVIEWED
                else QualityTier.A
            )
        self.review_status = status
        self.reviewer_notes = notes if notes is not None else self.reviewer_notes
        self.updated_at = datetime.now(UTC)
        self.status_history.append(ReviewEvent(status=status, reviewer=reviewer, notes=notes))
