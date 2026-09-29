"""Deterministic teaching-material ingestion and human review workflows."""

from lecture_slm.ingestion.models import (
    Authorship,
    CandidateTrainingExample,
    DocumentType,
    ExtractionStatus,
    NormalizedDocument,
    ReviewStatus,
    SourceManifest,
)

__all__ = [
    "Authorship",
    "CandidateTrainingExample",
    "DocumentType",
    "ExtractionStatus",
    "NormalizedDocument",
    "ReviewStatus",
    "SourceManifest",
]
