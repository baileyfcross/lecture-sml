"""Conservative plain-text extraction."""

from pathlib import Path

from lecture_slm.ingestion.extractors.base import clean_lines
from lecture_slm.ingestion.models import (
    DocumentType,
    ExtractionStatus,
    NormalizedDocument,
    NormalizedSection,
    ProvenanceRecord,
)


class TextExtractor:
    name = "text"
    version = "1"
    document_type = DocumentType.TEXT

    def extract(
        self,
        path: Path,
        *,
        source_id: str,
        source_hash: str,
        relative_path: str,
        source_version: int = 1,
    ) -> NormalizedDocument:
        text = clean_lines(path.read_text(encoding="utf-8"))
        sections = (
            [
                NormalizedSection(
                    section_id="text-1",
                    section_type="paragraph",
                    order=0,
                    text=text,
                    source_location="text",
                )
            ]
            if text
            else []
        )
        warnings = ["File has no extractable text"] if not text else []
        status = ExtractionStatus.SUCCESS_WITH_WARNINGS if warnings else ExtractionStatus.SUCCESS
        return NormalizedDocument(
            source_id=source_id,
            title=path.stem,
            document_type=self.document_type,
            sections=sections,
            extraction_warnings=warnings,
            extraction_quality=status,
            provenance=ProvenanceRecord(
                source_file=relative_path,
                source_hash=source_hash,
                source_version=source_version,
                extraction_method=self.name,
            ),
        )
