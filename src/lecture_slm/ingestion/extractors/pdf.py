"""PDF text extraction without OCR."""

from pathlib import Path

from pypdf import PdfReader

from lecture_slm.ingestion.models import (
    DocumentType,
    ExtractionStatus,
    NormalizedDocument,
    NormalizedSection,
    ProvenanceRecord,
)


class PdfExtractor:
    name = "pypdf"
    version = "1"
    document_type = DocumentType.PDF

    def extract(
        self,
        path: Path,
        *,
        source_id: str,
        source_hash: str,
        relative_path: str,
        source_version: int = 1,
    ) -> NormalizedDocument:
        reader = PdfReader(path)
        sections: list[NormalizedSection] = []
        warnings: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append(
                    NormalizedSection(
                        section_id=f"page-{page_number}",
                        section_type="page",
                        order=page_number - 1,
                        text=text,
                        page_number=page_number,
                        source_location=f"page:{page_number}",
                    )
                )
            else:
                warnings.append(f"Page {page_number} has no extractable text; OCR is not enabled")
        if not sections:
            warnings.append(
                "PDF has little or no extractable text; manual/OCR handling is required"
            )
        quality = ExtractionStatus.SUCCESS_WITH_WARNINGS if warnings else ExtractionStatus.SUCCESS
        return NormalizedDocument(
            source_id=source_id,
            title=path.stem,
            document_type=self.document_type,
            sections=sections,
            extraction_warnings=warnings,
            extraction_quality=quality,
            provenance=ProvenanceRecord(
                source_file=relative_path,
                source_hash=source_hash,
                source_version=source_version,
                extraction_method=self.name,
            ),
        )
