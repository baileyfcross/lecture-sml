"""DOCX extraction preserving document order, headings, lists, and tables."""

from pathlib import Path

from docx import Document

from lecture_slm.ingestion.models import (
    DocumentType,
    ExtractionStatus,
    NormalizedDocument,
    NormalizedSection,
    ProvenanceRecord,
)


class DocxExtractor:
    name = "python-docx"
    version = "1"
    document_type = DocumentType.DOCX

    def extract(
        self,
        path: Path,
        *,
        source_id: str,
        source_hash: str,
        relative_path: str,
        source_version: int = 1,
    ) -> NormalizedDocument:
        document = Document(str(path))
        sections: list[NormalizedSection] = []
        warnings: list[str] = []
        order = 0
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            style = paragraph.style.name.lower() if paragraph.style else ""
            is_heading = style.startswith("heading")
            level = (
                int(style.rsplit(" ", 1)[-1])
                if is_heading and style.rsplit(" ", 1)[-1].isdigit()
                else 0
            )
            section_type = "heading" if is_heading else "paragraph"
            if "list" in style:
                section_type = "list_item"
            sections.append(
                NormalizedSection(
                    section_id=f"docx-{order + 1}",
                    section_type=section_type,
                    title=text if is_heading else None,
                    order=order,
                    text=text,
                    hierarchy_level=level,
                    source_location=f"paragraph:{order + 1}",
                )
            )
            order += 1
        for table_index, table in enumerate(document.tables, start=1):
            rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
            text = "\n".join(rows).strip()
            if text:
                sections.append(
                    NormalizedSection(
                        section_id=f"docx-table-{table_index}",
                        section_type="table",
                        order=order,
                        text=text,
                        source_location=f"table:{table_index}",
                    )
                )
                order += 1
        if not sections:
            warnings.append("File has no extractable text")
        return NormalizedDocument(
            source_id=source_id,
            title=path.stem,
            document_type=self.document_type,
            sections=sections,
            extraction_warnings=warnings,
            extraction_quality=(
                ExtractionStatus.SUCCESS_WITH_WARNINGS if warnings else ExtractionStatus.SUCCESS
            ),
            provenance=ProvenanceRecord(
                source_file=relative_path,
                source_hash=source_hash,
                source_version=source_version,
                extraction_method=self.name,
            ),
        )
