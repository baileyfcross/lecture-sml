"""Deterministic Markdown extraction without executing or following links."""

import re
from pathlib import Path

from lecture_slm.ingestion.extractors.base import clean_lines
from lecture_slm.ingestion.models import (
    DocumentType,
    ExtractionStatus,
    NormalizedDocument,
    NormalizedSection,
    ProvenanceRecord,
)

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class MarkdownExtractor:
    name = "markdown"
    version = "1"
    document_type = DocumentType.MARKDOWN

    def extract(
        self,
        path: Path,
        *,
        source_id: str,
        source_hash: str,
        relative_path: str,
        source_version: int = 1,
    ) -> NormalizedDocument:
        lines = path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
        sections: list[NormalizedSection] = []
        current: list[str] = []
        current_title: str | None = None
        current_type = "paragraph"
        current_level = 0
        order = 0

        def flush() -> None:
            nonlocal current, order
            text = clean_lines("\n".join(current))
            if text:
                sections.append(
                    NormalizedSection(
                        section_id=f"md-{order + 1}",
                        section_type=current_type,
                        title=current_title,
                        order=order,
                        text=text,
                        hierarchy_level=current_level,
                        source_location=f"markdown:{order + 1}",
                    )
                )
                order += 1
            current = []

        in_fence = False
        for line in lines:
            if line.strip().startswith("```"):
                if not in_fence:
                    flush()
                    current_type = "code_block"
                    current_level = 0
                current.append(line)
                in_fence = not in_fence
                if not in_fence:
                    flush()
                    current_type = "paragraph"
                continue
            heading = None if in_fence else _HEADING.match(line)
            if heading:
                flush()
                current_title = heading.group(2)
                current_type = "heading"
                current_level = len(heading.group(1))
                current.append(line)
                flush()
                current_title = None
                current_type = "paragraph"
                current_level = 0
            elif not in_fence and not line.strip():
                flush()
            else:
                current.append(line)
        flush()

        warnings = ["File has no extractable text"] if not sections else []
        return NormalizedDocument(
            source_id=source_id,
            title=next((s.title for s in sections if s.section_type == "heading"), path.stem),
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
