"""Extractor protocol and shared section helpers."""

from pathlib import Path
from typing import Protocol

from lecture_slm.ingestion.models import DocumentType, NormalizedDocument


class Extractor(Protocol):
    """Contract implemented by each supported file parser."""

    name: str
    version: str
    document_type: DocumentType

    def extract(
        self,
        path: Path,
        *,
        source_id: str,
        source_hash: str,
        relative_path: str,
        source_version: int = 1,
    ) -> NormalizedDocument: ...


def clean_lines(text: str) -> str:
    """Normalize line endings without rewriting source wording."""

    return text.replace("\r\n", "\n").replace("\r", "\n").strip()
