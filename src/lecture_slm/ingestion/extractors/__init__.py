"""Type-specific deterministic document extractors."""

from lecture_slm.ingestion.extractors.base import Extractor
from lecture_slm.ingestion.extractors.docx import DocxExtractor
from lecture_slm.ingestion.extractors.markdown import MarkdownExtractor
from lecture_slm.ingestion.extractors.pdf import PdfExtractor
from lecture_slm.ingestion.extractors.pptx import PptxExtractor
from lecture_slm.ingestion.extractors.text import TextExtractor

__all__ = [
    "DocxExtractor",
    "Extractor",
    "MarkdownExtractor",
    "PdfExtractor",
    "PptxExtractor",
    "TextExtractor",
]
