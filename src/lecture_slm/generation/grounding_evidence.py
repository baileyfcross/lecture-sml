"""Deterministic evidence ledger construction for grounding review."""

from __future__ import annotations

import re

from lecture_slm.generation.models import EvidenceSpan, GenerationRequest


def _evidence_sentence_spans(text: str) -> list[str]:
    spans: list[str] = []
    in_fence = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line or line.startswith("#"):
            continue
        line = re.sub(r"^(?:[-*+]\s+|\d+\.\s+)", "", line)
        line = re.sub(r"\b(?:e\.g|i\.e)\.", lambda match: match.group().replace(".", "<DOT>"), line)
        parts = re.split(r'(?<=[.!?])["”’)]*\s+(?=[A-Z0-9“"($\[])', line)
        for part in parts:
            span = part.replace("<DOT>", ".").strip()
            if span and any(character.isalpha() for character in span):
                spans.append(span)
    return spans


def build_evidence_ledger(request: GenerationRequest) -> list[EvidenceSpan]:
    """Create stable evidence IDs for the supplied ordered source material."""

    ledger: list[EvidenceSpan] = []
    for source_index, source in enumerate(request.source_material, start=1):
        spans = _evidence_sentence_spans(source.text)
        if not spans:
            spans = [source.text.strip()]
        for span_index, span in enumerate(spans, start=1):
            ledger.append(
                EvidenceSpan(
                    evidence_id=f"S{source_index:02d}-E{span_index:03d}",
                    source_id=source.source_id,
                    source_title=source.title,
                    text=span,
                    order=len(ledger) + 1,
                )
            )
    return ledger
