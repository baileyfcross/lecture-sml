"""Deterministic section-aware chunking of normalized documents."""

import hashlib
import re
from collections.abc import Iterable
from typing import Any

from lecture_slm.generation.context import estimate_tokens_from_characters
from lecture_slm.ingestion.models import NormalizedDocument, NormalizedSection
from lecture_slm.knowledge.config import ChunkingConfig
from lecture_slm.knowledge.models import KnowledgeChunk
from lecture_slm.knowledge.obsidian import remove_frontmatter
from lecture_slm.knowledge.roles import classify_chunk_role

CHUNKING_VERSION = "2"
_WORD = re.compile(r"\S+")


def chunk_document(
    document: NormalizedDocument,
    *,
    relative_path: str,
    source_version: int,
    embedding_model: str,
    embedding_version: str,
    config: ChunkingConfig,
    note_metadata: dict[str, Any] | None = None,
) -> list[KnowledgeChunk]:
    """Chunk normalized sections without crossing source/page/slide boundaries."""

    metadata = note_metadata or {}
    title = str(metadata.get("title") or document.title or relative_path.rsplit("/", 1)[-1])
    tags = [str(tag) for tag in metadata.get("tags", [])]
    aliases = [str(alias) for alias in metadata.get("aliases", [])]
    links = [
        str(link["target"])
        for link in metadata.get("outgoing_links", [])
        if isinstance(link, dict) and link.get("target")
    ]
    course = metadata.get("course")
    groups: list[tuple[NormalizedSection, list[str], str | None, str]] = []
    heading_stack: list[tuple[int, str]] = []
    pending_sections: list[NormalizedSection] = []
    pending_path: list[str] = []
    pending_title: str | None = None
    pending_key: tuple[Any, ...] | None = None

    def flush_group() -> None:
        nonlocal pending_sections, pending_path, pending_title, pending_key
        if pending_sections:
            source_section = pending_sections[0]
            groups.append(
                (
                    source_section,
                    pending_path,
                    pending_title,
                    "\n\n".join(section.text.strip() for section in pending_sections),
                )
            )
        pending_sections = []
        pending_path = []
        pending_title = None
        pending_key = None

    for section in document.sections:
        if section.section_type == "heading":
            heading = section.title or section.text.lstrip("#").strip()
            heading_stack = [
                (level, value) for level, value in heading_stack if level < section.hierarchy_level
            ]
            if heading:
                heading_stack.append((section.hierarchy_level, heading))
            continue
        section_text = section.text
        if document.document_type.value == "markdown":
            section_text = remove_frontmatter(section_text)
        if not section_text.strip():
            continue
        section_path = [name for _, name in heading_stack]
        section_title = section_path[-1] if section_path else section.title
        boundary_path = tuple(section_path) if config.preserve_sections else ()
        boundary = (boundary_path, section.page_number, section.slide_number)
        if pending_key is not None and pending_key != boundary:
            flush_group()
        if not pending_sections:
            pending_path = section_path
            pending_title = section_title
            pending_key = boundary
        if (
            not config.preserve_sections
            and pending_sections
            and section_path
            and section_path != pending_path
        ):
            pending_sections.append(
                section.model_copy(update={"text": section_path[-1], "section_type": "heading"})
            )
        pending_sections.append(section.model_copy(update={"text": section_text}))
    flush_group()

    chunks: list[KnowledgeChunk] = []
    for section, section_path, section_title, section_text in groups:
        parts = _split_text(section_text, config)
        for part in parts:
            index = len(chunks)
            identity = "\0".join(
                (
                    document.source_id,
                    section.section_id,
                    str(index),
                    str(section.page_number or ""),
                    str(section.slide_number or ""),
                    part,
                )
            )
            chunk_id = "chunk-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    source_id=document.source_id,
                    source_hash=document.provenance.source_hash,
                    source_version=source_version,
                    title=title,
                    section_title=section_title,
                    section_path=section_path,
                    text=part,
                    chunk_index=index,
                    page_number=section.page_number,
                    slide_number=section.slide_number,
                    note_path=relative_path,
                    document_type=document.document_type.value,
                    role=classify_chunk_role(
                        " > ".join(section_path) or section_title,
                        part,
                        document_type=document.document_type.value,
                    ),
                    tags=tags,
                    aliases=aliases,
                    outgoing_links=links,
                    course=str(course) if course is not None else None,
                    metadata={
                        **section.metadata,
                        "section_id": section.section_id,
                        "section_type": section.section_type,
                        "source_location": section.source_location,
                        **{
                            key: value
                            for key, value in metadata.items()
                            if key not in {"outgoing_links", "tags", "aliases"}
                        },
                    },
                    approximate_token_count=estimate_tokens_from_characters(part),
                    embedding_model=embedding_model,
                    embedding_version=embedding_version,
                )
            )

    for index, chunk in enumerate(chunks):
        chunk.previous_chunk_id = chunks[index - 1].chunk_id if index else None
        chunk.next_chunk_id = chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
    return chunks


def _split_text(text: str, config: ChunkingConfig) -> list[str]:
    words = list(_WORD.finditer(text))
    if not words:
        return []
    target = config.target_tokens
    max_tokens = config.max_tokens
    overlap_chars = config.overlap_tokens * 4
    output: list[str] = []
    start = 0
    while start < len(words):
        end = start
        candidate = ""
        while end < len(words):
            next_text = text[words[start].start() : words[end].end()].strip()
            if end > start and estimate_tokens_from_characters(next_text) > max_tokens:
                break
            candidate = next_text
            end += 1
            if estimate_tokens_from_characters(candidate) >= target:
                break
        if not candidate:
            candidate = words[start].group()
            end = start + 1
        output.append(candidate)
        if end >= len(words):
            break
        next_start = end
        if overlap_chars:
            while next_start > start + 1:
                span_size = words[end - 1].end() - words[next_start - 1].start()
                if span_size > overlap_chars:
                    break
                next_start -= 1
        start = next_start if next_start > start else end
    return output


def section_path_for_chunk(chunk: KnowledgeChunk) -> str:
    """Format the hierarchy for display and exact section matching."""

    return " > ".join(chunk.section_path) or chunk.section_title or ""


def chunk_texts(chunks: Iterable[KnowledgeChunk]) -> list[str]:
    return [chunk.text for chunk in chunks]
