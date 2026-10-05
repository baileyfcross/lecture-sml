"""Typed records for the local knowledge index and retrieval pipeline."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_hash: str = Field(min_length=64, max_length=64)
    source_version: int = Field(ge=1)
    title: str = Field(min_length=1)
    section_title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    slide_number: int | None = Field(default=None, ge=1)
    note_path: str = Field(min_length=1)
    document_type: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    outgoing_links: list[str] = Field(default_factory=list)
    course: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    approximate_token_count: int = Field(gt=0)
    embedding_model: str = Field(min_length=1)
    embedding_version: str = Field(min_length=1)


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    mode: Literal["lexical", "semantic", "hybrid"] = "hybrid"
    source_title: str | None = None
    source_id: str | None = None
    section: str | None = None
    course: str | None = None
    tags: list[str] = Field(default_factory=list)
    document_types: list[str] = Field(default_factory=list)
    folder: str | None = None
    page: int | None = Field(default=None, ge=1)
    top_k: int | None = Field(default=None, gt=0)
    lexical_k: int | None = Field(default=None, gt=0)
    semantic_k: int | None = Field(default=None, gt=0)
    rrf_constant: int | None = Field(default=None, gt=0)
    neighbor_expansion: int | None = Field(default=None, ge=0)
    rerank: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_id: str
    source_title: str
    source_path: str
    section: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_number: int | None = None
    slide_number: int | None = None
    text: str
    lexical_rank: int | None = None
    lexical_score: float | None = None
    semantic_rank: int | None = None
    semantic_score: float | None = None
    fused_rank: int
    fused_score: float
    rerank_rank: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    neighbor_of: str | None = None


class RetrievalTimings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_resolution_seconds: float = 0.0
    lexical_search_seconds: float = 0.0
    query_embedding_seconds: float = 0.0
    semantic_search_seconds: float = 0.0
    fusion_seconds: float = 0.0
    rerank_seconds: float = 0.0
    total_seconds: float = 0.0


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    resolved_source: dict[str, Any] | None = None
    resolved_section: str | None = None
    matches: list[RetrievalMatch] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    timings: RetrievalTimings = Field(default_factory=RetrievalTimings)
    retrieval_configuration: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class IndexMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files_scanned: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    deleted: int = 0
    duplicates: int = 0
    failed: int = 0
    normalized: int = 0
    chunks_created: int = 0
    chunks_reused: int = 0
    embeddings_generated: int = 0
    embeddings_reused: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
