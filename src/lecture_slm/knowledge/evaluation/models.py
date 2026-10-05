"""Typed retrieval-evaluation inputs and reviewer annotations."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lecture_slm.knowledge.roles import ChunkRole

RelevanceLabel = Literal["highly_relevant", "relevant", "partially_relevant", "irrelevant"]
SufficiencyLabel = Literal["sufficient", "partially_sufficient", "insufficient"]
FailureCategory = Literal[
    "source_not_found",
    "source_ambiguous",
    "section_not_found",
    "relevant_source_not_retrieved",
    "relevant_chunk_ranked_too_low",
    "lexical_noise",
    "semantic_noise",
    "wrong_course",
    "insufficient_context",
    "chunk_boundary_problem",
    "extraction_problem",
    "metadata_problem",
]


class RetrievalEvalCase(BaseModel):
    """One manually authored retrieval evaluation query."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    category: str = Field(min_length=1)
    source_title: str | None = None
    source_id: str | None = None
    section: str | None = None
    course: str | None = None
    tags: list[str] = Field(default_factory=list)
    document_types: list[str] = Field(default_factory=list)
    passage_roles: list[ChunkRole] | None = None
    folder: str | None = None
    page: int | None = Field(default=None, ge=1)
    expected_source_ids: list[str] = Field(default_factory=list)
    expected_source_titles: list[str] = Field(default_factory=list)
    expected_sections: list[str] = Field(default_factory=list)
    expected_pages: list[int] = Field(default_factory=list)
    notes: str | None = None
    version: str = "1"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalReview(BaseModel):
    """Human assessment for one returned passage and assembled context."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    mode: str
    chunk_id: str
    relevance: RelevanceLabel | None = None
    context_sufficiency: SufficiencyLabel | None = None
    failure_category: FailureCategory | None = None
    notes: str = ""
