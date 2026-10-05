"""Lexical, dense, and deterministic hybrid knowledge retrieval."""

import json
import time
from typing import Any, Protocol

from lecture_slm.knowledge.config import RetrievalConfig
from lecture_slm.knowledge.embeddings import EmbeddingProvider
from lecture_slm.knowledge.models import (
    RetrievalMatch,
    RetrievalRequest,
    RetrievalResult,
    RetrievalTimings,
)
from lecture_slm.knowledge.ranking import reciprocal_rank_fusion
from lecture_slm.knowledge.resolution import SourceResolver, normalize_title
from lecture_slm.knowledge.storage import KnowledgeStore
from lecture_slm.knowledge.vector import SQLiteCosineSearch, VectorSearch


class Reranker(Protocol):
    """Optional reranker contract, applied only to the fused shortlist."""

    def rerank(
        self, query: str, candidates: list[RetrievalMatch], limit: int
    ) -> list[RetrievalMatch]: ...


class KnowledgeRetriever:
    """Return provenance-preserving candidates with diagnostic retrieval signals."""

    def __init__(
        self,
        store: KnowledgeStore,
        embeddings: EmbeddingProvider,
        config: RetrievalConfig,
        *,
        reranker: Reranker | None = None,
        vector_search: VectorSearch | None = None,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.config = config
        self.reranker = reranker
        self.vector_search = vector_search or SQLiteCosineSearch(store)
        self.resolver = SourceResolver(store)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        total_started = time.perf_counter()
        warnings: list[str] = []
        resolution_started = time.perf_counter()
        source, resolution_warnings = self.resolver.resolve_source(
            request.query, request.source_title
        )
        warnings.extend(resolution_warnings)
        if request.source_id:
            row = next(
                (
                    item
                    for item in self.store.source_records()
                    if str(item["source_id"]) == request.source_id
                ),
                None,
            )
            if row is None:
                warnings.append(f"Requested source ID is not indexed: {request.source_id}")
                source = None
            else:
                source = {
                    "source_id": request.source_id,
                    "title": str(row["title"]),
                    "relative_path": str(row["relative_path"] or ""),
                    "paths": str(row["paths"] or "").splitlines(),
                    "resolution_reason": "explicit source ID",
                }
        resolved_source_id = str(source["source_id"]) if source else None
        if (request.source_id or request.source_title) and source is None:
            return RetrievalResult(
                query=request.query,
                resolved_source=None,
                matches=[],
                warnings=warnings,
                filters_applied={
                    "source_id": request.source_id,
                    "source_title": request.source_title,
                },
                timings=RetrievalTimings(
                    source_resolution_seconds=time.perf_counter() - resolution_started,
                    total_seconds=time.perf_counter() - total_started,
                ),
                retrieval_configuration={
                    "mode": request.mode,
                    "lexical_candidates": request.lexical_k or self.config.lexical_candidates,
                    "semantic_candidates": request.semantic_k or self.config.semantic_candidates,
                    "fused_candidates": self.config.fused_candidates,
                    "rrf_constant": self.config.rrf_constant,
                    "reranking_enabled": False,
                },
                diagnostics={
                    "source_resolution": {
                        "requested_title": request.source_title,
                        "requested_id": request.source_id,
                        "resolved": None,
                        "warnings": warnings,
                    },
                    "section_resolution": {
                        "requested": request.section,
                        "resolved": None,
                        "resolved_chunk_count": 0,
                    },
                    "lexical_candidates": [],
                    "semantic_candidates": [],
                    "fused_candidates": [],
                    "neighbor_additions": [],
                },
            )
        section, section_ids, section_warning = self.resolver.resolve_section(
            resolved_source_id, request.section, request.query
        )
        if section_warning:
            warnings.append(section_warning)
        if section_warning:
            return RetrievalResult(
                query=request.query,
                resolved_source=source,
                resolved_section=None,
                matches=[],
                warnings=warnings,
                filters_applied={
                    "source_id": request.source_id or resolved_source_id,
                    "source_title": request.source_title,
                    "section": request.section,
                    "course": request.course,
                    "tags": request.tags,
                },
                timings=RetrievalTimings(
                    source_resolution_seconds=time.perf_counter() - resolution_started,
                    total_seconds=time.perf_counter() - total_started,
                ),
                retrieval_configuration={
                    "mode": request.mode,
                    "lexical_candidates": request.lexical_k or self.config.lexical_candidates,
                    "semantic_candidates": request.semantic_k or self.config.semantic_candidates,
                    "fused_candidates": self.config.fused_candidates,
                    "rrf_constant": request.rrf_constant or self.config.rrf_constant,
                    "reranking_enabled": False,
                },
                diagnostics={
                    "source_resolution": {
                        "requested_title": request.source_title,
                        "requested_id": request.source_id,
                        "resolved": source,
                        "warnings": resolution_warnings,
                    },
                    "section_resolution": {
                        "requested": request.section,
                        "resolved": None,
                        "resolved_chunk_count": 0,
                        "warning": section_warning,
                    },
                    "lexical_candidates": [],
                    "semantic_candidates": [],
                    "fused_candidates": [],
                    "neighbor_additions": [],
                },
            )
        if (request.rerank or self.config.reranker_enabled) and self.reranker is None:
            raise ValueError("Reranking was requested but no reranker provider is configured")
        resolution_seconds = time.perf_counter() - resolution_started

        filters = {
            "source_id": request.source_id or resolved_source_id,
            "source_title": request.source_title,
            "section": request.section or section,
            "course": request.course,
            "tags": request.tags,
            "document_types": request.document_types,
            "folder": request.folder,
            "page": request.page,
        }
        filter_source_id = request.source_id or resolved_source_id

        lexical_started = time.perf_counter()
        lexical_rows = (
            self.store.search_lexical(
                request.query,
                limit=request.lexical_k or self.config.lexical_candidates,
                source_id=filter_source_id,
            )
            if request.mode in {"lexical", "hybrid"}
            else []
        )
        lexical_seconds = time.perf_counter() - lexical_started

        embedding_seconds = 0.0
        semantic_seconds = 0.0
        semantic_rows: list[tuple[Any, float]] = []
        if request.mode in {"semantic", "hybrid"}:
            embedding_started = time.perf_counter()
            query_vector = self.embeddings.embed_query(request.query)
            if query_vector.size != self.embeddings.dimension:
                raise ValueError(
                    f"Query embedding dimension mismatch: expected {self.embeddings.dimension}, "
                    f"received {query_vector.size}"
                )
            embedding_seconds = time.perf_counter() - embedding_started

            semantic_started = time.perf_counter()
            semantic_rows = self.vector_search.search(
                query_vector,
                source_id=filter_source_id,
                limit=request.semantic_k or self.config.semantic_candidates,
            )
            semantic_seconds = time.perf_counter() - semantic_started

        candidates: dict[str, dict[str, Any]] = {}
        for rank, row in enumerate(lexical_rows, start=1):
            if not self._matches_filters(row, request):
                continue
            if section_ids and str(row["chunk_id"]) not in section_ids:
                continue
            candidates[str(row["chunk_id"])] = {
                "row": row,
                "lexical_rank": rank,
                "lexical_score": float(row["score"]),
                "semantic_rank": None,
                "semantic_score": None,
            }
        for rank, (row, score) in enumerate(semantic_rows, start=1):
            chunk_id = str(row["chunk_id"])
            if not self._matches_filters(row, request):
                continue
            if section_ids and chunk_id not in section_ids:
                continue
            candidate = candidates.setdefault(
                chunk_id,
                {
                    "row": row,
                    "lexical_rank": None,
                    "lexical_score": None,
                    "semantic_rank": None,
                    "semantic_score": None,
                },
            )
            candidate["semantic_rank"] = rank
            candidate["semantic_score"] = score

        fusion_started = time.perf_counter()
        fused: list[tuple[float, str, dict[str, Any]]] = []
        lexical_ranks = {
            chunk_id: int(candidate["lexical_rank"])
            for chunk_id, candidate in candidates.items()
            if candidate["lexical_rank"] is not None
        }
        semantic_ranks = {
            chunk_id: int(candidate["semantic_rank"])
            for chunk_id, candidate in candidates.items()
            if candidate["semantic_rank"] is not None
        }
        rrf_constant = request.rrf_constant or self.config.rrf_constant
        if request.mode == "hybrid":
            fused_scores = reciprocal_rank_fusion(
                lexical_ranks, semantic_ranks, constant=rrf_constant
            )
        elif request.mode == "lexical":
            fused_scores = {chunk_id: 1.0 / rank for chunk_id, rank in lexical_ranks.items()}
        else:
            fused_scores = {chunk_id: 1.0 / rank for chunk_id, rank in semantic_ranks.items()}
        lexical_diagnostics = [
            _candidate_diagnostic(chunk_id, candidates[chunk_id], rank, "lexical")
            for rank, chunk_id in enumerate(
                sorted(lexical_ranks, key=lambda item: (lexical_ranks[item], item)), start=1
            )
        ]
        semantic_diagnostics = [
            _candidate_diagnostic(chunk_id, candidates[chunk_id], rank, "semantic")
            for rank, chunk_id in enumerate(
                sorted(semantic_ranks, key=lambda item: (semantic_ranks[item], item)), start=1
            )
        ]
        boost_diagnostics: dict[str, dict[str, Any]] = {}
        for chunk_id, candidate in candidates.items():
            score = fused_scores[chunk_id]
            row = candidate["row"]
            boosts: list[str] = []
            source_boost = bool(resolved_source_id and row["source_id"] == resolved_source_id)
            if source_boost:
                score += self.config.exact_title_boost
                boosts.append("exact_source")
            section_path = " ".join(json.loads(row["section_path_json"]))
            section_boost = bool(
                section and normalize_title(section) in normalize_title(section_path)
            )
            if section_boost:
                score += self.config.exact_section_boost
                boosts.append("exact_section")
            resolution_reason = str((source or {}).get("resolution_reason", ""))
            boost_diagnostics[chunk_id] = {
                "source_resolution_boost": source_boost,
                "resolution_method": resolution_reason or None,
                "exact_title": source_boost
                and resolution_reason
                in {"exact title", "normalized title", "title named in query"},
                "exact_filename": source_boost
                and resolution_reason in {"exact filename", "filename named in query"},
                "alias": source_boost and resolution_reason == "alias",
                "section": section_boost,
                "course": False,
                "tag": False,
                "folder": False,
            }
            candidate["fused_score"] = score
            fused.append((score, chunk_id, candidate))
        fused.sort(key=lambda item: (-item[0], item[1]))
        shortlist = fused[: self.config.fused_candidates]
        fused_diagnostics = [
            {
                "chunk_id": chunk_id,
                "source_id": str(candidate["row"]["source_id"]),
                "source_title": str(candidate["row"]["title"]),
                "relative_path": str(
                    candidate["row"]["active_path"] or candidate["row"]["note_path"]
                ),
                "section": candidate["row"]["section_title"],
                "page_number": candidate["row"]["page_number"],
                "slide_number": candidate["row"]["slide_number"],
                "lexical_rank": candidate["lexical_rank"],
                "semantic_rank": candidate["semantic_rank"],
                "fused_rank": rank,
                "fused_score": score,
                "boosts": [
                    name
                    for name, applied in boost_diagnostics[chunk_id].items()
                    if isinstance(applied, bool) and applied
                ],
                "boost_details": boost_diagnostics[chunk_id],
            }
            for rank, (score, chunk_id, candidate) in enumerate(shortlist, start=1)
        ]
        fusion_seconds = time.perf_counter() - fusion_started

        top_k = request.top_k or self.config.final_results
        matches: list[RetrievalMatch] = []
        for rank, (score, _, candidate) in enumerate(shortlist, start=1):
            matches.append(self._match(candidate, rank, score))
        rerank_seconds = 0.0
        if (request.rerank or self.config.reranker_enabled) and self.reranker is not None:
            rerank_started = time.perf_counter()
            matches = self.reranker.rerank(request.query, matches, top_k)
            for rank, match in enumerate(matches, start=1):
                match.rerank_rank = rank
            rerank_seconds = time.perf_counter() - rerank_started
        matches = matches[:top_k]
        expansion = (
            self.config.neighbor_expansion
            if request.neighbor_expansion is None
            else request.neighbor_expansion
        )
        if expansion:
            matches = self._expand_neighbors(matches, expansion)
        neighbor_additions = [
            {
                "chunk_id": match.chunk_id,
                "neighbor_of": match.neighbor_of,
                "relative_path": match.source_path,
                "section": match.section,
                "page_number": match.page_number,
                "slide_number": match.slide_number,
            }
            for match in matches
            if match.neighbor_of is not None
        ]
        matches.sort(
            key=lambda match: (
                match.rerank_rank or match.fused_rank,
                match.neighbor_of is not None,
            )
        )

        timings = RetrievalTimings(
            source_resolution_seconds=resolution_seconds,
            lexical_search_seconds=lexical_seconds,
            query_embedding_seconds=embedding_seconds,
            semantic_search_seconds=semantic_seconds,
            fusion_seconds=fusion_seconds,
            rerank_seconds=rerank_seconds,
            total_seconds=time.perf_counter() - total_started,
        )
        return RetrievalResult(
            query=request.query,
            resolved_source=source,
            resolved_section=section,
            matches=matches,
            warnings=warnings,
            filters_applied={key: value for key, value in filters.items() if value},
            timings=timings,
            retrieval_configuration={
                "mode": request.mode,
                "lexical_candidates": request.lexical_k or self.config.lexical_candidates,
                "semantic_candidates": request.semantic_k or self.config.semantic_candidates,
                "fused_candidates": self.config.fused_candidates,
                "rrf_constant": rrf_constant,
                "exact_title_boost": self.config.exact_title_boost,
                "exact_section_boost": self.config.exact_section_boost,
                "neighbor_expansion": expansion,
                "reranking_enabled": self.reranker is not None
                and (request.rerank or self.config.reranker_enabled),
            },
            diagnostics={
                "source_resolution": {
                    "requested_title": request.source_title,
                    "requested_id": request.source_id,
                    "resolved": source,
                    "warnings": resolution_warnings,
                },
                "section_resolution": {
                    "requested": request.section,
                    "resolved": section,
                    "resolved_chunk_count": len(section_ids),
                    "warning": section_warning,
                },
                "lexical_candidates": lexical_diagnostics,
                "semantic_candidates": semantic_diagnostics,
                "fused_candidates": fused_diagnostics,
                "metadata_boosts": boost_diagnostics,
                "neighbor_additions": neighbor_additions,
                "final_chunk_ids": [match.chunk_id for match in matches],
            },
        )

    @staticmethod
    def _matches_filters(row: Any, request: RetrievalRequest) -> bool:
        if request.course and str(row["course"] or "").casefold() != request.course.casefold():
            return False
        if request.document_types and str(row["document_type"]) not in request.document_types:
            return False
        if request.page is not None and row["page_number"] != request.page:
            return False
        path = str(row["active_path"] or row["note_path"])
        if request.folder:
            folder = request.folder.strip("/\\").casefold()
            normalized_path = path.replace("\\", "/").casefold()
            if normalized_path != folder and not normalized_path.startswith(folder + "/"):
                return False
        tags = {str(tag).casefold() for tag in json.loads(row["tags_json"])}
        if any(tag.casefold() not in tags for tag in request.tags):
            return False
        metadata = json.loads(row["metadata_json"])
        if any(metadata.get(key) != value for key, value in request.metadata.items()):
            return False
        return True

    @staticmethod
    def _match(candidate: dict[str, Any], rank: int, score: float) -> RetrievalMatch:
        row = candidate["row"]
        metadata = json.loads(row["metadata_json"])
        metadata["chunk_index"] = int(row["chunk_index"])
        metadata["previous_chunk_id"] = row["previous_chunk_id"]
        metadata["next_chunk_id"] = row["next_chunk_id"]
        return RetrievalMatch(
            chunk_id=str(row["chunk_id"]),
            source_id=str(row["source_id"]),
            source_title=str(row["title"]),
            source_path=str(row["active_path"] or row["note_path"]),
            section=row["section_title"],
            section_path=json.loads(row["section_path_json"]),
            page_number=row["page_number"],
            slide_number=row["slide_number"],
            text=str(row["text"]),
            lexical_rank=candidate["lexical_rank"],
            lexical_score=candidate["lexical_score"],
            semantic_rank=candidate["semantic_rank"],
            semantic_score=candidate["semantic_score"],
            fused_rank=rank,
            fused_score=score,
            metadata=metadata,
            provenance={
                "source_id": str(row["source_id"]),
                "source_hash": str(row["source_hash"]),
                "source_version": int(row["source_version"]),
                "relative_path": str(row["active_path"] or row["note_path"]),
                "page_number": row["page_number"],
                "slide_number": row["slide_number"],
                "section_path": json.loads(row["section_path_json"]),
            },
        )

    def _expand_neighbors(
        self, matches: list[RetrievalMatch], expansion: int
    ) -> list[RetrievalMatch]:
        existing = {match.chunk_id for match in matches}
        additions: list[RetrievalMatch] = []
        for match in matches:
            chunk_index = int(match.metadata["chunk_index"])
            neighbor_ids = self.store.connection.execute(
                """
                SELECT chunk_id FROM chunks
                WHERE source_id=? AND chunk_index BETWEEN ? AND ?
                ORDER BY chunk_index
                """,
                (match.source_id, max(0, chunk_index - expansion), chunk_index + expansion),
            ).fetchall()
            wanted = [
                str(row["chunk_id"])
                for row in neighbor_ids
                if str(row["chunk_id"]) not in existing and str(row["chunk_id"]) != match.chunk_id
            ]
            rows = self.store.chunks_by_ids(wanted)
            for row in rows:
                neighbor = self._match(
                    {
                        "row": row,
                        "lexical_rank": None,
                        "lexical_score": None,
                        "semantic_rank": None,
                        "semantic_score": None,
                    },
                    match.fused_rank,
                    match.fused_score,
                )
                neighbor.neighbor_of = match.chunk_id
                neighbor.rerank_rank = match.rerank_rank
                additions.append(neighbor)
                existing.add(neighbor.chunk_id)
        return [*matches, *additions]


def _candidate_diagnostic(
    chunk_id: str,
    candidate: dict[str, Any],
    rank: int,
    source: str,
) -> dict[str, Any]:
    row = candidate["row"]
    return {
        "chunk_id": chunk_id,
        "source_id": str(row["source_id"]),
        "source_title": str(row["title"]),
        "relative_path": str(row["active_path"] or row["note_path"]),
        "section": row["section_title"],
        "page_number": row["page_number"],
        "slide_number": row["slide_number"],
        "rank": rank,
        "score": (
            candidate["lexical_score"] if source == "lexical" else candidate["semantic_score"]
        ),
    }
