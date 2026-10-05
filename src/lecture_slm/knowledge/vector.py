"""Replaceable dense-search boundary with a local NumPy cosine implementation."""

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from lecture_slm.knowledge.storage import KnowledgeStore


class VectorSearch(Protocol):
    """Dense candidate-search API independent of the backing vector store."""

    def search(
        self,
        query_vector: NDArray[np.floating[Any]],
        *,
        source_id: str | None,
        limit: int,
    ) -> list[tuple[Any, float]]: ...


class SQLiteCosineSearch:
    """Exact cosine search over float32 vectors stored alongside SQLite metadata."""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def search(
        self,
        query_vector: NDArray[np.floating[Any]],
        *,
        source_id: str | None,
        limit: int,
    ) -> list[tuple[Any, float]]:
        rows: Sequence[Any] = self.store.active_chunk_rows()
        if source_id:
            rows = [row for row in rows if row["source_id"] == source_id]
        scored: list[tuple[Any, float]] = []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0:
            raise ValueError("Query embedding has zero magnitude")
        for row in rows:
            vector = self.store.cache_embedding(str(row["embedding_key"]))
            if vector is None:
                raise ValueError(f"Embedding is missing for indexed chunk {row['chunk_id']}")
            if vector.size != query.size:
                raise ValueError(f"Stored embedding dimension mismatch for chunk {row['chunk_id']}")
            denominator = query_norm * float(np.linalg.norm(vector))
            score = float(np.dot(query, vector) / denominator) if denominator else 0.0
            scored.append((row, score))
        scored.sort(key=lambda item: (-item[1], str(item[0]["chunk_id"])))
        return scored[:limit]
