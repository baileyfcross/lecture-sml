"""Embedding interfaces and the local FastEmbed implementation."""

from collections.abc import Iterable, Sequence
from importlib.metadata import version
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provider boundary used by index and retrieval implementations."""

    @property
    def dimension(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[NDArray[np.float32]]: ...

    def embed_query(self, text: str) -> NDArray[np.float32]: ...


class FastEmbedProvider:
    """Locally embed text using FastEmbed's passage and query APIs."""

    def __init__(self, model_name: str, *, cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding

        self._model_name = model_name
        self._dimension = TextEmbedding.get_embedding_size(model_name)
        self._provider_version = fastembed_provider_version()
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_version(self) -> str:
        return self._provider_version

    def embed_documents(self, texts: Sequence[str]) -> list[NDArray[np.float32]]:
        return [np.asarray(vector, dtype=np.float32) for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> NDArray[np.float32]:
        vector = next(iter(self._model.query_embed(text)))
        return np.asarray(vector, dtype=np.float32)


def normalize_vectors(vectors: Iterable[NDArray[np.floating[Any]]]) -> list[NDArray[np.float32]]:
    """Normalize provider vectors to finite, one-dimensional float32 arrays."""

    normalized: list[NDArray[np.float32]] = []
    for vector in vectors:
        result = np.asarray(vector, dtype=np.float32).reshape(-1)
        if not result.size or not np.isfinite(result).all():
            raise ValueError("embedding provider returned an empty or non-finite vector")
        normalized.append(result)
    return normalized


def fastembed_provider_version() -> str:
    """Return the installed FastEmbed version without initializing a model."""

    return version("fastembed")
