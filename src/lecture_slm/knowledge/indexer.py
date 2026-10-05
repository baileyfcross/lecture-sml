"""Incremental indexing that reuses the shared deterministic extractors."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lecture_slm.ingestion.extractors import (
    DocxExtractor,
    MarkdownExtractor,
    PdfExtractor,
    PptxExtractor,
    TextExtractor,
)
from lecture_slm.ingestion.extractors.base import Extractor
from lecture_slm.ingestion.manifest import sha256_file, source_id_for_hash
from lecture_slm.ingestion.models import DocumentType
from lecture_slm.knowledge.chunking import CHUNKING_VERSION, chunk_document
from lecture_slm.knowledge.config import KnowledgeConfig
from lecture_slm.knowledge.embeddings import EmbeddingProvider, normalize_vectors
from lecture_slm.knowledge.models import IndexMetrics, KnowledgeChunk
from lecture_slm.knowledge.obsidian import parse_obsidian_markdown
from lecture_slm.knowledge.storage import KnowledgeStore
from lecture_slm.knowledge.vault import (
    VaultFile,
    discover_vault_files,
    validate_knowledge_paths,
)

EXTRACTORS: dict[str, Extractor] = {
    ".pptx": PptxExtractor(),
    ".docx": DocxExtractor(),
    ".pdf": PdfExtractor(),
    ".md": MarkdownExtractor(),
    ".txt": TextExtractor(),
}


class KnowledgeIndexer:
    """Read source files, then write all derived knowledge state under data_dir."""

    def __init__(
        self,
        config: KnowledgeConfig,
        embeddings: EmbeddingProvider,
        *,
        vault_path: Path | None = None,
        rebuild: bool = False,
    ) -> None:
        resolved_vault = vault_path or config.vault_path
        if resolved_vault is None:
            raise ValueError(
                "No vault path configured. Supply a path explicitly or set "
                "LECTURE_SLM_VAULT_PATH; no directory is scanned implicitly."
            )
        self.vault_path = resolved_vault.expanduser().resolve(strict=True)
        if not self.vault_path.is_dir():
            raise ValueError(f"Vault path must be a directory: {self.vault_path}")
        self.data_dir = config.data_dir.expanduser().resolve()
        cache_dir = config.embeddings.cache_dir or config.data_dir / "models"
        validate_knowledge_paths(self.data_dir, cache_dir, self.vault_path)
        if self.vault_path.is_relative_to(self.data_dir):
            raise ValueError("The vault must not be inside the knowledge data directory")
        self.config = config
        self.embeddings = embeddings
        self.rebuild = rebuild

    def index(self, *, dry_run: bool = False) -> IndexMetrics:
        started = time.perf_counter()
        discovered = discover_vault_files(
            self.vault_path,
            extensions=set(self.config.indexing.include_extensions),
            ignored_directories=set(self.config.indexing.ignore_directories),
            recursive=self.config.indexing.recursive,
        )
        metrics = IndexMetrics(files_scanned=len(discovered))
        if dry_run:
            metrics.elapsed_seconds = time.perf_counter() - started
            return metrics

        with KnowledgeStore(
            self.data_dir,
            embedding_model=self.embeddings.model_name,
            embedding_version=self.embeddings.provider_version,
            chunking_version=CHUNKING_VERSION,
            rebuild=self.rebuild,
        ) as store:
            store.ensure_vault_root(self.vault_path, rebuild=self.rebuild)
            vault_id = store.ensure_vault_id()
            present_paths = {file.relative_path for file in discovered}
            for file in discovered:
                self._index_one(file, store, vault_id, metrics)
            metrics.deleted = store.remove_missing_paths(present_paths)
            metrics.elapsed_seconds = time.perf_counter() - started
            indexed_at = datetime.now(UTC).isoformat()
            store.set_metadata("last_indexed_at", indexed_at)
            metrics.indexed_at = datetime.fromisoformat(indexed_at)
        return metrics

    def _index_one(
        self,
        file: VaultFile,
        store: KnowledgeStore,
        vault_id: str,
        metrics: IndexMetrics,
    ) -> None:
        relative_path = file.relative_path
        modified_time = datetime.now(UTC)
        try:
            modified_time = datetime.fromtimestamp(file.path.stat().st_mtime, UTC)
            source_hash = sha256_file(file.path)
            current = store.source_path(relative_path)
            if (
                current
                and current["source_hash"] == source_hash
                and current["extraction_status"] == "success"
            ):
                metrics.unchanged += 1
                metrics.chunks_reused += self._source_chunk_count(store, str(current["source_id"]))
                return

            version = int(current["source_version"]) + 1 if current else 1
            duplicate = store.source_by_hash(source_hash)
            if duplicate:
                metrics.duplicates += 1
                store.add_source_path(
                    relative_path=relative_path,
                    source_id=str(duplicate["source_id"]),
                    vault_id=vault_id,
                    modified_time=modified_time,
                    source_version=version,
                )
                metrics.new += int(current is None)
                metrics.changed += int(current is not None)
                metrics.chunks_reused += self._source_chunk_count(
                    store, str(duplicate["source_id"])
                )
                return

            if current:
                metrics.changed += 1
            else:
                metrics.new += 1
            extractor = EXTRACTORS[file.path.suffix.casefold()]
            source_id = source_id_for_hash(source_hash)
            document = extractor.extract(
                file.path,
                source_id=source_id,
                source_hash=source_hash,
                relative_path=relative_path,
                source_version=version,
            )
            note_metadata: dict[str, Any] = {}
            warnings = list(document.extraction_warnings)
            if document.document_type is DocumentType.MARKDOWN:
                note_metadata, metadata_warnings = parse_obsidian_markdown(
                    file.path.read_text(encoding="utf-8")
                )
                warnings.extend(metadata_warnings)
                document.metadata.update(note_metadata)

            chunks = chunk_document(
                document,
                relative_path=relative_path,
                source_version=version,
                embedding_model=self.embeddings.model_name,
                embedding_version=self.embeddings.provider_version,
                config=self.config.chunking,
                note_metadata=note_metadata,
            )
            embedding_keys, generated, reused = self._embed_missing(chunks, store)
            store.replace_source(
                source_id=source_id,
                source_hash=source_hash,
                title=str(note_metadata.get("title") or document.title or file.path.stem),
                document_type=document.document_type.value,
                source_version=version,
                relative_path=relative_path,
                vault_id=vault_id,
                modified_time=modified_time,
                metadata=document.metadata,
                warnings=warnings,
                chunks=chunks,
                embedding_keys=embedding_keys,
            )
            metrics.normalized += 1
            metrics.chunks_created += len(chunks)
            metrics.embeddings_generated += generated
            metrics.embeddings_reused += reused
            metrics.warnings.extend(f"{relative_path}: {warning}" for warning in warnings)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            self._record_failure(file, store, vault_id, modified_time, error, metrics)
        except Exception as error:
            self._record_failure(file, store, vault_id, modified_time, error, metrics)

    def _embed_missing(
        self, chunks: list[KnowledgeChunk], store: KnowledgeStore
    ) -> tuple[dict[str, str], int, int]:
        keys = {
            chunk.chunk_id: embedding_cache_key(
                self.embeddings.model_name,
                self.embeddings.provider_version,
                chunk.text,
            )
            for chunk in chunks
        }
        missing: list[KnowledgeChunk] = []
        reused = 0
        for chunk in chunks:
            if store.cache_embedding(keys[chunk.chunk_id]) is None:
                missing.append(chunk)
            else:
                reused += 1
        batch_size = self.config.embeddings.batch_size
        generated = 0
        for offset in range(0, len(missing), batch_size):
            batch = missing[offset : offset + batch_size]
            vectors = normalize_vectors(
                self.embeddings.embed_documents([item.text for item in batch])
            )
            if len(vectors) != len(batch):
                raise ValueError(
                    "Embedding provider returned a different number of vectors than texts"
                )
            for chunk, vector in zip(batch, vectors, strict=True):
                if vector.size != self.embeddings.dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch for {self.embeddings.model_name}: "
                        f"expected {self.embeddings.dimension}, received {vector.size}"
                    )
                store.save_embedding(
                    keys[chunk.chunk_id],
                    vector,
                    model_name=self.embeddings.model_name,
                    provider_version=self.embeddings.provider_version,
                )
                generated += 1
        return keys, generated, reused

    def _record_failure(
        self,
        file: VaultFile,
        store: KnowledgeStore,
        vault_id: str,
        modified_time: datetime,
        error: Exception,
        metrics: IndexMetrics,
    ) -> None:
        source_hash = "0" * 64
        try:
            source_hash = sha256_file(file.path)
        except OSError:
            pass
        current = store.source_path(file.relative_path)
        version = int(current["source_version"]) + 1 if current else 1
        source_id = source_id_for_hash(source_hash)
        message = f"{file.relative_path}: {type(error).__name__}: {error}"
        store.record_failure(
            source_id=source_id,
            source_hash=source_hash,
            relative_path=file.relative_path,
            vault_id=vault_id,
            modified_time=modified_time,
            source_version=version,
            title=file.path.stem,
            document_type=DocumentType.UNKNOWN.value,
            error_type=type(error).__name__,
            message=message,
            preserve_existing_path=bool(
                current
                and current["extraction_status"] == "success"
                and current["source_hash"] != source_hash
            ),
        )
        metrics.failed += 1
        metrics.errors.append(message)

    @staticmethod
    def _source_chunk_count(store: KnowledgeStore, source_id: str) -> int:
        row = store.connection.execute(
            "SELECT COUNT(*) AS count FROM chunks WHERE source_id = ?", (source_id,)
        ).fetchone()
        return int(row["count"])


def embedding_cache_key(model_name: str, provider_version: str, text: str) -> str:
    key_material = json.dumps([model_name, provider_version, text], ensure_ascii=False)
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()
