"""Index an explicitly supplied local knowledge directory without modifying it."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lecture_slm.knowledge.chunking import CHUNKING_VERSION
from lecture_slm.knowledge.config import load_knowledge_config
from lecture_slm.knowledge.embeddings import (
    EmbeddingProvider,
    FastEmbedProvider,
    fastembed_provider_version,
)
from lecture_slm.knowledge.indexer import KnowledgeIndexer
from lecture_slm.knowledge.stats import knowledge_stats
from lecture_slm.knowledge.storage import KnowledgeStore
from lecture_slm.knowledge.vault import dry_run_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vault_path", nargs="?", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/knowledge/default.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    try:
        config = load_knowledge_config(args.config)
        if args.status:
            print(json.dumps(knowledge_stats(config.data_dir), ensure_ascii=False, indent=2))
            return 0
        if args.vault_path is None and config.vault_path is None:
            raise ValueError(
                "Vault path required: pass it explicitly or set LECTURE_SLM_VAULT_PATH."
            )
        if args.dry_run:
            indexer = KnowledgeIndexer(
                config,
                _DryRunEmbeddingProvider(),
                vault_path=args.vault_path,
                rebuild=args.rebuild,
            )
            report = dry_run_report(
                indexer.vault_path,
                extensions=set(config.indexing.include_extensions),
                ignored_directories=set(config.indexing.ignore_directories),
                recursive=config.indexing.recursive,
                data_dir=config.data_dir,
                embedding_model=config.embeddings.model,
                embedding_cache_dir=config.embeddings.cache_dir or config.data_dir / "models",
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        else:
            indexer = KnowledgeIndexer(
                config,
                _DryRunEmbeddingProvider(),
                vault_path=args.vault_path,
                rebuild=args.rebuild,
            )
            if not args.rebuild:
                with KnowledgeStore(
                    config.data_dir,
                    embedding_model=config.embeddings.model,
                    embedding_version=fastembed_provider_version(),
                    chunking_version=CHUNKING_VERSION,
                ):
                    pass
            provider = FastEmbedProvider(
                config.embeddings.model,
                cache_dir=str(config.embeddings.cache_dir or config.data_dir / "models"),
            )
            indexer.embeddings = provider
            metrics = indexer.index()
            metrics_report = metrics.model_dump(mode="json")
            metrics_report["indexed_content"] = knowledge_stats(config.data_dir)
    except (OSError, ValueError) as error:
        print(f"Knowledge indexing failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(metrics_report, ensure_ascii=False, indent=2))
    return 0 if not metrics.errors else 1


class _DryRunEmbeddingProvider(EmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "dry-run"

    @property
    def provider_version(self) -> str:
        return "dry-run"

    @property
    def dimension(self) -> int:
        return 1

    def embed_documents(self, texts: Sequence[str]) -> list[NDArray[np.float32]]:
        return []

    def embed_query(self, text: str) -> NDArray[np.float32]:
        raise RuntimeError("Dry-run must not embed")


if __name__ == "__main__":
    sys.exit(main())
