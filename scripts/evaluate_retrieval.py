"""Run local retrieval experiments against a manually authored JSONL case set."""

import argparse
import json
import sys
import time
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
from lecture_slm.knowledge.evaluation.runner import load_cases, run_evaluation
from lecture_slm.knowledge.storage import KnowledgeStore
from lecture_slm.knowledge.vault import validate_knowledge_paths


class _LexicalOnlyEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str, provider_version: str) -> None:
        self._model_name = model_name
        self._provider_version = provider_version

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_version(self) -> str:
        return self._provider_version

    @property
    def dimension(self) -> int:
        return 1

    def embed_documents(self, texts: Sequence[str]) -> list[NDArray[np.float32]]:
        raise RuntimeError("Lexical-only retrieval must not embed documents")

    def embed_query(self, text: str) -> NDArray[np.float32]:
        raise RuntimeError("Lexical-only retrieval must not embed queries")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/knowledge/default.yaml"))
    parser.add_argument(
        "--mode",
        choices=["lexical", "semantic", "hybrid", "all"],
        default="all",
    )
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--lexical-candidates", type=int)
    parser.add_argument("--semantic-candidates", type=int)
    parser.add_argument("--fused-candidates", type=int)
    parser.add_argument("--rrf-constant", type=int)
    parser.add_argument("--neighbor-expansion", type=int, choices=[0, 1])
    parser.add_argument("--compare-neighbors", action="store_true")
    parser.add_argument("--case-version", default="1")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    try:
        cases = load_cases(args.cases)
        config = load_knowledge_config(args.config)
        database = config.data_dir / "knowledge.sqlite"
        if not database.is_file():
            raise FileNotFoundError(
                f"No knowledge index at {database}; explicitly index a vault before evaluation."
            )
        vault_root = KnowledgeStore.indexed_vault_root(config.data_dir)
        validate_knowledge_paths(
            config.data_dir,
            config.embeddings.cache_dir or config.data_dir / "models",
            vault_root,
        )
        output_root = args.output_root or config.data_dir / "evaluation" / "results"
        if (
            not output_root.expanduser()
            .resolve()
            .is_relative_to(config.data_dir.expanduser().resolve())
        ):
            raise ValueError(
                "Evaluation results must be stored under the configured knowledge data directory"
            )
        with KnowledgeStore(
            config.data_dir,
            embedding_model=config.embeddings.model,
            embedding_version=fastembed_provider_version(),
            chunking_version=CHUNKING_VERSION,
        ) as store:
            initialized_seconds = 0.0
            if args.mode == "lexical":
                provider: EmbeddingProvider = _LexicalOnlyEmbeddingProvider(
                    config.embeddings.model,
                    fastembed_provider_version(),
                )
            else:
                initialized_at = time.perf_counter()
                provider = FastEmbedProvider(
                    config.embeddings.model,
                    cache_dir=str(config.embeddings.cache_dir or config.data_dir / "models"),
                )
                initialized_seconds = time.perf_counter() - initialized_at
            run_dir = run_evaluation(
                cases=cases,
                store=store,
                embeddings=provider,
                config=config.retrieval,
                output_root=output_root,
                mode=args.mode,
                top_k=args.top_k,
                lexical_candidates=args.lexical_candidates,
                semantic_candidates=args.semantic_candidates,
                fused_candidates=args.fused_candidates,
                rrf_constant=args.rrf_constant,
                neighbor_expansion=args.neighbor_expansion,
                compare_neighbors=args.compare_neighbors,
                case_version=args.case_version,
                model_initialization_seconds=initialized_seconds,
            )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Retrieval evaluation failed: {error}", file=sys.stderr)
        return 1
    print(f"Evaluation run saved locally: {run_dir}")
    print(f"Cases: {len(cases)}; mode: {args.mode}")
    summary = (run_dir / "summary.json").read_text(encoding="utf-8")
    print("Comparison summary:")
    print(json.dumps(json.loads(summary)["mode_metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
