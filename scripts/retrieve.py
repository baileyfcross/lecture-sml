"""Search the local knowledge index without contacting a generation model."""

import argparse
import json
import sys
from pathlib import Path

from lecture_slm.generation.context import estimate_tokens_from_characters
from lecture_slm.knowledge.assembler import KnowledgeContextAssembler
from lecture_slm.knowledge.chunking import CHUNKING_VERSION
from lecture_slm.knowledge.config import load_knowledge_config
from lecture_slm.knowledge.embeddings import FastEmbedProvider, fastembed_provider_version
from lecture_slm.knowledge.models import RetrievalRequest
from lecture_slm.knowledge.retrieval import KnowledgeRetriever
from lecture_slm.knowledge.roles import ChunkRole
from lecture_slm.knowledge.storage import KnowledgeStore
from lecture_slm.knowledge.vault import validate_knowledge_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--config", type=Path, default=Path("configs/knowledge/default.yaml"))
    parser.add_argument("--source-title")
    parser.add_argument("--source-id")
    parser.add_argument("--section")
    parser.add_argument("--course")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--folder")
    parser.add_argument("--document-type", action="append", default=[])
    parser.add_argument("--page", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--neighbor-expansion", type=int)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument(
        "--include-role",
        action="append",
        choices=[role.value for role in ChunkRole],
        help="Also include a non-default role in factual passage retrieval.",
    )
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    try:
        config = load_knowledge_config(args.config)
        database = config.data_dir / "knowledge.sqlite"
        if not database.exists():
            raise FileNotFoundError(
                f"No knowledge index at {database}; run scripts/index_knowledge.py first."
            )
        validate_knowledge_paths(
            config.data_dir,
            config.embeddings.cache_dir or config.data_dir / "models",
            KnowledgeStore.indexed_vault_root(config.data_dir),
        )
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
        with KnowledgeStore(
            config.data_dir,
            embedding_model=provider.model_name,
            embedding_version=provider.provider_version,
            chunking_version=CHUNKING_VERSION,
        ) as store:
            passage_roles = list(config.retrieval.default_passage_roles)
            for role in args.include_role or []:
                selected_role = ChunkRole(role)
                if selected_role not in passage_roles:
                    passage_roles.append(selected_role)
            result = KnowledgeRetriever(store, provider, config.retrieval).retrieve(
                RetrievalRequest(
                    query=args.query,
                    source_title=args.source_title,
                    source_id=args.source_id,
                    section=args.section,
                    course=args.course,
                    tags=args.tag,
                    document_types=args.document_type,
                    folder=args.folder,
                    page=args.page,
                    top_k=args.top_k,
                    neighbor_expansion=args.neighbor_expansion,
                    passage_roles=passage_roles,
                )
            )
    except (OSError, ValueError) as error:
        print(f"Knowledge retrieval failed: {error}", file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if args.explain:
        print("Retrieval diagnostics:")
        print(json.dumps(result.diagnostics, ensure_ascii=False, indent=2))
        assembled = KnowledgeContextAssembler().assemble(
            result,
            source_context_budget=config.retrieval.context_budgets["standard"],
        )
        selected_ids = [
            chunk_id for source in assembled for chunk_id in source.metadata.get("chunk_ids", [])
        ]
        print(
            "Context assembly: "
            + json.dumps(
                {
                    "candidate_chunks": len(result.matches),
                    "selected_chunk_ids": selected_ids,
                    "source_material_count": len(assembled),
                    "configured_budget": config.retrieval.context_budgets["standard"],
                    "estimated_tokens": sum(
                        estimate_tokens_from_characters(
                            f"{source.title}\n{source.section or ''}\n{source.text}"
                        )
                        for source in assembled
                    ),
                },
                ensure_ascii=False,
            )
        )
    for match in result.matches:
        print(f"{match.fused_rank}. {match.source_title}")
        print(f"   Path: {match.source_path}")
        if match.section:
            print(f"   Section: {match.section}")
        if match.page_number:
            print(f"   Page: {match.page_number}")
        if match.slide_number:
            print(f"   Slide: {match.slide_number}")
        print(
            f"   Hybrid rank: {match.fused_rank}; lexical rank: {match.lexical_rank}; "
            f"semantic rank: {match.semantic_rank}"
        )
        preview = match.text if args.full else _preview(match.text)
        print(f"\n   {preview}\n")
    print(f"Retrieved {len(result.matches)} matches in {result.timings.total_seconds:.3f}s")
    return 0


def _preview(text: str, limit: int = 360) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


if __name__ == "__main__":
    sys.exit(main())
