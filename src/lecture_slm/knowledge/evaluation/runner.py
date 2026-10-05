"""Execute repeatable, local retrieval evaluation runs."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from lecture_slm.generation.context import estimate_tokens_from_characters
from lecture_slm.knowledge.assembler import KnowledgeContextAssembler
from lecture_slm.knowledge.config import RetrievalConfig
from lecture_slm.knowledge.embeddings import EmbeddingProvider
from lecture_slm.knowledge.evaluation.metrics import (
    aggregate_metrics,
    case_metrics,
    classify_resolution_failures,
)
from lecture_slm.knowledge.evaluation.models import RetrievalEvalCase
from lecture_slm.knowledge.models import RetrievalRequest
from lecture_slm.knowledge.retrieval import KnowledgeRetriever
from lecture_slm.knowledge.storage import KnowledgeStore

EvaluationMode = Literal["lexical", "semantic", "hybrid", "all"]


def load_cases(path: Path) -> list[RetrievalEvalCase]:
    cases: list[RetrievalEvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(RetrievalEvalCase.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"Invalid evaluation case at {path}:{line_number}: {error}") from error
    if not cases:
        raise ValueError(f"No retrieval evaluation cases found in {path}")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation case IDs must be unique within a JSONL file")
    return cases


def run_evaluation(
    *,
    cases: list[RetrievalEvalCase],
    store: KnowledgeStore,
    embeddings: EmbeddingProvider,
    config: RetrievalConfig,
    output_root: Path,
    mode: EvaluationMode,
    top_k: int | None = None,
    lexical_candidates: int | None = None,
    semantic_candidates: int | None = None,
    fused_candidates: int | None = None,
    rrf_constant: int | None = None,
    neighbor_expansion: int | None = None,
    compare_neighbors: bool = False,
    case_version: str = "1",
    model_initialization_seconds: float = 0.0,
) -> Path:
    """Write a fully local run bundle and return its private output directory."""

    selected_modes = ["lexical", "semantic", "hybrid"] if mode == "all" else [mode]
    overrides = {
        key: value
        for key, value in {
            "lexical_candidates": lexical_candidates,
            "semantic_candidates": semantic_candidates,
            "fused_candidates": fused_candidates,
            "rrf_constant": rrf_constant,
            "final_results": top_k,
        }.items()
        if value is not None
    }
    retrieval_config = RetrievalConfig.model_validate({**config.model_dump(), **overrides})
    variants = [(name, name, neighbor_expansion) for name in selected_modes]
    if compare_neighbors and "hybrid" in selected_modes:
        variants.extend([("hybrid-neighbors-0", "hybrid", 0), ("hybrid-neighbors-1", "hybrid", 1)])
    elif neighbor_expansion is not None:
        variants = [(name, name, neighbor_expansion) for name in selected_modes]

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    assembler = KnowledgeContextAssembler()
    context_budget = retrieval_config.context_budgets["standard"]
    for mode_name, retrieval_mode, neighbor_count in variants:
        retriever = KnowledgeRetriever(store, embeddings, retrieval_config)
        for case in cases:
            request = RetrievalRequest(
                query=case.query,
                mode=cast(Literal["lexical", "semantic", "hybrid"], retrieval_mode),
                source_title=case.source_title,
                source_id=case.source_id,
                section=case.section,
                course=case.course,
                tags=case.tags,
                document_types=case.document_types,
                folder=case.folder,
                page=case.page,
                top_k=top_k,
                neighbor_expansion=neighbor_count,
                metadata=case.metadata,
            )
            result = retriever.retrieve(request)
            assembly_started = time.perf_counter()
            assembled = assembler.assemble(result, source_context_budget=context_budget)
            assembly_seconds = time.perf_counter() - assembly_started
            assembled_text = [source.text for source in assembled]
            selected_ids = {
                chunk_id
                for source in assembled
                for chunk_id in source.metadata.get("chunk_ids", [])
            }
            raw_text_chars = sum(
                len(match.text) for match in result.matches if match.chunk_id in selected_ids
            )
            merged_text_chars = sum(len(text) for text in assembled_text)
            estimated_tokens = sum(
                estimate_tokens_from_characters(
                    f"{source.title}\n{source.section or ''}\n{source.text}"
                )
                for source in assembled
            )
            effective_top_k = top_k or retrieval_config.final_results
            metrics = case_metrics(case, result, top_k=effective_top_k)
            results.append(
                {
                    "case_id": case.id,
                    "case_category": case.category,
                    "mode": mode_name,
                    "query": case.query,
                    "filters": {
                        "source_title": case.source_title,
                        "source_id": case.source_id,
                        "section": case.section,
                        "course": case.course,
                        "tags": case.tags,
                        "document_types": case.document_types,
                        "folder": case.folder,
                        "page": case.page,
                        "metadata": case.metadata,
                    },
                    "expected": {
                        "source_ids": case.expected_source_ids,
                        "source_titles": case.expected_source_titles,
                        "sections": case.expected_sections,
                        "pages": case.expected_pages,
                    },
                    "metrics": metrics,
                    "failure_categories": classify_resolution_failures(result),
                    "timings": result.timings.model_dump(mode="json"),
                    "context_assembly_seconds": assembly_seconds,
                    "retrieval_and_assembly_seconds": (
                        result.timings.total_seconds + assembly_seconds
                    ),
                    "resolution": result.diagnostics,
                    "retrieval_configuration": result.retrieval_configuration,
                    "matches": [match.model_dump(mode="json") for match in result.matches],
                    "assembled_source_material": [
                        source.model_dump(mode="json") for source in assembled
                    ],
                    "context_assembly": {
                        "candidate_chunks": len(result.matches),
                        "selected_chunk_ids": sorted(selected_ids),
                        "selected_chunks": len(selected_ids),
                        "merged_source_materials": len(assembled),
                        "merged_chunks": sum(
                            max(0, len(source.metadata.get("chunk_ids", [])) - 1)
                            for source in assembled
                        ),
                        "removed_overlap_characters": max(0, raw_text_chars - merged_text_chars),
                        "source_material_count": len(assembled),
                        "estimated_tokens": estimated_tokens,
                        "configured_budget": context_budget,
                        "unused_budget": max(0, context_budget - estimated_tokens),
                        "excluded_chunk_ids": [
                            match.chunk_id
                            for match in result.matches
                            if match.chunk_id not in selected_ids
                        ],
                    },
                }
            )
    elapsed = time.perf_counter() - start
    db_meta = {
        key: store.get_metadata(key)
        for key in (
            "vault_id",
            "last_indexed_at",
            "schema_version",
            "embedding_model",
            "chunking_version",
        )
    }
    run_record = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "case_version": case_version,
        "case_count": len(cases),
        "modes": [variant[0] for variant in variants],
        "index_identity": db_meta,
        "git_commit": _git_commit(),
        "retrieval_configuration": {
            "lexical_candidates": retrieval_config.lexical_candidates,
            "semantic_candidates": retrieval_config.semantic_candidates,
            "fused_candidates": retrieval_config.fused_candidates,
            "top_k": top_k or retrieval_config.final_results,
            "rrf_constant": retrieval_config.rrf_constant,
            "neighbor_expansion": neighbor_expansion,
            "compare_neighbors": compare_neighbors,
            "embedding_model": store.get_metadata("embedding_model"),
        },
        "model_initialization_seconds": model_initialization_seconds,
        "evaluation_elapsed_seconds": elapsed,
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "cases.jsonl").write_text(
        "".join(case.model_dump_json() + "\n" for case in cases), encoding="utf-8"
    )
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in results),
        encoding="utf-8",
    )
    (run_dir / "reviews.jsonl").write_text("", encoding="utf-8")
    summaries = {
        mode_name: aggregate_metrics([record for record in results if record["mode"] == mode_name])
        for mode_name, _, _ in variants
    }
    comparison = _mode_comparisons(summaries)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "mode_metrics": summaries,
                "mode_comparison": comparison,
                "latency_seconds": _latency_summary(results),
                "human_review": _review_summary([]),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def update_review_summary(run_dir: Path) -> None:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reviews = [
        json.loads(line)
        for line in (run_dir / "reviews.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary["human_review"] = _review_summary(reviews)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _latency_summary(results: list[dict[str, Any]]) -> dict[str, float | None]:
    timings = [item["timings"] for item in results]
    fields = (
        "source_resolution_seconds",
        "lexical_search_seconds",
        "query_embedding_seconds",
        "semantic_search_seconds",
        "fusion_seconds",
        "total_seconds",
    )
    summary = {
        key: (sum(float(item[key]) for item in timings) / len(timings) if timings else None)
        for key in fields
    }
    summary["context_assembly_seconds"] = (
        sum(float(item["context_assembly_seconds"]) for item in results) / len(results)
        if results
        else None
    )
    summary["retrieval_and_assembly_seconds"] = (
        sum(float(item["retrieval_and_assembly_seconds"]) for item in results) / len(results)
        if results
        else None
    )
    return summary


def _review_summary(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    for review in reviews:
        for field in ("relevance", "context_sufficiency", "failure_category"):
            value = review.get(field)
            if value:
                field_counts = counts.setdefault(field, {})
                field_counts[str(value)] = field_counts.get(str(value), 0) + 1
    return {"review_count": len(reviews), "label_counts": counts}


def _mode_comparisons(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = summaries.get("hybrid")
    if baseline is None:
        return {}
    metrics = ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_k", "mrr")
    comparison: dict[str, Any] = {}
    for mode_name, summary in summaries.items():
        if mode_name == "hybrid":
            continue
        comparison[mode_name] = {
            key: (
                float(summary[key]) - float(baseline[key])
                if summary.get(key) is not None and baseline.get(key) is not None
                else None
            )
            for key in metrics
        }
    return comparison


def _git_commit() -> str | None:
    repository = Path(__file__).resolve().parents[4]
    git_entry = repository / ".git"
    try:
        git_directory = git_entry
        if git_entry.is_file():
            pointer = git_entry.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir: "):
                return None
            git_directory = (repository / pointer[8:]).resolve()
        head = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        reference = head[5:]
        ref_path = git_directory / reference
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip()
        packed_refs = git_directory / "packed-refs"
        if packed_refs.is_file():
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + reference) and not line.startswith(("^", "#")):
                    return line.split(" ", maxsplit=1)[0]
    except OSError:
        return None
    return None
