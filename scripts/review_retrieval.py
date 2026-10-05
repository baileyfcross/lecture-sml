"""Interactively review private retrieval results without an LLM judge."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from lecture_slm.knowledge.evaluation.models import RetrievalReview
from lecture_slm.knowledge.evaluation.runner import update_review_summary

RELEVANCE = ("highly_relevant", "relevant", "partially_relevant", "irrelevant")
SUFFICIENCY = ("sufficient", "partially_sufficient", "insufficient")
FAILURES = (
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
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--preview-characters", type=int, default=400)
    args = parser.parse_args()
    try:
        _review_run(args.run_dir, max(80, args.preview_characters))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Retrieval review failed: {error}", file=sys.stderr)
        return 1
    return 0


def _review_run(run_dir: Path, preview_characters: int) -> None:
    run_dir = run_dir.expanduser().resolve(strict=True)
    results_path = run_dir / "results.jsonl"
    reviews_path = run_dir / "reviews.jsonl"
    if not results_path.is_file() or not reviews_path.is_file():
        raise FileNotFoundError(f"Not a retrieval evaluation run directory: {run_dir}")
    existing = {
        (item["case_id"], item["mode"], item["chunk_id"]): item for item in _jsonl(reviews_path)
    }
    results = _jsonl(results_path)
    for record in results:
        print("\n" + "=" * 78)
        print(f"Case: {record['case_id']} [{record['case_category']}] | Mode: {record['mode']}")
        print(f"Query: {record['query']}")
        print(f"Filters: {json.dumps(record['filters'], ensure_ascii=False)}")
        print(f"Resolved source: {record['resolution']['source_resolution'].get('resolved')}")
        print(f"Resolved section: {record['resolution']['section_resolution'].get('resolved')}")
        print(
            f"Context: {record['context_assembly']['selected_chunks']} chunks; "
            f"~{record['context_assembly']['estimated_tokens']} tokens / "
            f"{record['context_assembly']['configured_budget']} budget"
        )
        print(
            "Enter a relevance label for each passage; blank leaves the existing review unchanged."
        )
        print("Assembled context previews:")
        for source in record.get("assembled_source_material", []):
            print(
                f"- {source['title']} | {source.get('section') or ''}: "
                f"{_preview(source['text'], preview_characters)}"
            )
        for rank, match in enumerate(record["matches"], start=1):
            print(
                f"\n{rank}. {match['source_title']} | {match['source_path']} | "
                f"{match.get('section') or ''} | page={match.get('page_number')} "
                f"slide={match.get('slide_number')}"
            )
            print(
                f"   lexical={match.get('lexical_rank')} semantic={match.get('semantic_rank')} "
                f"fused={match.get('fused_rank')}"
            )
            print(_preview(match["text"], preview_characters))
            key = (record["case_id"], record["mode"], match["chunk_id"])
            existing_review = existing.get(key, {})
            label = _prompt_choice(
                "Relevance",
                RELEVANCE,
                existing_review.get("relevance"),
            )
            if label is not None:
                failure = _prompt_choice(
                    "Failure category (optional)",
                    FAILURES,
                    existing_review.get("failure_category"),
                    allow_blank=True,
                )
                note = input("Notes (optional): ").strip()
                existing[key] = RetrievalReview.model_validate(
                    {
                        "case_id": record["case_id"],
                        "mode": record["mode"],
                        "chunk_id": match["chunk_id"],
                        "relevance": label,
                        "failure_category": failure,
                        "notes": note,
                    }
                ).model_dump(mode="json")
        sufficiency_key = (record["case_id"], record["mode"], "__context__")
        existing_review = existing.get(sufficiency_key, {})
        sufficiency = _prompt_choice(
            "Assembled context sufficiency",
            SUFFICIENCY,
            existing_review.get("context_sufficiency"),
        )
        if sufficiency is not None:
            failure = _prompt_choice(
                "Context failure category (optional)",
                FAILURES,
                existing_review.get("failure_category"),
                allow_blank=True,
            )
            note = input("Context notes (optional): ").strip()
            existing[sufficiency_key] = RetrievalReview.model_validate(
                {
                    "case_id": record["case_id"],
                    "mode": record["mode"],
                    "chunk_id": "__context__",
                    "context_sufficiency": sufficiency,
                    "failure_category": failure,
                    "notes": note,
                }
            ).model_dump(mode="json")
        reviews_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in existing.values()),
            encoding="utf-8",
        )
        update_review_summary(run_dir)
    print(f"\nSaved human review locally in {reviews_path}")


def _prompt_choice(
    title: str,
    options: tuple[str, ...],
    existing: str | None,
    *,
    allow_blank: bool = False,
) -> str | None:
    display = " / ".join(f"{index}:{option}" for index, option in enumerate(options, start=1))
    suffix = f" [current: {existing}]" if existing else ""
    answer = input(f"{title} ({display}; Enter to skip{suffix}): ").strip()
    if not answer:
        return None
    if allow_blank and answer.casefold() in {"none", "n"}:
        return None
    if answer.isdigit() and 1 <= int(answer) <= len(options):
        return options[int(answer) - 1]
    if answer in options:
        return answer
    print("Unrecognized choice; review value was not changed.")
    return None


def _preview(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


if __name__ == "__main__":
    sys.exit(main())
