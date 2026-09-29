"""Explicit approved-candidate export to the canonical dataset schema."""

import json
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lecture_slm.ingestion.models import CandidateTrainingExample, ReviewStatus
from lecture_slm.ingestion.review import CandidateStore
from lecture_slm.schemas.dataset import DatasetExample, DatasetSplit, Provenance, QualityMetadata


class ExportError(ValueError):
    """Raised when approved candidates cannot safely become dataset records."""


def _assert_safe_provenance(candidate: CandidateTrainingExample) -> None:
    forbidden = ("evals/", "evals\\", "artifacts/generations/", "artifacts\\generations\\")
    for provenance in candidate.provenance:
        path = provenance.source_file.lower()
        if any(marker in path for marker in forbidden):
            raise ExportError(f"Evaluation or generated runtime source is not exportable: {path}")


def _to_dataset_example(candidate: CandidateTrainingExample, version: str) -> DatasetExample:
    _assert_safe_provenance(candidate)
    missing = [
        name
        for name, value in {
            "task": candidate.task,
            "course": candidate.course,
            "level": candidate.level,
            "instruction": candidate.instruction,
            "expected_output": candidate.expected_output,
        }.items()
        if value is None or (isinstance(value, str) and not value.strip())
    ]
    if missing:
        raise ExportError(
            f"Approved candidate {candidate.candidate_id} is missing: {', '.join(missing)}"
        )
    task = candidate.task
    course = candidate.course
    level = candidate.level
    instruction = candidate.instruction
    if task is None or course is None or level is None or instruction is None:
        raise ExportError(f"Approved candidate {candidate.candidate_id} has incomplete metadata")
    provenance = Provenance(
        sources=[record.source_file for record in candidate.provenance],
        human_created=candidate.authorship.value.startswith("instructor"),
        existing_material_type=task.value,
        instructor_revision=candidate.authorship.value == "instructor_edited",
        synthetic=candidate.authorship.value == "synthetic_unreviewed",
        teacher_model_generated=candidate.authorship.value == "ai_assisted_human_reviewed",
        human_reviewed=True,
        approved=True,
        quality_tier=candidate.quality_tier,
    )
    return DatasetExample(
        id=candidate.candidate_id,
        split=DatasetSplit.TRAINING,
        task=task,
        course=course,
        level=level,
        instruction=instruction,
        context=candidate.context,
        source_material=candidate.input_material,
        response=candidate.expected_output,
        pedagogy_tags=candidate.pedagogy_tags,
        provenance=provenance,
        quality=QualityMetadata(notes=candidate.reviewer_notes),
        version=version,
        metadata={"candidate_version": candidate.version, "source_ids": candidate.source_ids},
    )


def _git_commit() -> str | None:
    try:
        git = shutil.which("git")
        if git is None:
            return None
        return subprocess.run(  # noqa: S603 - executable is resolved from PATH
            [git, "rev-parse", "HEAD"], capture_output=True, check=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def export_approved(
    candidates_path: Path,
    output_dir: Path,
    *,
    version: str,
    overwrite: bool = False,
) -> tuple[Path, Path, int]:
    """Export only approved candidates and create a reproducibility manifest."""

    store = CandidateStore(candidates_path)
    approved = store.list(status=ReviewStatus.APPROVED)
    if not approved:
        raise ExportError("No approved candidates are available for export")
    records = [_to_dataset_example(candidate, version) for candidate in approved]
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ExportError("Duplicate approved candidate IDs detected")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"approved-sft-v{version}.jsonl"
    manifest_path = output_dir / f"approved-sft-v{version}.manifest.json"
    if (output_path.exists() or manifest_path.exists()) and not overwrite:
        raise ExportError(f"Dataset version already exists: {version}; use --overwrite explicitly")
    output_path.write_text(
        "".join(json.dumps(record.model_dump(mode="json")) + "\n" for record in records),
        encoding="utf-8",
    )
    source_hashes = sorted(
        {record.source_hash for candidate in approved for record in candidate.provenance}
    )
    manifest: dict[str, Any] = {
        "dataset_version": version,
        "exported_at": datetime.now(UTC).isoformat(),
        "record_count": len(records),
        "task_distribution": dict(Counter(record.task.value for record in records)),
        "course_distribution": dict(Counter(record.course for record in records)),
        "quality_tier_distribution": dict(
            Counter(record.provenance.quality_tier.value for record in records)
        ),
        "source_count": len(source_hashes),
        "source_hashes": source_hashes,
        "candidate_ids": ids,
        "schema_version": "dataset-v1",
        "git_commit": _git_commit(),
        "exporter_version": "1",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path, manifest_path, len(records)
