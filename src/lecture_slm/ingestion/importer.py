"""Incremental local import orchestration."""

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lecture_slm.ingestion.candidate_builder import build_candidate_from_path
from lecture_slm.ingestion.discovery import DiscoveryResult, discover_sources
from lecture_slm.ingestion.extractors import (
    DocxExtractor,
    MarkdownExtractor,
    PdfExtractor,
    PptxExtractor,
    TextExtractor,
)
from lecture_slm.ingestion.extractors.base import Extractor
from lecture_slm.ingestion.manifest import (
    ManifestStore,
    mark_unchanged,
    next_source_version,
    sha256_file,
    source_id_for_hash,
)
from lecture_slm.ingestion.models import (
    DocumentType,
    ExtractionStatus,
    ReviewStatus,
    SourceManifest,
)
from lecture_slm.ingestion.review import CandidateStore

LOGGER = logging.getLogger(__name__)

EXTRACTORS: dict[str, Extractor] = {
    ".pptx": PptxExtractor(),
    ".docx": DocxExtractor(),
    ".pdf": PdfExtractor(),
    ".md": MarkdownExtractor(),
    ".txt": TextExtractor(),
}


@dataclass
class ImportReport:
    discovered: int = 0
    supported: int = 0
    extracted: int = 0
    changed: int = 0
    unchanged: int = 0
    duplicates: int = 0
    warnings: int = 0
    failed: int = 0
    candidates_created: int = 0
    pending_review: int = 0
    unsupported: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "files_discovered": self.discovered,
            "supported": self.supported,
            "extracted": self.extracted,
            "changed": self.changed,
            "unchanged": self.unchanged,
            "duplicates": self.duplicates,
            "warnings": self.warnings,
            "failed": self.failed,
            "candidates_created": self.candidates_created,
            "pending_review": self.pending_review,
            "unsupported": self.unsupported,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class ImportPaths:
    root: Path
    manifest: Path
    normalized: Path
    candidates: Path

    @classmethod
    def under(cls, root: Path) -> "ImportPaths":
        return cls(
            root=root,
            manifest=root / "manifests" / "sources.json",
            normalized=root / "normalized",
            candidates=root / "candidates.json",
        )


def _document_type_for_suffix(suffix: str) -> DocumentType:
    return {
        ".pptx": DocumentType.PPTX,
        ".docx": DocumentType.DOCX,
        ".pdf": DocumentType.PDF,
        ".md": DocumentType.MARKDOWN,
        ".txt": DocumentType.TEXT,
    }.get(suffix, DocumentType.UNKNOWN)


def _save_normalized(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document.model_dump(mode="json"), indent=2), encoding="utf-8")


def _load_normalized(path: Path) -> Any:
    from lecture_slm.ingestion.models import NormalizedDocument

    return NormalizedDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _discovery_message(discovery: DiscoveryResult) -> str:
    return f"Discovered {len(discovery.files)} files ({len(discovery.supported)} supported)"


def import_teaching_materials(
    supplied_path: Path,
    *,
    data_dir: Path = Path("data/ingestion"),
    recursive: bool = True,
    course: str | None = None,
    level: str | None = None,
    dry_run: bool = False,
    force_reextract: bool = False,
) -> ImportReport:
    """Import explicitly supplied local files without contacting a model or cloud service."""

    discovery = discover_sources(supplied_path, recursive=recursive)
    report = ImportReport(discovered=len(discovery.files), supported=len(discovery.supported))
    report.unsupported = [file.relative_path for file in discovery.unsupported]
    LOGGER.info("%s", _discovery_message(discovery))
    paths = ImportPaths.under(data_dir)
    manifest = ManifestStore(paths.manifest)
    candidates = CandidateStore(paths.candidates)

    for discovered in discovery.supported:
        extractor = EXTRACTORS[discovered.path.suffix.lower()]
        sha256 = sha256_file(discovered.path)
        source_id = source_id_for_hash(sha256)
        existing_hash = manifest.by_hash(sha256)
        latest_path = manifest.latest_for_path(discovered.relative_path)
        if (
            existing_hash
            and existing_hash.extraction_status
            in {
                ExtractionStatus.SUCCESS,
                ExtractionStatus.SUCCESS_WITH_WARNINGS,
                ExtractionStatus.UNCHANGED,
                ExtractionStatus.DUPLICATE,
            }
            and not force_reextract
        ):
            if latest_path and latest_path.sha256 == sha256:
                report.unchanged += 1
                if not dry_run:
                    manifest.records[manifest.records.index(latest_path)] = mark_unchanged(
                        latest_path
                    )
                continue
            report.duplicates += 1
            if dry_run:
                continue
            duplicate_record = SourceManifest(
                source_id=source_id,
                relative_path=discovered.relative_path,
                filename=discovered.path.name,
                extension=discovered.path.suffix.lower(),
                file_size=discovered.path.stat().st_size,
                modified_time=datetime.fromtimestamp(discovered.path.stat().st_mtime, UTC),
                sha256=sha256,
                extractor=extractor.name,
                extractor_version=extractor.version,
                extraction_status=ExtractionStatus.DUPLICATE,
                imported_at=datetime.now(UTC),
                document_type=_document_type_for_suffix(discovered.path.suffix.lower()),
                course=course,
                normalized_path=existing_hash.normalized_path,
                canonical_source_id=existing_hash.canonical_source_id or existing_hash.source_id,
                notes=["Byte-identical source; canonical content is already extracted"],
            )
            manifest.add(duplicate_record)
            continue

        version = next_source_version(manifest, discovered.relative_path)
        if latest_path and latest_path.sha256 != sha256:
            report.changed += 1
        canonical = existing_hash.canonical_source_id if existing_hash else source_id
        record_base: dict[str, Any] = dict(
            source_id=source_id,
            relative_path=discovered.relative_path,
            filename=discovered.path.name,
            extension=discovered.path.suffix.lower(),
            file_size=discovered.path.stat().st_size,
            modified_time=datetime.fromtimestamp(discovered.path.stat().st_mtime, UTC),
            sha256=sha256,
            extractor=extractor.name,
            extractor_version=extractor.version,
            imported_at=datetime.now(UTC),
            document_type=_document_type_for_suffix(discovered.path.suffix.lower()),
            course=course,
            previous_source_id=(
                latest_path.source_id if latest_path and latest_path.sha256 != sha256 else None
            ),
            canonical_source_id=canonical,
        )
        if dry_run:
            continue
        try:
            document = extractor.extract(
                discovered.path,
                source_id=source_id,
                source_hash=sha256,
                relative_path=discovered.relative_path,
                source_version=version,
            )
            normalized_path = paths.normalized / f"{source_id}.json"
            _save_normalized(normalized_path, document)
            record = SourceManifest(
                **record_base,
                extraction_status=document.extraction_quality,
                provenance=document.provenance,
                notes=document.extraction_warnings,
                normalized_path=normalized_path.as_posix(),
            )
            manifest.add(record)
            report.extracted += 1
            report.warnings += len(document.extraction_warnings)
            if (
                document.extraction_quality
                in {
                    ExtractionStatus.SUCCESS,
                    ExtractionStatus.SUCCESS_WITH_WARNINGS,
                }
                and canonical == source_id
            ):
                candidate = build_candidate_from_path(
                    document,
                    discovered.path.as_posix(),
                    course=course,
                    level=level,
                )
                if candidate and candidates.upsert(candidate):
                    report.candidates_created += 1
        except Exception as error:  # noqa: BLE001 - one bad source must not stop a batch
            message = f"{discovered.relative_path}: {type(error).__name__}: {error}"
            report.failed += 1
            report.errors.append(message)
            LOGGER.warning("Extraction failed for %s", message)
            manifest.add(
                SourceManifest(
                    **record_base,
                    extraction_status=ExtractionStatus.FAILED,
                    notes=[str(error)],
                )
            )

    if not dry_run:
        manifest.save()
        candidates.save()
    report.pending_review = len(candidates.list(status=ReviewStatus.PENDING))
    return report
