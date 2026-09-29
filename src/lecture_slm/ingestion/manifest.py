"""Persistent source manifest and content identity helpers."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from lecture_slm.ingestion.models import ExtractionStatus, SourceManifest


class ManifestStore:
    """JSON manifest store with atomic replacement and version lookup."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[SourceManifest] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.records = []
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.records = [SourceManifest.model_validate(item) for item in payload]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps([record.model_dump(mode="json") for record in self.records], indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def by_path(self, relative_path: str) -> list[SourceManifest]:
        return [record for record in self.records if record.relative_path == relative_path]

    def by_hash(self, sha256: str) -> SourceManifest | None:
        return next((record for record in self.records if record.sha256 == sha256), None)

    def latest_for_path(self, relative_path: str) -> SourceManifest | None:
        records = self.by_path(relative_path)
        return max(
            records,
            key=lambda record: record.imported_at or record.discovered_at,
            default=None,
        )

    def add(self, record: SourceManifest) -> None:
        self.records.append(record)


def sha256_file(path: Path) -> str:
    """Hash original bytes, never normalized extracted text."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_id_for_hash(sha256: str) -> str:
    return f"src-{sha256}"


def next_source_version(store: ManifestStore, relative_path: str) -> int:
    records = store.by_path(relative_path)
    return (
        max(
            (record.provenance.source_version for record in records if record.provenance),
            default=0,
        )
        + 1
    )


def mark_unchanged(record: SourceManifest) -> SourceManifest:
    return record.model_copy(
        update={"extraction_status": ExtractionStatus.UNCHANGED, "discovered_at": datetime.now(UTC)}
    )
