"""Durable candidate queue and explicit review transitions."""

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from lecture_slm.ingestion.models import CandidateTrainingExample, ReviewStatus


class CandidateStore:
    """Atomic JSON queue; every update rewrites the complete validated state."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.candidates: dict[str, CandidateTrainingExample] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.candidates = {}
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.candidates = {
            candidate.candidate_id: candidate
            for candidate in (CandidateTrainingExample.model_validate(item) for item in payload)
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                [candidate.model_dump(mode="json") for candidate in self.candidates.values()],
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def upsert(self, candidate: CandidateTrainingExample) -> bool:
        """Insert only new candidate identities; return whether it was inserted."""

        if candidate.candidate_id in self.candidates:
            return False
        self.candidates[candidate.candidate_id] = candidate
        return True

    def get(self, candidate_id: str) -> CandidateTrainingExample:
        try:
            return self.candidates[candidate_id]
        except KeyError as error:
            raise KeyError(f"Unknown candidate: {candidate_id}") from error

    def list(
        self,
        *,
        status: ReviewStatus | None = None,
        task: str | None = None,
        course: str | None = None,
    ) -> list[CandidateTrainingExample]:
        values: Iterable[CandidateTrainingExample] = self.candidates.values()
        if status is not None:
            values = (candidate for candidate in values if candidate.review_status is status)
        if task is not None:
            values = (
                candidate for candidate in values if candidate.task and candidate.task.value == task
            )
        if course is not None:
            values = (candidate for candidate in values if candidate.course == course)
        return sorted(values, key=lambda candidate: candidate.candidate_id)

    def transition(
        self,
        candidate_id: str,
        status: ReviewStatus,
        *,
        reviewer: str | None = None,
        notes: str | None = None,
    ) -> CandidateTrainingExample:
        candidate = self.get(candidate_id)
        candidate.transition(status, reviewer=reviewer, notes=notes)
        self.save()
        return candidate

    def update_fields(self, candidate_id: str, **fields: object) -> CandidateTrainingExample:
        candidate = self.get(candidate_id)
        for field, value in fields.items():
            if field in {"candidate_id", "source_ids", "provenance", "status_history"}:
                raise ValueError(f"Field cannot be edited in review: {field}")
            if not hasattr(candidate, field):
                raise ValueError(f"Unknown candidate field: {field}")
            setattr(candidate, field, value)
        candidate.updated_at = datetime.now(UTC)
        self.save()
        return candidate


def validate_candidate_store(path: Path) -> list[CandidateTrainingExample]:
    store = CandidateStore(path)
    return list(store.candidates.values())


def iter_candidates(path: Path) -> Iterable[CandidateTrainingExample]:
    return CandidateStore(path).candidates.values()
