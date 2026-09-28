"""Dataset manifest models for reproducible experiments."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class DatasetManifest(BaseModel):
    """A small, committed description of the data used by an experiment."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    files: list[str] = Field(default_factory=list)
    splits: dict[str, int] = Field(default_factory=dict)
    notes: str | None = None
