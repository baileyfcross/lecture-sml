"""Typed records used by local Workspace storage and APIs."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _require_nonblank(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must contain non-whitespace characters")
    return normalized


class WorkspaceItemRole(StrEnum):
    CONTEXT = "context"
    HISTORY = "history"
    REFERENCE = "reference"


class WorkspaceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime


class WorkspaceFolder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    workspace_id: str
    parent_id: str | None = None
    name: str
    created_at: datetime
    updated_at: datetime


class WorkspaceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    workspace_id: str
    folder_id: str | None = None
    title: str
    role: WorkspaceItemRole
    content: str
    pinned: bool = False
    source_request_id: str | None = None
    created_at: datetime
    updated_at: datetime


class WorkspaceDetail(WorkspaceSummary):
    folders: list[WorkspaceFolder] = Field(default_factory=list)
    items: list[WorkspaceItem] = Field(default_factory=list)


class WorkspaceContextItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    role: WorkspaceItemRole
    title: str
    content: str


class WorkspaceContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    workspace_name: str
    items: list[WorkspaceContextItem] = Field(default_factory=list)


class WorkspaceItemInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=300)
    folder_id: str | None = None
    role: WorkspaceItemRole = WorkspaceItemRole.CONTEXT
    content: str = Field(min_length=1)
    pinned: bool = False
    source_request_id: str | None = None

    @field_validator("title")
    @classmethod
    def title_is_not_blank(cls, value: str) -> str:
        return _require_nonblank(value)

    @field_validator("content")
    @classmethod
    def content_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must contain non-whitespace characters")
        return value


class WorkspaceFolderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    parent_id: str | None = None

    @field_validator("name")
    @classmethod
    def name_is_not_blank(cls, value: str) -> str:
        return _require_nonblank(value)


class WorkspaceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    description: str | None = None

    @field_validator("name")
    @classmethod
    def name_is_not_blank(cls, value: str) -> str:
        return _require_nonblank(value)


class WorkspaceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None

    @field_validator("name")
    @classmethod
    def name_is_not_blank(cls, value: str | None) -> str | None:
        return None if value is None else _require_nonblank(value)
