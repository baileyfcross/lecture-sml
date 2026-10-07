"""Workspace persistence and context-selection configuration."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class WorkspaceStorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database_path: Path = Path("data/workspaces/workspaces.sqlite")


class WorkspaceContextBudgets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quick: int = Field(default=2048, gt=0)
    standard: int = Field(default=4096, gt=0)
    deep: int = Field(default=8192, gt=0)

    def for_profile(self, profile: str) -> int:
        if profile == "quick":
            return self.quick
        if profile == "standard":
            return self.standard
        if profile == "deep":
            return self.deep
        raise ValueError(f"Unknown generation profile for Workspace context: {profile}")


class WorkspaceSelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recent_history_items: int = Field(default=2, ge=0)
    search_top_k: int = Field(default=6, gt=0)
    max_pinned_items: int = Field(default=8, ge=0)


class WorkspaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    storage: WorkspaceStorageConfig = Field(default_factory=WorkspaceStorageConfig)
    context_budgets: WorkspaceContextBudgets = Field(default_factory=WorkspaceContextBudgets)
    selection: WorkspaceSelectionConfig = Field(default_factory=WorkspaceSelectionConfig)


def load_workspace_config(path: Path = Path("configs/workspaces/default.yaml")) -> WorkspaceConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Workspace configuration not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Workspace configuration must be a YAML mapping: {path}")
    return WorkspaceConfig.model_validate(loaded)
