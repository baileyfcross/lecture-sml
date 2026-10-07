"""Workspace operations and bounded, role-aware generation context selection."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lecture_slm.generation.context import estimate_tokens_from_characters
from lecture_slm.generation.models import GenerationProfileName, SourceMaterial
from lecture_slm.knowledge.query import canonicalize_retrieval_query
from lecture_slm.workspaces.config import WorkspaceConfig, load_workspace_config
from lecture_slm.workspaces.models import (
    WorkspaceContext,
    WorkspaceContextItem,
    WorkspaceFolder,
    WorkspaceFolderInput,
    WorkspaceItem,
    WorkspaceItemInput,
    WorkspaceItemRole,
)
from lecture_slm.workspaces.storage import WorkspaceStore


@dataclass(frozen=True)
class WorkspaceGenerationContext:
    continuity: WorkspaceContext
    references: list[SourceMaterial]
    diagnostics: dict[str, Any]


def _item_tokens(item: WorkspaceItem, workspace_name: str = "") -> int:
    return estimate_tokens_from_characters(
        f"{workspace_name}: {item.title}\n{item.role.value}\n{item.content}"
    )


class WorkspaceService:
    def __init__(
        self,
        config: WorkspaceConfig | None = None,
        *,
        database_path: Path | None = None,
    ) -> None:
        self.config = config or load_workspace_config()
        self.store = WorkspaceStore(database_path or self.config.storage.database_path)

    def select_generation_context(
        self,
        workspace_id: str,
        instruction: str,
        profile: GenerationProfileName,
        source_budget: int,
    ) -> WorkspaceGenerationContext:
        workspace = self.store.get_workspace(workspace_id)
        canonical_query = canonicalize_retrieval_query(instruction).canonical
        history = self.store.list_items(
            workspace_id,
            roles={WorkspaceItemRole.HISTORY},
        )
        recent_history = history[: self.config.selection.recent_history_items]
        pinned = [
            item
            for item in self.store.list_items(workspace_id, pinned=True)
            if item.role in {WorkspaceItemRole.CONTEXT, WorkspaceItemRole.HISTORY}
        ][: self.config.selection.max_pinned_items]
        matched_continuity = self.store.search_items(
            workspace_id,
            canonical_query,
            roles={WorkspaceItemRole.CONTEXT, WorkspaceItemRole.HISTORY},
            limit=self.config.selection.search_top_k,
        )
        continuity_candidates = _unique_items([*pinned, *recent_history, *matched_continuity])
        continuity_budget = self.config.context_budgets.for_profile(profile.value)
        selected_continuity, skipped_continuity = _fit_items(
            continuity_candidates,
            continuity_budget,
        )

        pinned_references = [
            item
            for item in self.store.list_items(
                workspace_id,
                roles={WorkspaceItemRole.REFERENCE},
                pinned=True,
            )[: self.config.selection.max_pinned_items]
        ]
        matched_references = self.store.search_items(
            workspace_id,
            canonical_query,
            roles={WorkspaceItemRole.REFERENCE},
            limit=self.config.selection.search_top_k,
        )
        reference_candidates = _unique_items([*pinned_references, *matched_references])
        selected_references, skipped_references = _fit_items(
            reference_candidates,
            max(0, source_budget),
            workspace_name=workspace.name,
        )
        context = WorkspaceContext(
            workspace_id=workspace.id,
            workspace_name=workspace.name,
            items=[
                WorkspaceContextItem(
                    id=item.id,
                    role=item.role,
                    title=item.title,
                    content=item.content,
                )
                for item in selected_continuity
            ],
        )
        materials = [
            SourceMaterial(
                source_id=f"workspace:{workspace.id}:{item.id}",
                title=f"{workspace.name}: {item.title}",
                section=item.role.value,
                text=item.content,
                metadata={
                    "source_origin": "workspace_reference",
                    "workspace_id": workspace.id,
                    "workspace_item_id": item.id,
                    "workspace_item_role": item.role.value,
                },
            )
            for item in selected_references
        ]
        diagnostics = {
            "workspace_id": workspace.id,
            "workspace_name": workspace.name,
            "continuity_budget": continuity_budget,
            "continuity_selected_item_ids": [item.id for item in selected_continuity],
            "continuity_estimated_tokens": sum(_item_tokens(item) for item in selected_continuity),
            "continuity_skipped_item_ids": [item.id for item in skipped_continuity],
            "reference_budget": max(0, source_budget),
            "reference_selected_item_ids": [item.id for item in selected_references],
            "reference_estimated_tokens": sum(
                _item_tokens(item, workspace.name) for item in selected_references
            ),
            "reference_skipped_item_ids": [item.id for item in skipped_references],
        }
        return WorkspaceGenerationContext(context, materials, diagnostics)

    def create_item(self, workspace_id: str, item: WorkspaceItemInput) -> WorkspaceItem:
        return self.store.create_item(workspace_id, item)

    def create_folder(
        self,
        workspace_id: str,
        folder: WorkspaceFolderInput,
    ) -> WorkspaceFolder:
        return self.store.create_folder(workspace_id, folder.name, folder.parent_id)

    @staticmethod
    def anchor_retrieval_query(canonical_query: str, workspace_name: str) -> str:
        normalized_name = " ".join(workspace_name.split())
        name_pattern = re.escape(normalized_name)
        if re.search(
            rf"(?<!\w){name_pattern}(?!\w)",
            canonical_query,
            flags=re.IGNORECASE,
        ):
            return canonical_query
        return f"{normalized_name} {canonical_query}"


def _unique_items(items: list[WorkspaceItem]) -> list[WorkspaceItem]:
    seen: set[str] = set()
    unique: list[WorkspaceItem] = []
    for item in items:
        if item.id not in seen:
            unique.append(item)
            seen.add(item.id)
    return unique


def _fit_items(
    candidates: list[WorkspaceItem],
    budget: int,
    *,
    workspace_name: str = "",
) -> tuple[list[WorkspaceItem], list[WorkspaceItem]]:
    selected: list[WorkspaceItem] = []
    skipped: list[WorkspaceItem] = []
    remaining = budget
    for item in candidates:
        tokens = _item_tokens(item, workspace_name)
        if tokens <= remaining:
            selected.append(item)
            remaining -= tokens
        else:
            skipped.append(item)
    return selected, skipped
