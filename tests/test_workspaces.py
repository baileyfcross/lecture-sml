from pathlib import Path

import pytest

from lecture_slm.generation.models import GenerationProfileName
from lecture_slm.workspaces.config import WorkspaceConfig
from lecture_slm.workspaces.models import (
    WorkspaceContext,
    WorkspaceItemInput,
    WorkspaceItemRole,
)
from lecture_slm.workspaces.service import WorkspaceService
from lecture_slm.workspaces.storage import (
    WorkspaceConflictError,
    WorkspaceNotFoundError,
    WorkspaceStore,
)


def test_workspace_storage_scopes_search_and_enforces_folder_safety(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "workspaces.sqlite")
    first = store.create_workspace("CSC 220")
    second = store.create_workspace("CSC 320")
    foreign_folder = store.create_folder(second.id, "Other course")
    parent = store.create_folder(first.id, "Lectures")
    child = store.create_folder(first.id, "Week 1", parent.id)
    item = store.create_item(
        first.id,
        WorkspaceItemInput(
            title="Predicate logic",
            role=WorkspaceItemRole.REFERENCE,
            folder_id=child.id,
            content="Predicates describe properties of objects.",
        ),
    )
    store.create_item(
        second.id,
        WorkspaceItemInput(
            title="Private item",
            role=WorkspaceItemRole.REFERENCE,
            content="Private knowledge belongs to a separate Workspace.",
        ),
    )

    assert [result.id for result in store.search_items(first.id, "predicate logic")] == [item.id]
    assert store.search_items(first.id, "private knowledge") == []
    with pytest.raises(WorkspaceConflictError, match="not empty"):
        store.delete_folder(first.id, parent.id)
    with pytest.raises(WorkspaceConflictError, match="inside itself"):
        store.update_folder(first.id, parent.id, name="Lectures", parent_id=child.id, move=True)
    with pytest.raises(WorkspaceNotFoundError):
        store.create_item(
            first.id,
            WorkspaceItemInput(
                title="Cross-scope",
                folder_id=foreign_folder.id,
                content="Must not be attached across Workspaces.",
            ),
        )


def test_workspace_selection_separates_continuity_from_reference_sources(
    tmp_path: Path,
) -> None:
    config = WorkspaceConfig()
    service = WorkspaceService(config, database_path=tmp_path / "workspaces.sqlite")
    workspace = service.store.create_workspace("CSC 220 Logic")
    context = service.store.create_item(
        workspace.id,
        WorkspaceItemInput(
            title="Class convention",
            role=WorkspaceItemRole.CONTEXT,
            content="Learners use truth tables before natural deduction.",
        ),
    )
    history = service.store.create_item(
        workspace.id,
        WorkspaceItemInput(
            title="Previous session",
            role=WorkspaceItemRole.HISTORY,
            content="The previous session introduced propositional variables.",
        ),
    )
    reference = service.store.create_item(
        workspace.id,
        WorkspaceItemInput(
            title="Course reference",
            role=WorkspaceItemRole.REFERENCE,
            content="A proposition is a declarative statement with a truth value.",
        ),
    )

    selected = service.select_generation_context(
        workspace.id,
        "Explain propositions with a truth table",
        GenerationProfileName.QUICK,
        source_budget=512,
    )

    assert isinstance(selected.continuity, WorkspaceContext)
    assert {item.id for item in selected.continuity.items} == {context.id, history.id}
    assert [material.metadata["workspace_item_id"] for material in selected.references] == [
        reference.id
    ]
    assert selected.references[0].metadata["source_origin"] == "workspace_reference"
    assert all(
        source.metadata["workspace_item_id"] not in {context.id, history.id}
        for source in selected.references
    )
    assert service.anchor_retrieval_query("truth table logic", workspace.name) == (
        "CSC 220 Logic truth table logic"
    )
    assert (
        service.anchor_retrieval_query("csc 220 logic truth table", workspace.name)
        == "csc 220 logic truth table"
    )
    assert service.anchor_retrieval_query("csc 2201 logic", workspace.name) == (
        "CSC 220 Logic csc 2201 logic"
    )


def test_workspace_context_and_reference_budgets_are_bounded(tmp_path: Path) -> None:
    config = WorkspaceConfig.model_validate(
        {
            "context_budgets": {"quick": 1, "standard": 10, "deep": 10},
            "selection": {"recent_history_items": 0, "search_top_k": 6},
        }
    )
    service = WorkspaceService(config, database_path=tmp_path / "workspaces.sqlite")
    workspace = service.store.create_workspace("CSC 220")
    for role in WorkspaceItemRole:
        service.store.create_item(
            workspace.id,
            WorkspaceItemInput(
                title=f"{role.value} item",
                role=role,
                content="Long continuity/reference content " * 30,
            ),
        )

    selected = service.select_generation_context(
        workspace.id,
        "continuity reference",
        GenerationProfileName.QUICK,
        source_budget=1,
    )

    assert selected.continuity.items == []
    assert selected.references == []
    assert selected.diagnostics["continuity_estimated_tokens"] <= 1
    assert selected.diagnostics["reference_estimated_tokens"] <= 1
