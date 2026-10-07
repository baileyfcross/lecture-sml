"""SQLite storage for isolated per-Workspace folders and text items."""

import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from lecture_slm.workspaces.models import (
    WorkspaceDetail,
    WorkspaceFolder,
    WorkspaceItem,
    WorkspaceItemInput,
    WorkspaceItemRole,
    WorkspaceSummary,
)


class WorkspaceNotFoundError(KeyError):
    """Raised when a Workspace-owned record cannot be found in its owner scope."""


class WorkspaceConflictError(ValueError):
    """Raised for unsafe or conflicting Workspace record mutations."""


class WorkspaceStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workspace_folders (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    parent_id TEXT,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (workspace_id, id),
                    FOREIGN KEY (workspace_id, parent_id)
                        REFERENCES workspace_folders(workspace_id, id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS workspace_items (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    folder_id TEXT,
                    title TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('context', 'history', 'reference')),
                    content TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0, 1)),
                    source_request_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (workspace_id, folder_id)
                        REFERENCES workspace_folders(workspace_id, id) ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS workspace_items_scope
                    ON workspace_items(workspace_id, role, updated_at DESC);
                CREATE VIRTUAL TABLE IF NOT EXISTS workspace_items_fts USING fts5(
                    item_id UNINDEXED,
                    workspace_id UNINDEXED,
                    title,
                    content,
                    tokenize='unicode61 remove_diacritics 2'
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _summary(row: sqlite3.Row) -> WorkspaceSummary:
        return WorkspaceSummary.model_validate(dict(row))

    @staticmethod
    def _folder(row: sqlite3.Row) -> WorkspaceFolder:
        return WorkspaceFolder.model_validate(dict(row))

    @staticmethod
    def _item(row: sqlite3.Row) -> WorkspaceItem:
        values = dict(row)
        values["pinned"] = bool(values["pinned"])
        return WorkspaceItem.model_validate(values)

    def list_workspaces(self) -> list[WorkspaceSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM workspaces ORDER BY name COLLATE NOCASE, id"
            ).fetchall()
        return [self._summary(row) for row in rows]

    def create_workspace(self, name: str, description: str | None = None) -> WorkspaceSummary:
        identifier = str(uuid4())
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO workspaces (id, name, description, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (identifier, name.strip(), description, now, now),
            )
            row = connection.execute(
                "SELECT * FROM workspaces WHERE id = ?", (identifier,)
            ).fetchone()
        return self._summary(row)

    def get_workspace(self, workspace_id: str) -> WorkspaceDetail:
        with self._connect() as connection:
            workspace = connection.execute(
                "SELECT * FROM workspaces WHERE id = ?", (workspace_id,)
            ).fetchone()
            if workspace is None:
                raise WorkspaceNotFoundError(workspace_id)
            folders = connection.execute(
                "SELECT * FROM workspace_folders WHERE workspace_id = ? "
                "ORDER BY COALESCE(parent_id, ''), name COLLATE NOCASE, id",
                (workspace_id,),
            ).fetchall()
            items = connection.execute(
                "SELECT * FROM workspace_items WHERE workspace_id = ? ORDER BY updated_at DESC, id",
                (workspace_id,),
            ).fetchall()
        return WorkspaceDetail(
            **self._summary(workspace).model_dump(),
            folders=[self._folder(row) for row in folders],
            items=[self._item(row) for row in items],
        )

    def update_workspace(
        self,
        workspace_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        description_is_set: bool = False,
    ) -> WorkspaceSummary:
        update_description = description_is_set or description is not None
        if name is None and not update_description:
            row = self._workspace_row(workspace_id)
            return self._summary(row)
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE workspaces SET "
                "name = CASE WHEN ? THEN ? ELSE name END, "
                "description = CASE WHEN ? THEN ? ELSE description END, "
                "updated_at = ? WHERE id = ?",
                (
                    int(name is not None),
                    None if name is None else name.strip(),
                    int(update_description),
                    description,
                    self._now(),
                    workspace_id,
                ),
            )
            if cursor.rowcount == 0:
                raise WorkspaceNotFoundError(workspace_id)
            row = connection.execute(
                "SELECT * FROM workspaces WHERE id = ?", (workspace_id,)
            ).fetchone()
        return self._summary(row)

    def _workspace_row(self, workspace_id: str) -> sqlite3.Row:
        with self._connect() as connection:
            row: sqlite3.Row | None = connection.execute(
                "SELECT * FROM workspaces WHERE id = ?", (workspace_id,)
            ).fetchone()
        if row is None:
            raise WorkspaceNotFoundError(workspace_id)
        return row

    def delete_workspace(self, workspace_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))
            if cursor.rowcount == 0:
                raise WorkspaceNotFoundError(workspace_id)
            connection.execute(
                "DELETE FROM workspace_items_fts WHERE workspace_id = ?", (workspace_id,)
            )

    def create_folder(
        self,
        workspace_id: str,
        name: str,
        parent_id: str | None = None,
    ) -> WorkspaceFolder:
        identifier = str(uuid4())
        now = self._now()
        with self._connect() as connection:
            self._require_workspace(connection, workspace_id)
            if (
                parent_id is not None
                and connection.execute(
                    "SELECT 1 FROM workspace_folders WHERE workspace_id = ? AND id = ?",
                    (workspace_id, parent_id),
                ).fetchone()
                is None
            ):
                raise WorkspaceNotFoundError(parent_id)
            if self._folder_name_exists(connection, workspace_id, parent_id, name):
                raise WorkspaceConflictError("a folder with that name already exists here")
            connection.execute(
                "INSERT INTO workspace_folders "
                "(id, workspace_id, parent_id, name, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (identifier, workspace_id, parent_id, name.strip(), now, now),
            )
            row = connection.execute(
                "SELECT * FROM workspace_folders WHERE id = ?", (identifier,)
            ).fetchone()
        return self._folder(row)

    @staticmethod
    def _folder_name_exists(
        connection: sqlite3.Connection,
        workspace_id: str,
        parent_id: str | None,
        name: str,
        *,
        exclude_id: str | None = None,
    ) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM workspace_folders WHERE workspace_id = ? "
                "AND parent_id IS ? AND name = ? COLLATE NOCASE AND id IS NOT ?",
                (workspace_id, parent_id, name.strip(), exclude_id),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _require_workspace(connection: sqlite3.Connection, workspace_id: str) -> None:
        if (
            connection.execute("SELECT 1 FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()
            is None
        ):
            raise WorkspaceNotFoundError(workspace_id)

    def update_folder(
        self,
        workspace_id: str,
        folder_id: str,
        *,
        name: str,
        parent_id: str | None = None,
        move: bool = False,
    ) -> WorkspaceFolder:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM workspace_folders WHERE workspace_id = ? AND id = ?",
                (workspace_id, folder_id),
            ).fetchone()
            if row is None:
                raise WorkspaceNotFoundError(folder_id)
            target_parent = parent_id if move else row["parent_id"]
            if (
                target_parent is not None
                and connection.execute(
                    "SELECT 1 FROM workspace_folders WHERE workspace_id = ? AND id = ?",
                    (workspace_id, target_parent),
                ).fetchone()
                is None
            ):
                raise WorkspaceNotFoundError(target_parent)
            if target_parent == folder_id or self._is_descendant(
                connection, workspace_id, target_parent, folder_id
            ):
                raise WorkspaceConflictError("a folder cannot be moved inside itself")
            if self._folder_name_exists(
                connection,
                workspace_id,
                target_parent,
                name,
                exclude_id=folder_id,
            ):
                raise WorkspaceConflictError("a folder with that name already exists here")
            connection.execute(
                "UPDATE workspace_folders SET name = ?, parent_id = ?, updated_at = ? "
                "WHERE workspace_id = ? AND id = ?",
                (name.strip(), target_parent, self._now(), workspace_id, folder_id),
            )
            updated = connection.execute(
                "SELECT * FROM workspace_folders WHERE id = ?", (folder_id,)
            ).fetchone()
        return self._folder(updated)

    @staticmethod
    def _is_descendant(
        connection: sqlite3.Connection,
        workspace_id: str,
        possible_descendant: str | None,
        ancestor_id: str,
    ) -> bool:
        current = possible_descendant
        while current is not None:
            if current == ancestor_id:
                return True
            row = connection.execute(
                "SELECT parent_id FROM workspace_folders WHERE workspace_id = ? AND id = ?",
                (workspace_id, current),
            ).fetchone()
            current = None if row is None else row["parent_id"]
        return False

    def delete_folder(self, workspace_id: str, folder_id: str) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM workspace_folders WHERE workspace_id = ? AND id = ?",
                (workspace_id, folder_id),
            ).fetchone()
            if row is None:
                raise WorkspaceNotFoundError(folder_id)
            populated = connection.execute(
                "SELECT EXISTS(SELECT 1 FROM workspace_folders "
                "WHERE workspace_id = ? AND parent_id = ?) OR "
                "EXISTS(SELECT 1 FROM workspace_items "
                "WHERE workspace_id = ? AND folder_id = ?)",
                (workspace_id, folder_id, workspace_id, folder_id),
            ).fetchone()[0]
            if populated:
                raise WorkspaceConflictError("folder is not empty")
            connection.execute(
                "DELETE FROM workspace_folders WHERE workspace_id = ? AND id = ?",
                (workspace_id, folder_id),
            )

    def create_item(self, workspace_id: str, item: WorkspaceItemInput) -> WorkspaceItem:
        identifier = str(uuid4())
        now = self._now()
        with self._connect() as connection:
            self._require_workspace(connection, workspace_id)
            self._validate_folder(connection, workspace_id, item.folder_id)
            connection.execute(
                "INSERT INTO workspace_items "
                "(id, workspace_id, folder_id, title, role, content, pinned, "
                "source_request_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    identifier,
                    workspace_id,
                    item.folder_id,
                    item.title.strip(),
                    item.role.value,
                    item.content,
                    int(item.pinned),
                    item.source_request_id,
                    now,
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO workspace_items_fts (item_id, workspace_id, title, content) "
                "VALUES (?, ?, ?, ?)",
                (identifier, workspace_id, item.title, item.content),
            )
            row = connection.execute(
                "SELECT * FROM workspace_items WHERE id = ?", (identifier,)
            ).fetchone()
        return self._item(row)

    @staticmethod
    def _validate_folder(
        connection: sqlite3.Connection,
        workspace_id: str,
        folder_id: str | None,
    ) -> None:
        if folder_id is None:
            return
        if (
            connection.execute(
                "SELECT 1 FROM workspace_folders WHERE workspace_id = ? AND id = ?",
                (workspace_id, folder_id),
            ).fetchone()
            is None
        ):
            raise WorkspaceNotFoundError(folder_id)

    def update_item(
        self,
        workspace_id: str,
        item_id: str,
        updates: dict[str, object],
    ) -> WorkspaceItem:
        allowed = {"folder_id", "title", "role", "content", "pinned"}
        if set(updates) - allowed:
            raise WorkspaceConflictError("item updates cannot change workspace ownership")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM workspace_items WHERE workspace_id = ? AND id = ?",
                (workspace_id, item_id),
            ).fetchone()
            if row is None:
                raise WorkspaceNotFoundError(item_id)
            if "folder_id" in updates:
                folder_id = updates["folder_id"]
                self._validate_folder(
                    connection,
                    workspace_id,
                    folder_id if isinstance(folder_id, str) else None,
                )
            values = dict(updates)
            role = values.get("role")
            if isinstance(role, WorkspaceItemRole):
                values["role"] = role.value
            connection.execute(
                "UPDATE workspace_items SET "
                "folder_id = CASE WHEN ? THEN ? ELSE folder_id END, "
                "title = CASE WHEN ? THEN ? ELSE title END, "
                "role = CASE WHEN ? THEN ? ELSE role END, "
                "content = CASE WHEN ? THEN ? ELSE content END, "
                "pinned = CASE WHEN ? THEN ? ELSE pinned END, "
                "updated_at = ? WHERE workspace_id = ? AND id = ?",
                (
                    int("folder_id" in values),
                    values.get("folder_id"),
                    int("title" in values),
                    values.get("title"),
                    int("role" in values),
                    values.get("role"),
                    int("content" in values),
                    values.get("content"),
                    int("pinned" in values),
                    values.get("pinned"),
                    self._now(),
                    workspace_id,
                    item_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM workspace_items WHERE id = ?", (item_id,)
            ).fetchone()
            if "title" in updates or "content" in updates:
                connection.execute(
                    "UPDATE workspace_items_fts SET title = ?, content = ? "
                    "WHERE item_id = ? AND workspace_id = ?",
                    (
                        updated["title"],
                        updated["content"],
                        item_id,
                        workspace_id,
                    ),
                )
        return self._item(updated)

    def delete_item(self, workspace_id: str, item_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM workspace_items WHERE workspace_id = ? AND id = ?",
                (workspace_id, item_id),
            )
            if cursor.rowcount == 0:
                raise WorkspaceNotFoundError(item_id)
            connection.execute(
                "DELETE FROM workspace_items_fts WHERE item_id = ? AND workspace_id = ?",
                (item_id, workspace_id),
            )

    def search_items(
        self,
        workspace_id: str,
        query: str,
        *,
        roles: set[WorkspaceItemRole] | None = None,
        limit: int = 6,
    ) -> list[WorkspaceItem]:
        terms = list(dict.fromkeys(re.findall(r"[\w.+#-]+", query.casefold())))
        if not terms:
            return []
        match_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"*' for term in terms)
        role_values: list[object] = []
        if roles is not None:
            role_values.extend(role.value for role in roles)
        else:
            role_values.extend(role.value for role in WorkspaceItemRole)
        role_values.extend([None] * (len(WorkspaceItemRole) - len(role_values)))
        filter_roles = int(roles is not None)
        parameters: list[object] = [workspace_id, match_query]
        parameters.extend([filter_roles, *role_values])
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT i.* FROM workspace_items_fts f "
                "JOIN workspace_items i ON i.id = f.item_id "
                "WHERE f.workspace_id = ? AND workspace_items_fts MATCH ? "
                "AND (? = 0 OR i.role IN (?, ?, ?)) "
                "ORDER BY bm25(workspace_items_fts), i.updated_at DESC "
                "LIMIT ?",
                parameters,
            ).fetchall()
        return [self._item(row) for row in rows]

    def list_items(
        self,
        workspace_id: str,
        *,
        roles: set[WorkspaceItemRole] | None = None,
        pinned: bool | None = None,
    ) -> list[WorkspaceItem]:
        role_values: list[object] = []
        if roles is not None:
            role_values.extend(sorted(role.value for role in roles))
        else:
            role_values.extend(role.value for role in WorkspaceItemRole)
        role_values.extend([None] * (len(WorkspaceItemRole) - len(role_values)))
        with self._connect() as connection:
            self._require_workspace(connection, workspace_id)
            rows = connection.execute(
                "SELECT * FROM workspace_items WHERE workspace_id = ? "
                "AND (? = 0 OR role IN (?, ?, ?)) "
                "AND (? = 0 OR pinned = ?) "
                "ORDER BY updated_at DESC, id",
                (
                    workspace_id,
                    int(roles is not None),
                    *role_values,
                    int(pinned is not None),
                    int(pinned) if pinned is not None else 0,
                ),
            ).fetchall()
        return [self._item(row) for row in rows]
