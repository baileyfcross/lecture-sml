"""Transactional SQLite storage for knowledge sources, chunks, and local vectors."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from lecture_slm.knowledge.models import KnowledgeChunk
from lecture_slm.knowledge.roles import ChunkRole

SCHEMA_VERSION = 2


class KnowledgeStore:
    """SQLite-backed metadata, FTS5, and float32 embedding cache."""

    def __init__(
        self,
        data_dir: Path,
        *,
        embedding_model: str,
        embedding_version: str,
        chunking_version: str,
        rebuild: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "knowledge.sqlite"
        if rebuild:
            for stale_path in (
                self.path,
                self.path.with_name(self.path.name + "-wal"),
                self.path.with_name(self.path.name + "-shm"),
            ):
                stale_path.unlink(missing_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        try:
            self._preflight_schema_version()
            self._create_schema()
            self._validate_schema()
            self._validate_index_versions(embedding_model, embedding_version, chunking_version)
        except Exception:
            self.connection.close()
            raise

    @staticmethod
    def indexed_vault_root(data_dir: Path) -> Path:
        """Read the root binding needed to validate any configured model-cache path."""

        database = data_dir / "knowledge.sqlite"
        if not database.is_file():
            raise FileNotFoundError(f"No knowledge index at {database}")
        connection = sqlite3.connect(database)
        try:
            row = connection.execute(
                "SELECT value FROM index_metadata WHERE key='vault_root'"
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise ValueError(f"Knowledge index at {database} has no vault identity metadata")
        return Path(str(row[0]))

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                source_hash TEXT NOT NULL,
                title TEXT NOT NULL,
                document_type TEXT NOT NULL,
                source_version INTEGER NOT NULL,
                extraction_status TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS source_paths (
                relative_path TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(source_id),
                vault_id TEXT NOT NULL,
                modified_time TEXT NOT NULL,
                source_version INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS source_paths_source_id ON source_paths(source_id);
            CREATE TABLE IF NOT EXISTS embedding_cache (
                embedding_key TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                provider_version TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(source_id),
                source_hash TEXT NOT NULL,
                source_version INTEGER NOT NULL,
                title TEXT NOT NULL,
                section_title TEXT,
                section_path_json TEXT NOT NULL,
                text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                previous_chunk_id TEXT,
                next_chunk_id TEXT,
                page_number INTEGER,
                slide_number INTEGER,
                note_path TEXT NOT NULL,
                document_type TEXT NOT NULL,
                role TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                outgoing_links_json TEXT NOT NULL,
                course TEXT,
                metadata_json TEXT NOT NULL,
                approximate_token_count INTEGER NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_version TEXT NOT NULL,
                embedding_key TEXT REFERENCES embedding_cache(embedding_key)
            );
            CREATE INDEX IF NOT EXISTS chunks_source_id ON chunks(source_id);
            CREATE INDEX IF NOT EXISTS chunks_role ON chunks(role);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                source_id UNINDEXED,
                title,
                section_title,
                text,
                tags,
                path,
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO index_metadata(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.connection.commit()

    def _preflight_schema_version(self) -> None:
        existing = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='index_metadata'"
        ).fetchone()
        if not existing:
            return
        row = self.connection.execute(
            "SELECT value FROM index_metadata WHERE key='schema_version'"
        ).fetchone()
        if row is not None and str(row[0]) != str(SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported knowledge schema version {row[0]!r}; expected {SCHEMA_VERSION}. "
                "Run index_knowledge.py --rebuild."
            )

    def _validate_schema(self) -> None:
        version = self.get_metadata("schema_version")
        if version != str(SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported knowledge schema version {version!r}; "
                f"expected {SCHEMA_VERSION}. Run index_knowledge.py --rebuild."
            )

    def _validate_index_versions(
        self, embedding_model: str, embedding_version: str, chunking_version: str
    ) -> None:
        has_sources = self.connection.execute("SELECT 1 FROM sources LIMIT 1").fetchone()
        expected = {
            "embedding_model": embedding_model,
            "embedding_provider_version": embedding_version,
            "chunking_version": chunking_version,
        }
        for key, value in expected.items():
            existing = self.get_metadata(key)
            if has_sources and existing is not None and existing != value:
                raise ValueError(
                    f"Knowledge index {key} is {existing!r}, configured value is {value!r}; "
                    "use --rebuild to re-index with the new version."
                )
            self.set_metadata(key, value)

    def get_metadata(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM index_metadata WHERE key = ?", (key,)
        ).fetchone()
        return str(row["value"]) if row else None

    def set_metadata(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO index_metadata(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.connection.commit()

    def ensure_vault_id(self) -> str:
        if existing := self.get_metadata("vault_id"):
            return existing
        import uuid

        vault_id = str(uuid.uuid4())
        self.set_metadata("vault_id", vault_id)
        return vault_id

    def ensure_vault_root(self, root: Path, *, rebuild: bool = False) -> None:
        root_value = str(root.resolve())
        existing = self.get_metadata("vault_root")
        if existing and existing != root_value and not rebuild:
            raise ValueError(
                "This knowledge index belongs to a different vault root. "
                "Use a separate data_dir or pass --rebuild explicitly."
            )
        self.set_metadata("vault_root", root_value)

    def clear(self) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM chunks_fts")
            self.connection.execute("DELETE FROM chunks")
            self.connection.execute("DELETE FROM source_paths")
            self.connection.execute("DELETE FROM sources")
            self.connection.execute("DELETE FROM embedding_cache")
            for key in (
                "embedding_model",
                "embedding_provider_version",
                "chunking_version",
                "vault_id",
                "vault_root",
                "last_indexed_at",
            ):
                self.connection.execute("DELETE FROM index_metadata WHERE key = ?", (key,))

    def source_path(self, relative_path: str) -> sqlite3.Row | None:
        row = self.connection.execute(
            """
            SELECT p.*, s.source_hash, s.extraction_status, s.source_id,
                   s.source_version AS indexed_version
            FROM source_paths p JOIN sources s USING(source_id)
            WHERE p.relative_path = ?
            """,
            (relative_path,),
        ).fetchone()
        return cast(sqlite3.Row | None, row)

    def source_by_hash(self, source_hash: str) -> sqlite3.Row | None:
        row = self.connection.execute(
            "SELECT * FROM sources WHERE source_hash = ? AND deleted = 0 "
            "AND extraction_status != 'failed' LIMIT 1",
            (source_hash,),
        ).fetchone()
        return cast(sqlite3.Row | None, row)

    def add_source_path(
        self,
        *,
        relative_path: str,
        source_id: str,
        vault_id: str,
        modified_time: datetime,
        source_version: int,
    ) -> None:
        previous = self.connection.execute(
            "SELECT source_id FROM source_paths WHERE relative_path = ?", (relative_path,)
        ).fetchone()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO source_paths(
                    relative_path, source_id, vault_id, modified_time, source_version
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(relative_path) DO UPDATE SET source_id=excluded.source_id,
                    vault_id=excluded.vault_id, modified_time=excluded.modified_time,
                    source_version=excluded.source_version
                """,
                (
                    relative_path,
                    source_id,
                    vault_id,
                    modified_time.isoformat(),
                    source_version,
                ),
            )
            if previous and str(previous["source_id"]) != source_id:
                self._mark_unreferenced_deleted(str(previous["source_id"]))
        primary_path = min(self.paths_for_source(source_id))
        self.update_source_paths(source_id, primary_path)

    def paths_for_source(self, source_id: str) -> list[str]:
        rows = self.connection.execute(
            "SELECT relative_path FROM source_paths WHERE source_id = ? ORDER BY relative_path",
            (source_id,),
        ).fetchall()
        return [str(row["relative_path"]) for row in rows]

    def known_paths(self) -> set[str]:
        return {
            str(row["relative_path"])
            for row in self.connection.execute("SELECT relative_path FROM source_paths")
        }

    def active_source_for_path(self, relative_path: str) -> str | None:
        row = self.source_path(relative_path)
        return str(row["source_id"]) if row and row["extraction_status"] != "failed" else None

    def cache_embedding(self, key: str) -> NDArray[np.float32] | None:
        row = self.connection.execute(
            "SELECT dimension, vector FROM embedding_cache WHERE embedding_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        vector = np.frombuffer(row["vector"], dtype=np.float32)
        if vector.size != row["dimension"]:
            raise ValueError(f"Corrupt embedding cache entry: {key}")
        return vector.copy()

    def save_embedding(
        self,
        key: str,
        vector: NDArray[np.float32],
        *,
        model_name: str,
        provider_version: str,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO embedding_cache
                    (embedding_key, model_name, provider_version, dimension, vector, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    model_name,
                    provider_version,
                    int(vector.size),
                    vector.astype(np.float32, copy=False).tobytes(),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def replace_source(
        self,
        *,
        source_id: str,
        source_hash: str,
        title: str,
        document_type: str,
        source_version: int,
        relative_path: str,
        vault_id: str,
        modified_time: datetime,
        metadata: dict[str, Any],
        warnings: list[str],
        chunks: list[KnowledgeChunk],
        embedding_keys: dict[str, str],
    ) -> None:
        """Replace a source and its FTS records as one per-source transaction."""

        indexed_at = datetime.now(UTC).isoformat()
        with self.connection:
            prior = self.connection.execute(
                "SELECT source_id FROM source_paths WHERE relative_path = ?", (relative_path,)
            ).fetchone()
            if prior and str(prior["source_id"]) != source_id:
                self.connection.execute(
                    "DELETE FROM source_paths WHERE relative_path = ?", (relative_path,)
                )
                self._mark_unreferenced_deleted(str(prior["source_id"]))

            self.connection.execute(
                """
                INSERT INTO sources(
                    source_id, source_hash, title, document_type, source_version,
                    extraction_status, indexed_at, metadata_json, warnings_json, deleted
                ) VALUES (?, ?, ?, ?, ?, 'success', ?, ?, ?, 0)
                ON CONFLICT(source_id) DO UPDATE SET
                    title=excluded.title, document_type=excluded.document_type,
                    source_version=excluded.source_version, extraction_status='success',
                    indexed_at=excluded.indexed_at, metadata_json=excluded.metadata_json,
                    warnings_json=excluded.warnings_json, deleted=0
                """,
                (
                    source_id,
                    source_hash,
                    title,
                    document_type,
                    source_version,
                    indexed_at,
                    json.dumps(metadata, ensure_ascii=False),
                    json.dumps(warnings, ensure_ascii=False),
                ),
            )
            self.connection.execute(
                """
                INSERT INTO source_paths(
                    relative_path, source_id, vault_id, modified_time, source_version
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(relative_path) DO UPDATE SET
                    source_id=excluded.source_id, vault_id=excluded.vault_id,
                    modified_time=excluded.modified_time, source_version=excluded.source_version
                """,
                (
                    relative_path,
                    source_id,
                    vault_id,
                    modified_time.isoformat(),
                    source_version,
                ),
            )
            self.connection.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
            self.connection.execute("DELETE FROM chunks_fts WHERE source_id = ?", (source_id,))
            for chunk in chunks:
                serialized = chunk.model_dump(mode="json")
                serialized["section_path_json"] = json.dumps(chunk.section_path)
                serialized["tags_json"] = json.dumps(chunk.tags)
                serialized["aliases_json"] = json.dumps(chunk.aliases)
                serialized["outgoing_links_json"] = json.dumps(chunk.outgoing_links)
                serialized["metadata_json"] = json.dumps(chunk.metadata, ensure_ascii=False)
                serialized["embedding_key"] = embedding_keys.get(chunk.chunk_id)
                self.connection.execute(
                    """
                    INSERT INTO chunks(
                        chunk_id, source_id, source_hash, source_version, title, section_title,
                        section_path_json, text, chunk_index, previous_chunk_id, next_chunk_id,
                        page_number, slide_number, note_path, document_type, role, tags_json,
                        aliases_json, outgoing_links_json, course, metadata_json,
                        approximate_token_count, embedding_model, embedding_version, embedding_key
                    ) VALUES (
                        :chunk_id, :source_id, :source_hash, :source_version, :title,
                        :section_title,
                        :section_path_json, :text, :chunk_index, :previous_chunk_id, :next_chunk_id,
                        :page_number, :slide_number, :note_path, :document_type, :role, :tags_json,
                        :aliases_json, :outgoing_links_json, :course, :metadata_json,
                        :approximate_token_count, :embedding_model, :embedding_version,
                        :embedding_key
                    )
                    """,
                    serialized,
                )
                self.connection.execute(
                    """
                    INSERT INTO chunks_fts(
                        chunk_id, source_id, title, section_title, text, tags, path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        source_id,
                        chunk.title,
                        chunk.section_title or "",
                        chunk.text,
                        " ".join(chunk.tags),
                        relative_path,
                    ),
                )

    def record_failure(
        self,
        *,
        source_id: str,
        source_hash: str,
        relative_path: str,
        vault_id: str,
        modified_time: datetime,
        source_version: int,
        title: str,
        document_type: str,
        error_type: str,
        message: str,
        preserve_existing_path: bool = False,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO sources(
                    source_id, source_hash, title, document_type, source_version,
                    extraction_status, indexed_at, metadata_json, warnings_json, deleted
                ) VALUES (?, ?, ?, ?, ?, 'failed', ?, ?, ?, 0)
                ON CONFLICT(source_id) DO UPDATE SET
                    extraction_status='failed', indexed_at=excluded.indexed_at,
                    warnings_json=excluded.warnings_json, deleted=0
                """,
                (
                    source_id,
                    source_hash,
                    title,
                    document_type,
                    source_version,
                    now,
                    json.dumps(
                        {
                            "error_type": error_type,
                            "relative_path": relative_path,
                            "retry_eligible": True,
                        }
                    ),
                    json.dumps([message]),
                ),
            )
            previous = self.connection.execute(
                "SELECT source_id FROM source_paths WHERE relative_path = ?", (relative_path,)
            ).fetchone()
            if preserve_existing_path:
                return
            if previous and str(previous["source_id"]) != source_id:
                self.connection.execute(
                    "DELETE FROM source_paths WHERE relative_path = ?", (relative_path,)
                )
                self._mark_unreferenced_deleted(str(previous["source_id"]))
            self.connection.execute(
                """
                INSERT INTO source_paths(
                    relative_path, source_id, vault_id, modified_time, source_version
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(relative_path) DO UPDATE SET source_id=excluded.source_id,
                    vault_id=excluded.vault_id, modified_time=excluded.modified_time,
                    source_version=excluded.source_version
                """,
                (relative_path, source_id, vault_id, modified_time.isoformat(), source_version),
            )

    def remove_missing_paths(self, present_paths: set[str]) -> int:
        removed = 0
        with self.connection:
            rows = self.connection.execute(
                "SELECT relative_path, source_id FROM source_paths"
            ).fetchall()
            for row in rows:
                relative_path = str(row["relative_path"])
                if relative_path not in present_paths:
                    self.connection.execute(
                        "DELETE FROM source_paths WHERE relative_path = ?", (relative_path,)
                    )
                    self._mark_unreferenced_deleted(str(row["source_id"]))
                    remaining = self.connection.execute(
                        "SELECT MIN(relative_path) AS path FROM source_paths WHERE source_id = ?",
                        (row["source_id"],),
                    ).fetchone()
                    if remaining and remaining["path"]:
                        self.update_source_paths(str(row["source_id"]), str(remaining["path"]))
                    removed += 1
        return removed

    def _mark_unreferenced_deleted(self, source_id: str) -> None:
        remaining = self.connection.execute(
            "SELECT 1 FROM source_paths WHERE source_id = ? LIMIT 1", (source_id,)
        ).fetchone()
        if not remaining:
            self.connection.execute(
                "UPDATE sources SET deleted = 1 WHERE source_id = ?", (source_id,)
            )

    def update_source_paths(self, source_id: str, new_primary_path: str) -> None:
        """Refresh path-bearing FTS rows when identical content is renamed."""

        rows = self.connection.execute(
            """
            SELECT chunk_id, title, section_title, text, tags_json
            FROM chunks WHERE source_id = ?
            """,
            (source_id,),
        ).fetchall()
        with self.connection:
            self.connection.execute("DELETE FROM chunks_fts WHERE source_id = ?", (source_id,))
            self.connection.execute(
                "UPDATE chunks SET note_path = ? WHERE source_id = ?", (new_primary_path, source_id)
            )
            for row in rows:
                self.connection.execute(
                    """
                    INSERT INTO chunks_fts(
                        chunk_id, source_id, title, section_title, text, tags, path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["chunk_id"],
                        source_id,
                        row["title"],
                        row["section_title"] or "",
                        row["text"],
                        " ".join(json.loads(row["tags_json"])),
                        new_primary_path,
                    ),
                )

    def fts_available(self) -> bool:
        try:
            self.connection.execute("SELECT count(*) FROM chunks_fts").fetchone()
        except sqlite3.OperationalError:
            return False
        return True

    def active_chunk_rows(self, *, roles: list[ChunkRole] | None = None) -> list[sqlite3.Row]:
        selected_roles = set(roles or [])
        parameters: list[Any] = [int(bool(roles))]
        parameters.extend(int(role in selected_roles) for role in ChunkRole)
        return self.connection.execute(
            """
            SELECT c.*, s.metadata_json AS source_metadata_json,
                   (SELECT MIN(p.relative_path) FROM source_paths p
                    WHERE p.source_id=c.source_id) AS active_path
            FROM chunks c JOIN sources s USING(source_id)
            WHERE s.deleted=0 AND s.extraction_status='success'
              AND c.embedding_key IS NOT NULL
              AND (
                ?=0
                OR (c.role='content' AND ?=1)
                OR (c.role='reference' AND ?=1)
                OR (c.role='metadata' AND ?=1)
                OR (c.role='navigation' AND ?=1)
              )
            ORDER BY c.source_id, c.chunk_index
            """,
            parameters,
        ).fetchall()

    def stats(self) -> dict[str, Any]:
        scalar_queries = {
            "indexed_sources": (
                "SELECT COUNT(DISTINCT p.source_id) FROM source_paths p "
                "JOIN sources s USING(source_id) "
                "WHERE s.deleted=0 AND s.extraction_status='success'"
            ),
            "indexed_chunks": (
                "SELECT COUNT(*) FROM chunks c JOIN sources s USING(source_id) "
                "WHERE s.deleted=0 AND s.extraction_status='success'"
            ),
            "failed_sources": (
                "SELECT COUNT(*) FROM sources WHERE extraction_status='failed' AND deleted=0"
            ),
            "stale_sources": "SELECT COUNT(*) FROM sources WHERE deleted=1",
        }
        result: dict[str, Any] = {
            key: int(self.connection.execute(query).fetchone()[0])
            for key, query in scalar_queries.items()
        }
        result["formats"] = {
            str(row["document_type"]): int(row["count"])
            for row in self.connection.execute(
                "SELECT document_type, COUNT(DISTINCT source_id) AS count FROM sources "
                "WHERE deleted=0 AND extraction_status='success' GROUP BY document_type"
            )
        }
        result["roles"] = self.role_stats()
        result["schema_version"] = self.get_metadata("schema_version")
        result["embedding_model"] = self.get_metadata("embedding_model")
        result["embedding_dimension"] = self.connection.execute(
            "SELECT dimension FROM embedding_cache LIMIT 1"
        ).fetchone()
        dimension = result["embedding_dimension"]
        result["embedding_dimension"] = int(dimension["dimension"]) if dimension else None
        result["last_indexed_at"] = self.get_metadata("last_indexed_at")
        result["fts_available"] = self.fts_available()
        result["warnings"] = [
            warning
            for row in self.connection.execute(
                "SELECT warnings_json FROM sources WHERE deleted=0 AND warnings_json != '[]'"
            )
            for warning in json.loads(row["warnings_json"])
        ]
        result["failed_source_details"] = [
            {
                "source_id": str(row["source_id"]),
                "relative_path": json.loads(row["metadata_json"]).get("relative_path"),
                "error_type": json.loads(row["metadata_json"]).get("error_type"),
                "message": (json.loads(row["warnings_json"]) or [""])[0],
                "failed_at": str(row["indexed_at"]),
                "retry_eligible": bool(
                    json.loads(row["metadata_json"]).get("retry_eligible", False)
                ),
            }
            for row in self.connection.execute(
                "SELECT source_id, metadata_json, warnings_json, indexed_at FROM sources "
                "WHERE extraction_status='failed' AND deleted=0"
            )
        ]
        return result

    def role_stats(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for role in ChunkRole:
            aggregate = self.connection.execute(
                """
                SELECT COUNT(*) AS count, AVG(approximate_token_count) AS average,
                       MIN(approximate_token_count) AS minimum,
                       MAX(approximate_token_count) AS maximum,
                       SUM(CASE WHEN embedding_key IS NOT NULL THEN 1 ELSE 0 END) AS embedded
                FROM chunks c JOIN sources s USING(source_id)
                WHERE s.deleted=0 AND s.extraction_status='success' AND c.role=?
                """,
                (role.value,),
            ).fetchone()
            count = int(aggregate["count"])
            median = 0.0
            if count:
                offsets = sorted({(count - 1) // 2, count // 2})
                values = [
                    int(
                        self.connection.execute(
                            """
                            SELECT approximate_token_count FROM chunks c
                            JOIN sources s USING(source_id)
                            WHERE s.deleted=0 AND s.extraction_status='success' AND c.role=?
                            ORDER BY c.approximate_token_count LIMIT 1 OFFSET ?
                            """,
                            (role.value, offset),
                        ).fetchone()[0]
                    )
                    for offset in offsets
                ]
                median = sum(values) / len(values)
            result[role.value] = {
                "count": count,
                "average_tokens": round(float(aggregate["average"] or 0), 2),
                "median_tokens": median,
                "minimum_tokens": int(aggregate["minimum"] or 0),
                "maximum_tokens": int(aggregate["maximum"] or 0),
                "embedded_chunks": int(aggregate["embedded"] or 0),
                "non_embedded_chunks": count - int(aggregate["embedded"] or 0),
            }
        return result

    def search_lexical(
        self,
        query: str,
        *,
        limit: int,
        source_id: str | None = None,
        roles: list[ChunkRole] | None = None,
    ) -> list[sqlite3.Row]:
        terms = [term.replace('"', '""') for term in query.split() if term.strip()]
        if not terms:
            return []
        match_query = " OR ".join(f'"{term}"' for term in terms)
        parameters: list[Any] = [match_query, source_id, source_id]
        selected_roles = set(roles or [])
        parameters.append(int(bool(roles)))
        parameters.extend(int(role in selected_roles) for role in ChunkRole)
        parameters.append(limit)
        try:
            return self.connection.execute(
                """
                SELECT c.*, bm25(chunks_fts, 0.0, 0.0, 4.0, 2.0, 1.0, 1.5, 1.0) AS score,
                       (SELECT MIN(p.relative_path) FROM source_paths p
                        WHERE p.source_id=c.source_id) AS active_path
                FROM chunks_fts JOIN chunks c USING(chunk_id)
                JOIN sources s USING(source_id)
                WHERE chunks_fts MATCH ? AND s.deleted=0 AND s.extraction_status='success'
                  AND (? IS NULL OR c.source_id = ?)
                  AND (
                    ?=0
                    OR (c.role='content' AND ?=1)
                    OR (c.role='reference' AND ?=1)
                    OR (c.role='metadata' AND ?=1)
                    OR (c.role='navigation' AND ?=1)
                  )
                ORDER BY score ASC, c.chunk_id ASC LIMIT ?
                """,
                parameters,
            ).fetchall()
        except sqlite3.OperationalError as error:
            raise ValueError(f"Invalid or unsupported FTS query: {error}") from error

    def chunks_by_ids(self, chunk_ids: list[str]) -> list[sqlite3.Row]:
        rows: list[sqlite3.Row] = []
        for chunk_id in chunk_ids:
            row = self.connection.execute(
                """
                SELECT c.*, (SELECT MIN(p.relative_path) FROM source_paths p
                             WHERE p.source_id=c.source_id) AS active_path
                FROM chunks c JOIN sources s USING(source_id)
                WHERE c.chunk_id = ? AND s.deleted=0 AND s.extraction_status='success'
                """,
                (chunk_id,),
            ).fetchone()
            if row is not None:
                rows.append(row)
        return rows

    def source_records(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT s.*, MIN(p.relative_path) AS relative_path,
                   GROUP_CONCAT(p.relative_path, char(10)) AS paths
            FROM sources s LEFT JOIN source_paths p USING(source_id)
            WHERE s.deleted=0 AND s.extraction_status='success'
            GROUP BY s.source_id ORDER BY s.title
            """
        ).fetchall()
