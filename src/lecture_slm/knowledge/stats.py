"""Read-only knowledge-store status reporting."""

import json
import sqlite3
import statistics
from pathlib import Path
from typing import Any

from lecture_slm.knowledge.roles import ChunkRole
from lecture_slm.knowledge.storage import SCHEMA_VERSION


def knowledge_stats(data_dir: Path) -> dict[str, Any]:
    """Report local index state without opening models or modifying the database."""

    database = data_dir / "knowledge.sqlite"
    if not database.exists():
        return {
            "indexed_sources": 0,
            "indexed_chunks": 0,
            "last_indexed_at": None,
            "embedding_model": None,
            "embedding_dimension": None,
            "fts_available": False,
            "schema_version": None,
            "warnings": [],
            "message": f"No knowledge index found at {database}",
        }
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        schema_version = _metadata(connection, "schema_version")
        if schema_version != str(SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported knowledge schema version {schema_version!r}; "
                "run index_knowledge.py --rebuild."
            )
        indexed_sources = connection.execute(
            """
            SELECT COUNT(DISTINCT p.source_id)
            FROM source_paths p JOIN sources s USING(source_id)
            WHERE s.deleted=0 AND s.extraction_status='success'
            """
        ).fetchone()[0]
        indexed_chunks = connection.execute(
            """
            SELECT COUNT(*) FROM chunks c JOIN sources s USING(source_id)
            WHERE s.deleted=0 AND s.extraction_status='success'
            """
        ).fetchone()[0]
        size_stats = connection.execute(
            """
            SELECT COUNT(*) AS count, AVG(c.approximate_token_count) AS average,
                   MIN(c.approximate_token_count) AS minimum,
                   MAX(c.approximate_token_count) AS maximum
            FROM chunks c JOIN sources s USING(source_id)
            WHERE s.deleted=0 AND s.extraction_status='success'
            """
        ).fetchone()
        chunk_count = int(size_stats["count"])
        median: float = 0.0
        if chunk_count:
            median_rows = [
                connection.execute(
                    """
                    SELECT approximate_token_count FROM chunks c
                    JOIN sources s USING(source_id)
                    WHERE s.deleted=0 AND s.extraction_status='success'
                    ORDER BY c.approximate_token_count
                    LIMIT 1 OFFSET ?
                    """,
                    (offset,),
                ).fetchone()
                for offset in {(chunk_count - 1) // 2, chunk_count // 2}
            ]
            median = sum(int(row[0]) for row in median_rows) / len(median_rows)
        formats = {
            str(row["document_type"]): int(row["count"])
            for row in connection.execute(
                """
                SELECT document_type, COUNT(DISTINCT source_id) AS count
                FROM sources WHERE deleted=0 AND extraction_status='success'
                GROUP BY document_type
                """
            )
        }
        role_statistics: dict[str, dict[str, Any]] = {}
        for role in ChunkRole:
            rows = connection.execute(
                """
                SELECT c.approximate_token_count, c.embedding_key
                FROM chunks c JOIN sources s USING(source_id)
                WHERE s.deleted=0 AND s.extraction_status='success' AND c.role=?
                ORDER BY c.approximate_token_count
                """,
                (role.value,),
            ).fetchall()
            token_counts = [int(row["approximate_token_count"]) for row in rows]
            embedded = sum(row["embedding_key"] is not None for row in rows)
            role_statistics[role.value] = {
                "count": len(rows),
                "average_tokens": round(statistics.mean(token_counts), 2) if rows else 0,
                "median_tokens": statistics.median(token_counts) if rows else 0,
                "minimum_tokens": min(token_counts, default=0),
                "maximum_tokens": max(token_counts, default=0),
                "embedded_chunks": embedded,
                "non_embedded_chunks": len(rows) - embedded,
            }
        try:
            connection.execute("SELECT count(*) FROM chunks_fts").fetchone()
            fts_available = True
        except sqlite3.OperationalError:
            fts_available = False
        failed_details = [
            _failed_source(row)
            for row in connection.execute(
                """
                SELECT source_id, metadata_json, warnings_json, indexed_at
                FROM sources WHERE extraction_status='failed' AND deleted=0
                """
            )
        ]
        warnings = [
            warning
            for row in connection.execute(
                "SELECT warnings_json FROM sources WHERE deleted=0 AND warnings_json != '[]'"
            )
            for warning in json.loads(row["warnings_json"])
        ]
        dimension_row = connection.execute(
            "SELECT dimension FROM embedding_cache LIMIT 1"
        ).fetchone()
        return {
            "indexed_sources": int(indexed_sources),
            "indexed_chunks": int(indexed_chunks),
            "chunk_size_tokens": {
                "average": round(float(size_stats["average"] or 0), 2),
                "minimum": int(size_stats["minimum"] or 0),
                "median": median,
                "maximum": int(size_stats["maximum"] or 0),
            },
            "failed_sources": len(failed_details),
            "failed_source_details": failed_details,
            "stale_sources": int(
                connection.execute("SELECT COUNT(*) FROM sources WHERE deleted=1").fetchone()[0]
            ),
            "formats": formats,
            "total_chunks": int(indexed_chunks),
            "role_statistics": role_statistics,
            "schema_version": schema_version,
            "embedding_model": _metadata(connection, "embedding_model"),
            "embedding_dimension": int(dimension_row[0]) if dimension_row else None,
            "last_indexed_at": _metadata(connection, "last_indexed_at"),
            "fts_available": fts_available,
            "warnings": warnings,
        }
    finally:
        connection.close()


def _metadata(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM index_metadata WHERE key=?", (key,)).fetchone()
    return str(row["value"]) if row else None


def _failed_source(row: sqlite3.Row) -> dict[str, Any]:
    metadata = json.loads(row["metadata_json"])
    messages = json.loads(row["warnings_json"])
    return {
        "source_id": str(row["source_id"]),
        "relative_path": metadata.get("relative_path"),
        "error_type": metadata.get("error_type"),
        "message": messages[0] if messages else "",
        "failed_at": str(row["indexed_at"]),
        "retry_eligible": bool(metadata.get("retry_eligible", False)),
    }
