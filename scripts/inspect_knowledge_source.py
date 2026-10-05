"""Inspect indexed source metadata and chunk previews without loading embeddings."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from lecture_slm.knowledge.config import load_knowledge_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/knowledge/default.yaml"))
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--source-title")
    selector.add_argument("--source-id")
    selector.add_argument("--relative-path")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--preview-characters", type=int, default=360)
    args = parser.parse_args()
    try:
        config = load_knowledge_config(args.config)
        database = config.data_dir / "knowledge.sqlite"
        if not database.is_file():
            raise FileNotFoundError(f"No knowledge index at {database}")
        connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            source = _find_source(
                connection,
                source_title=args.source_title,
                source_id=args.source_id,
                relative_path=args.relative_path,
            )
            if source is None:
                raise ValueError("No indexed source matched the supplied selector")
            chunks = connection.execute(
                """
                SELECT * FROM chunks
                WHERE source_id=?
                ORDER BY chunk_index
                """,
                (source["source_id"],),
            ).fetchall()
        finally:
            connection.close()
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"Knowledge source inspection failed: {error}", file=sys.stderr)
        return 1
    metadata = json.loads(source["metadata_json"])
    print(f"Title: {source['title']}")
    print(f"Source ID: {source['source_id']}")
    print(f"Source hash: {source['source_hash']}")
    print(f"Type: {source['document_type']}; version: {source['source_version']}")
    print(f"Indexed: {source['indexed_at']}; chunks: {len(chunks)}")
    print(f"Paths: {source['paths']}")
    print(f"Tags: {json.dumps(metadata.get('tags', []), ensure_ascii=False)}")
    print(f"Aliases: {json.dumps(metadata.get('aliases', []), ensure_ascii=False)}")
    print(f"Links: {json.dumps(metadata.get('outgoing_links', []), ensure_ascii=False)}")
    print(f"Warnings: {json.dumps(json.loads(source['warnings_json']), ensure_ascii=False)}")
    for chunk in chunks:
        section_path = " > ".join(json.loads(chunk["section_path_json"]))
        print(
            f"\nChunk {chunk['chunk_index']} ({chunk['chunk_id']}) "
            f"section={section_path!r} page={chunk['page_number']} "
            f"slide={chunk['slide_number']} tokens~{chunk['approximate_token_count']}"
        )
        print(f"Role: {str(chunk['role']).upper()}; embedded: {chunk['embedding_key'] is not None}")
        text = str(chunk["text"])
        if not args.full:
            text = _preview(text, max(80, args.preview_characters))
        print(text)
    return 0


def _find_source(
    connection: sqlite3.Connection,
    *,
    source_title: str | None,
    source_id: str | None,
    relative_path: str | None,
) -> sqlite3.Row | None:
    rows = connection.execute(
        """
        SELECT s.*, GROUP_CONCAT(p.relative_path, char(10)) AS paths
        FROM sources s LEFT JOIN source_paths p USING(source_id)
        WHERE s.deleted=0 AND s.extraction_status='success'
        GROUP BY s.source_id
        ORDER BY s.title
        """
    ).fetchall()
    if source_id:
        return next((row for row in rows if str(row["source_id"]) == source_id), None)
    if relative_path:
        return next(
            (row for row in rows if relative_path in str(row["paths"] or "").splitlines()),
            None,
        )
    normalized = (source_title or "").casefold().strip()
    exact = [row for row in rows if str(row["title"]).casefold().strip() == normalized]
    if len(exact) > 1:
        raise ValueError("Source title is ambiguous; select it by source ID or relative path")
    return exact[0] if exact else None


def _preview(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


if __name__ == "__main__":
    sys.exit(main())
