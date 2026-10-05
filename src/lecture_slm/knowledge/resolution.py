"""Exact-first source and section resolution for knowledge queries."""

import json
import re
from difflib import SequenceMatcher
from pathlib import PurePosixPath
from typing import Any

from lecture_slm.knowledge.storage import KnowledgeStore

SECTION_NUMBER = re.compile(r"\b(?:section\s+)?(\d+(?:\.\d+)+)\b", re.IGNORECASE)


class SourceResolver:
    """Resolve names and sections explicitly before ranking passages."""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def resolve_source(
        self, query: str, requested_title: str | None = None
    ) -> tuple[dict[str, Any] | None, list[str]]:
        records = self.store.source_records()
        warnings: list[str] = []
        query_norm = normalize_title(requested_title or query)
        exact: list[tuple[int, Any, str]] = []
        for row in records:
            aliases = _aliases(str(row["metadata_json"]))
            paths = str(row["paths"] or "").splitlines()
            title = str(row["title"])
            filenames = [*paths, *(PurePosixPath(path).stem for path in paths)]
            if query_norm == normalize_title(title):
                lookup_text = requested_title or query
                reason = "exact title" if lookup_text == title else "normalized title"
                exact.append((len(query_norm), row, reason))
            elif query_norm and query_norm in [normalize_title(alias) for alias in aliases]:
                exact.append((len(query_norm), row, "alias"))
            elif query_norm and query_norm in [normalize_title(name) for name in filenames]:
                exact.append((len(query_norm), row, "exact filename"))
            else:
                named = [
                    (len(normalized), category)
                    for category, names in (
                        ("title named in query", [title]),
                        ("filename named in query", filenames),
                        ("alias named in query", aliases),
                    )
                    for name in names
                    if (normalized := normalize_title(name))
                    and len(normalized) >= 4
                    and re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", normalize_title(query))
                ]
                if named:
                    best_length = max(length for length, _ in named)
                    reasons = sorted({reason for length, reason in named if length == best_length})
                    exact.append((best_length, row, reasons[0]))
        if exact:
            longest = max(length for length, _, _ in exact)
            best_matches = [item for item in exact if item[0] == longest]
            unique = {str(item[1]["source_id"]): item for item in best_matches}
            if len(unique) == 1:
                _, row, reason = next(iter(unique.values()))
                return _source_payload(row, reason), warnings
            warnings.append(
                "Source resolution is ambiguous: "
                + ", ".join(sorted(str(row["title"]) for _, row, _ in best_matches))
            )
            return None, warnings

        if requested_title:
            scored = [
                (
                    max(
                        [
                            SequenceMatcher(
                                None, query_norm, normalize_title(str(row["title"]))
                            ).ratio(),
                            *[
                                SequenceMatcher(None, query_norm, normalize_title(alias)).ratio()
                                for alias in _aliases(str(row["metadata_json"]))
                            ],
                        ]
                    ),
                    row,
                )
                for row in records
            ]
            scored.sort(key=lambda item: (-item[0], str(item[1]["title"])))
            if scored and scored[0][0] >= 0.86:
                tied = [item for item in scored if abs(item[0] - scored[0][0]) < 0.02]
                if len(tied) == 1:
                    return _source_payload(scored[0][1], "fuzzy title match"), warnings
                warnings.append(
                    "Fuzzy source resolution is ambiguous: "
                    + ", ".join(str(row["title"]) for _, row in tied)
                )
            else:
                warnings.append(f"No indexed source matched requested title '{requested_title}'")
        return None, warnings

    def resolve_section(
        self, source_id: str | None, requested_section: str | None, query: str
    ) -> tuple[str | None, set[str], str | None]:
        section_query = requested_section
        if section_query is None:
            match = SECTION_NUMBER.search(query)
            section_query = match.group(1) if match else None
        if not section_query:
            return None, set(), None
        normalized = normalize_title(section_query)
        rows = self.store.connection.execute(
            """
            SELECT c.chunk_id, c.section_title, c.section_path_json, c.page_number
            FROM chunks c JOIN sources s USING(source_id)
            WHERE s.deleted=0 AND s.extraction_status='success'
              AND (? IS NULL OR c.source_id = ?)
            """,
            (source_id, source_id),
        ).fetchall()
        exact_chunks: set[str] = set()
        matched_heading: str | None = None
        for row in rows:
            path = json.loads(row["section_path_json"])
            candidates = [str(name) for name in path]
            if row["section_title"]:
                candidates.append(str(row["section_title"]))
            for candidate in candidates:
                candidate_norm = normalize_title(candidate)
                if candidate_norm == normalized or candidate_norm.startswith(normalized):
                    exact_chunks.add(str(row["chunk_id"]))
                    matched_heading = candidate
        if exact_chunks:
            return matched_heading or section_query, exact_chunks, None
        if re.fullmatch(r"\d+(?:\.\d+)+", section_query):
            return None, set(), f"No extracted heading matched section {section_query}"
        return None, set(), f"No extracted heading matched section '{section_query}'"

    @staticmethod
    def is_high_confidence_lookup(query: str, source: dict[str, Any] | None) -> bool:
        """Recognize exact document lookups without treating topical mentions as constraints."""

        if source is None:
            return False
        query_terms = normalize_title(query).split()
        if not query_terms:
            return False
        source_names = [
            str(source.get("title", "")),
            *(str(path) for path in source.get("paths", [])),
        ]
        lookup_terms = {"lecture", "chapter", "week", "unit", "lesson", "notes", "document", "pdf"}
        for name in source_names:
            normalized_name = normalize_title(PurePosixPath(name).stem).split()
            if not normalized_name:
                continue
            if query_terms == normalized_name:
                return True
            remaining = list(query_terms)
            for term in normalized_name:
                if term in remaining:
                    remaining.remove(term)
            if len(remaining) <= 2 and all(
                term in lookup_terms or term.isdigit() for term in remaining
            ):
                return True
        return False


def normalize_title(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _aliases(metadata_json: str) -> list[str]:
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError:
        return []
    value = metadata.get("aliases", []) if isinstance(metadata, dict) else []
    if isinstance(value, str):
        return [value]
    return [str(alias) for alias in value if isinstance(alias, (str, int, float))]


def _source_payload(row: Any, reason: str) -> dict[str, Any]:
    return {
        "source_id": str(row["source_id"]),
        "title": str(row["title"]),
        "relative_path": str(row["relative_path"] or ""),
        "paths": str(row["paths"] or "").splitlines(),
        "resolution_reason": reason,
    }
