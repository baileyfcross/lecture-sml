"""Explicit, read-only filesystem discovery for local knowledge roots."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def validate_knowledge_paths(data_dir: Path, cache_dir: Path, vault_root: Path) -> None:
    """Reject generated state paths that would write inside the read-only vault."""

    root = vault_root.expanduser().resolve()
    for name, path in (("data_dir", data_dir), ("embedding cache", cache_dir)):
        resolved = path.expanduser().resolve()
        if resolved == root or resolved.is_relative_to(root):
            raise ValueError(f"Knowledge {name} must be outside the read-only vault")


def dry_run_report(
    vault_path: Path,
    *,
    extensions: set[str],
    ignored_directories: set[str],
    recursive: bool,
    data_dir: Path,
    embedding_model: str,
    embedding_cache_dir: Path,
) -> dict[str, Any]:
    """Describe work without extracting files, loading embeddings, or writing state."""

    root = vault_path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Vault path must be a directory: {root}")
    ignored = {directory.casefold() for directory in ignored_directories}
    allowed = {extension.casefold() for extension in extensions}
    files: list[Path] = []
    ignored_paths: list[str] = []
    ignored_files: list[str] = []
    iterator = os.walk(root, followlinks=False)
    for directory, child_directories, names in iterator:
        current = Path(directory)
        kept_directories: list[str] = []
        for child in child_directories:
            child_path = current / child
            relative = child_path.relative_to(root).as_posix()
            if child.casefold() in ignored or child_path.is_symlink():
                ignored_paths.append(relative)
            else:
                kept_directories.append(child)
        child_directories[:] = kept_directories if recursive else []
        if not recursive and current != root:
            continue
        for name in names:
            file_path = current / name
            relative = file_path.relative_to(root).as_posix()
            if file_path.is_symlink() or name in {".DS_Store", "Thumbs.db"}:
                ignored_files.append(relative)
            else:
                files.append(file_path)
    supported = [file for file in files if file.suffix.casefold() in allowed]
    unsupported = [file for file in files if file.suffix.casefold() not in allowed]
    hashes: dict[str, list[str]] = {}
    extension_counts: dict[str, int] = {}
    total_bytes = 0
    for file in supported:
        digest_builder = hashlib.sha256()
        with file.open("rb") as source_file:
            for block in iter(lambda: source_file.read(1024 * 1024), b""):
                digest_builder.update(block)
        digest = digest_builder.hexdigest()
        hashes.setdefault(digest, []).append(file.relative_to(root).as_posix())
        extension_counts[file.suffix.casefold()] = (
            extension_counts.get(file.suffix.casefold(), 0) + 1
        )
        total_bytes += file.stat().st_size
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    cache_root = embedding_cache_dir.expanduser()
    cache_populated = cache_root.is_dir() and any(
        entry.is_file() for entry in cache_root.rglob("*")
    )
    return {
        "vault_root": str(root),
        "files_discovered": len(files),
        "supported_files": len(supported),
        "unsupported_files": len(unsupported),
        "supported_by_extension": dict(sorted(extension_counts.items())),
        "unsupported_by_extension": _extension_counts(unsupported),
        "ignored_directories": sorted(ignored_paths),
        "ignored_files": sorted(ignored_files),
        "ignored_file_count": len(ignored_files),
        "duplicate_candidate_groups": duplicate_groups,
        "duplicate_candidate_count": sum(len(paths) - 1 for paths in duplicate_groups),
        "estimated_extraction_workload": {
            "supported_files": len(supported),
            "input_bytes": total_bytes,
            "approximate_megabytes": round(total_bytes / (1024 * 1024), 2),
            "estimate_note": "File count and byte volume only; no extraction is performed.",
        },
        "target_knowledge_data_directory": str(data_dir.expanduser().resolve()),
        "embedding_model": embedding_model,
        "embedding_model_cache_present": cache_populated,
        "embedding_cache_directory": str(cache_root.resolve()),
        "dry_run": True,
    }


def _extension_counts(files: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for file in files:
        extension = file.suffix.casefold() or "[no extension]"
        counts[extension] = counts.get(extension, 0) + 1
    return dict(sorted(counts.items()))


@dataclass(frozen=True)
class VaultFile:
    path: Path
    relative_path: str


def discover_vault_files(
    vault_path: Path,
    *,
    extensions: set[str],
    ignored_directories: set[str],
    recursive: bool = True,
) -> list[VaultFile]:
    """List eligible files below an explicitly supplied vault without writing to it."""

    root = vault_path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Vault path must be a directory: {root}")
    ignored = {name.casefold() for name in ignored_directories}
    extension_set = {extension.casefold() for extension in extensions}
    discovered: list[VaultFile] = []
    iterator = root.rglob("*") if recursive else root.glob("*")
    for candidate in iterator:
        if candidate.is_symlink():
            continue
        if any(part.casefold() in ignored for part in candidate.relative_to(root).parts[:-1]):
            continue
        if candidate.is_file() and candidate.suffix.casefold() in extension_set:
            discovered.append(
                VaultFile(
                    path=candidate,
                    relative_path=candidate.relative_to(root).as_posix(),
                )
            )
    return sorted(discovered, key=lambda item: item.relative_path.casefold())
