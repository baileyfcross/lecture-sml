"""Deterministic, bounded source discovery."""

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pptx", ".docx", ".pdf", ".md", ".txt"}
DEFAULT_IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "__pycache__",
    "artifacts",
    "evals",
}

REPOSITORY_GENERATED_PATHS = {
    ("data", "knowledge"),
    ("data", "processed"),
    ("data", "ingestion"),
    ("evals",),
    ("artifacts",),
    ("logs",),
    ("models",),
    ("checkpoints",),
    ("outputs",),
    ("generated",),
}


@dataclass(frozen=True)
class DiscoveredFile:
    path: Path
    relative_path: str
    supported: bool


@dataclass(frozen=True)
class DiscoveryResult:
    root: Path
    files: list[DiscoveredFile]

    @property
    def supported(self) -> list[DiscoveredFile]:
        return [file for file in self.files if file.supported]

    @property
    def unsupported(self) -> list[DiscoveredFile]:
        return [file for file in self.files if not file.supported]


def discover_sources(
    supplied_path: Path,
    *,
    recursive: bool = True,
    ignored_directories: set[str] | None = None,
) -> DiscoveryResult:
    """Discover only within the explicitly supplied file or directory."""

    path = supplied_path.expanduser().resolve()
    ignored = DEFAULT_IGNORED_DIRECTORIES | (ignored_directories or set())
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if any(part in ignored for part in path.parts) or _is_repository_generated_path(path):
            raise ValueError(f"Refusing generated or evaluation path: {path}")
        root = path.parent
        files = [path]
    elif path.is_dir():
        root = path
        iterator = path.rglob("*") if recursive else path.glob("*")
        files = []
        for candidate in iterator:
            if (
                not candidate.is_file()
                or any(part in ignored for part in candidate.parts)
                or _is_repository_generated_path(candidate)
            ):
                continue
            if candidate.name.startswith("~$") or candidate.name in {".DS_Store", "Thumbs.db"}:
                continue
            files.append(candidate)
    else:
        raise ValueError(f"Supplied path is not a file or directory: {path}")
    discovered = [
        DiscoveredFile(
            path=file,
            relative_path=file.relative_to(root).as_posix(),
            supported=file.suffix.lower() in SUPPORTED_EXTENSIONS,
        )
        for file in sorted(files, key=lambda item: item.as_posix().lower())
    ]
    return DiscoveryResult(root=root, files=discovered)


def _is_repository_generated_path(path: Path) -> bool:
    """Exclude local index/training/evaluation outputs when scanning this repository."""

    repository_root = Path(__file__).resolve().parents[3]
    try:
        parts = tuple(part.casefold() for part in path.resolve().relative_to(repository_root).parts)
    except ValueError:
        return False
    return any(parts[: len(protected)] == protected for protected in REPOSITORY_GENERATED_PATHS)
