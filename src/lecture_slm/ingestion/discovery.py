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
        if any(part in ignored for part in path.parts):
            raise ValueError(f"Refusing generated or evaluation path: {path}")
        root = path.parent
        files = [path]
    elif path.is_dir():
        root = path
        iterator = path.rglob("*") if recursive else path.glob("*")
        files = []
        for candidate in iterator:
            if not candidate.is_file() or any(part in ignored for part in candidate.parts):
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
