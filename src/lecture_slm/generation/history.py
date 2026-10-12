"""Read and manage persisted generation archives without invoking generation stages."""

import json
import logging
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel, ConfigDict

from lecture_slm.generation.models import (
    GenerationProfileName,
    GenerationResult,
    GenerationStatus,
)
from lecture_slm.schemas.dataset import TaskType

logger = logging.getLogger(__name__)

_ARCHIVE_ID = re.compile(r"^\d{8}T\d{6}Z-[A-Za-z0-9._-]+-[0-9a-f]{6}$")


class GenerationHistorySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    archive_id: str
    request_id: str
    instruction: str
    task: TaskType
    profile: GenerationProfileName
    status: GenerationStatus
    created_at: datetime
    has_output: bool
    workspace_id: str | None = None


class GenerationArchive:
    def __init__(
        self,
        summary: GenerationHistorySummary,
        result: GenerationResult,
        output: str | None,
        retrieved_count: int,
        assembled_count: int,
    ) -> None:
        self.summary = summary
        self.result = result
        self.output = output
        self.retrieved_count = retrieved_count
        self.assembled_count = assembled_count


class ArchiveNotFoundError(LookupError):
    """Raised when an archive ID does not identify a direct child archive."""


class InvalidArchiveError(ValueError):
    """Raised when a selected archive is incomplete or malformed."""


class GenerationHistoryStore:
    """Filesystem-backed access to generation run archives."""

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root

    def list_archives(self, limit: int = 100) -> list[GenerationHistorySummary]:
        if not self.artifact_root.is_dir():
            return []
        limit = max(1, min(500, limit))
        candidates: list[tuple[datetime, Path]] = []
        for path in self.artifact_root.iterdir():
            if path.is_symlink() or not path.is_dir():
                continue
            try:
                candidates.append((_archive_timestamp(path), path))
            except OSError as error:
                logger.warning("Skipping invalid generation archive %s: %s", path.name, error)
        candidates.sort(key=lambda item: (item[0], item[1].name), reverse=True)
        archives: list[GenerationHistorySummary] = []
        for _, path in candidates:
            try:
                archives.append(self._read_summary(path))
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                logger.warning("Skipping invalid generation archive %s: %s", path.name, error)
            if len(archives) >= limit:
                break
        return archives

    def get_summary(self, archive_id: str) -> GenerationHistorySummary:
        return self._read_summary(self._resolve_archive(archive_id))

    def load_archive(self, archive_id: str) -> GenerationArchive:
        directory = self._resolve_archive(archive_id)
        try:
            request_data = json.loads(
                self._archive_file(directory, "request.json").read_text(encoding="utf-8")
            )
            result = GenerationResult.model_validate_json(
                self._archive_file(directory, "result.json").read_text(encoding="utf-8")
            )
            if not isinstance(request_data, dict):
                raise ValueError("request.json must contain a JSON object")
            if request_data.get("request_id", result.request_id) != result.request_id:
                raise ValueError("request and result IDs do not match")
            summary = self._summary_from_data(directory, request_data, result.model_dump())
            output = self._read_approved_output(directory, result.status)
            metadata = request_data.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("archive metadata has an invalid shape")
            retrieval = metadata.get("knowledge_retrieval", {})
            matches = retrieval.get("matches", []) if isinstance(retrieval, dict) else []
            sources = request_data.get("source_material", [])
            if not isinstance(matches, list) or not isinstance(sources, list):
                raise ValueError("archive source metadata has an invalid shape")
            return GenerationArchive(
                summary,
                result.model_copy(update={"final_output": output}),
                output,
                len(matches),
                len(sources),
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
            KeyError,
        ) as error:
            logger.warning("Could not load generation archive %s: %s", archive_id, error)
            raise InvalidArchiveError(
                "The generation archive is incomplete or malformed"
            ) from error

    def load_approved_output(self, archive_id: str) -> tuple[str, str]:
        directory = self._resolve_archive(archive_id)
        try:
            request_data = self._read_top_level_fields(
                self._archive_file(directory, "request.json"),
                {"request_id"},
            )
            result_data = self._read_top_level_fields(
                self._archive_file(directory, "result.json"),
                {"request_id", "status"},
            )
            request_id = request_data["request_id"]
            if (
                result_data["status"] != GenerationStatus.COMPLETED.value
                or result_data["request_id"] != request_id
            ):
                raise ValueError("Only completed generation archives can be saved to Workspace")
            output = self._read_approved_output(directory, GenerationStatus.COMPLETED)
            if output is None:
                raise ValueError("The archive has no approved output")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("request ID is missing")
            return request_id, output
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError) as error:
            logger.warning(
                "Could not save generation archive %s to Workspace: %s",
                archive_id,
                error,
            )
            raise InvalidArchiveError(str(error)) from error

    def delete_archive(self, archive_id: str) -> None:
        directory = self._resolve_archive(archive_id)
        if directory.is_symlink() or not directory.is_dir():
            raise ArchiveNotFoundError(archive_id)
        shutil.rmtree(directory)

    def _read_summary(self, directory: Path) -> GenerationHistorySummary:
        request_data = self._read_top_level_fields(
            self._archive_file(directory, "request.json"),
            {"request_id", "instruction"},
            optional={"workspace_id"},
        )
        result_data = self._read_top_level_fields(
            self._archive_file(directory, "result.json"),
            {"request_id", "task", "profile", "status"},
        )
        if request_data["request_id"] != result_data["request_id"]:
            raise InvalidArchiveError("request and result IDs do not match")
        try:
            return self._summary_from_data(directory, request_data, result_data)
        except (TypeError, ValueError, KeyError) as error:
            raise InvalidArchiveError("archive metadata is invalid") from error

    def _summary_from_data(
        self,
        directory: Path,
        request_data: dict[str, Any],
        result_data: dict[str, Any],
    ) -> GenerationHistorySummary:
        request_id = request_data.get("request_id", result_data.get("request_id"))
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("archive request ID is missing")
        status = GenerationStatus(result_data["status"])
        output_path = directory / "output.md"
        has_output = (
            status is GenerationStatus.COMPLETED
            and output_path.is_file()
            and not output_path.is_symlink()
            and output_path.stat().st_size > 0
        )
        instruction = request_data.get("instruction")
        if not isinstance(instruction, str):
            raise ValueError("request instruction is missing")
        workspace_id = request_data.get("workspace_id")
        if workspace_id is not None and not isinstance(workspace_id, str):
            raise ValueError("archive Workspace ID is invalid")
        return GenerationHistorySummary(
            archive_id=directory.name,
            request_id=request_id,
            instruction=instruction,
            task=result_data["task"],
            profile=result_data["profile"],
            status=status,
            created_at=_archive_timestamp(directory),
            has_output=has_output,
            workspace_id=workspace_id,
        )

    def _resolve_archive(self, archive_id: str) -> Path:
        if not _ARCHIVE_ID.fullmatch(archive_id):
            raise ArchiveNotFoundError(archive_id)
        try:
            root = self.artifact_root.resolve(strict=True)
            candidate = self.artifact_root / archive_id
            if candidate.is_symlink():
                raise ArchiveNotFoundError(archive_id)
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise ArchiveNotFoundError(archive_id) from error
        if resolved.parent != root or not resolved.is_dir():
            raise ArchiveNotFoundError(archive_id)
        return resolved

    @staticmethod
    def _archive_file(directory: Path, filename: str) -> Path:
        path = directory / filename
        if path.is_symlink() or not path.is_file():
            raise InvalidArchiveError(f"archive is missing its {filename}")
        return path

    @staticmethod
    def _read_approved_output(
        directory: Path,
        status: GenerationStatus,
    ) -> str | None:
        if status is not GenerationStatus.COMPLETED:
            return None
        path = directory / "output.md"
        if path.is_symlink() or not path.is_file():
            return None
        output = path.read_text(encoding="utf-8")
        return output if output.strip() else None

    @staticmethod
    def _read_top_level_fields(
        path: Path,
        selected: set[str],
        *,
        optional: set[str] | None = None,
    ) -> dict[str, Any]:
        optional = optional or set()
        values: dict[str, Any] = {}
        with path.open("r", encoding="utf-8") as source:
            if _read_non_whitespace(source) != "{":
                raise ValueError(f"{path.name} must contain a JSON object")
            while True:
                character = _read_non_whitespace(source)
                if character == "}":
                    break
                if character != '"':
                    raise ValueError(f"{path.name} contains invalid JSON")
                key_token = _read_json_string(source)
                key_value = json.loads(key_token)
                if not isinstance(key_value, str):
                    raise ValueError(f"{path.name} contains invalid JSON")
                key = key_value
                required_was_satisfied = selected.issubset(values)
                if required_was_satisfied and key not in optional:
                    break
                if _read_non_whitespace(source) != ":":
                    raise ValueError(f"{path.name} contains invalid JSON")
                raw_value = _read_json_value(source, key in selected or key in optional)
                if key in selected or key in optional:
                    values[key] = json.loads(raw_value)
                if selected.issubset(values) and optional.issubset(values):
                    break
                delimiter = _read_non_whitespace(source)
                if delimiter == "}":
                    break
                if delimiter != ",":
                    raise ValueError(f"{path.name} contains invalid JSON")
        missing = selected.difference(values)
        if missing:
            raise ValueError(f"{path.name} is missing required metadata")
        return values


def _archive_timestamp(directory: Path) -> datetime:
    match = re.match(r"^(\d{8}T\d{6}Z)-", directory.name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.fromtimestamp(directory.stat().st_mtime, tz=UTC)


def _read_non_whitespace(source: TextIO) -> str:
    character = source.read(1)
    while character and character.isspace():
        character = source.read(1)
    return character


def _read_json_string(source: TextIO) -> str:
    characters = ['"']
    escaped = False
    while True:
        character = source.read(1)
        if not character:
            raise ValueError("unexpected end of JSON string")
        characters.append(character)
        if character == '"' and not escaped:
            return "".join(characters)
        if character == "\\" and not escaped:
            escaped = True
        else:
            escaped = False


def _read_json_value(source: TextIO, capture: bool) -> str:
    first = _read_non_whitespace(source)
    if not first:
        raise ValueError("unexpected end of JSON value")
    characters = [first] if capture else []
    if first == '"':
        remainder = _read_json_string(source)
        if capture:
            characters.extend(remainder[1:])
        return "".join(characters)
    if first in "[{":
        stack = ["]" if first == "[" else "}"]
        in_string = False
        escaped = False
        while stack:
            character = source.read(1)
            if not character:
                raise ValueError("unexpected end of JSON container")
            if capture:
                characters.append(character)
            if in_string:
                if character == '"' and not escaped:
                    in_string = False
                if character == "\\" and not escaped:
                    escaped = True
                else:
                    escaped = False
            elif character == '"':
                in_string = True
            elif character == "[":
                stack.append("]")
            elif character == "{":
                stack.append("}")
            elif character in "]}":
                if not stack or character != stack.pop():
                    raise ValueError("mismatched JSON container")
        return "".join(characters)
    while True:
        character = source.read(1)
        if not character:
            break
        if character in ",}]":
            source.seek(source.tell() - 1)
            break
        if capture:
            characters.append(character)
    return "".join(characters).strip()
