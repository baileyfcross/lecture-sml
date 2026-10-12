import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lecture_slm.api.app import create_app
from lecture_slm.generation.history import (
    ArchiveNotFoundError,
    GenerationHistoryStore,
    InvalidArchiveError,
)
from lecture_slm.generation.models import (
    GenerationProfileName,
    GenerationResult,
    GenerationStatus,
    StageRecord,
    StageTiming,
)
from lecture_slm.schemas.dataset import TaskType
from lecture_slm.workspaces.service import WorkspaceService
from test_api import FakeService


def write_archive(
    root: Path,
    *,
    timestamp: str,
    request_id: str,
    status: GenerationStatus = GenerationStatus.COMPLETED,
    output: str | None = "Approved archive output",
) -> Path:
    directory = root / f"{timestamp}-{request_id}-a1b2c3"
    directory.mkdir(parents=True)
    request_data = {
        "request_id": request_id,
        "task": "lecture",
        "profile": "deep",
        "instruction": "Create lecture 4 about generic methods.",
        "workspace_id": None,
        "source_material": [{"source_id": "source-1"}],
        "metadata": {"knowledge_retrieval": {"matches": [{"chunk_id": "chunk-1"}]}},
    }
    (directory / "request.json").write_text(json.dumps(request_data), encoding="utf-8")
    result = GenerationResult(
        request_id=request_id,
        task=TaskType.LECTURE,
        profile=GenerationProfileName.DEEP,
        model="local-test-model",
        status=status,
        writer_result=StageRecord(
            status=GenerationStatus.COMPLETED,
            timing=StageTiming(duration_seconds=2),
            raw_response="Unapproved writer candidate",
        ),
        final_output=output if status is GenerationStatus.COMPLETED else None,
        timing=StageTiming(duration_seconds=3),
        errors=[] if status is GenerationStatus.COMPLETED else ["Generation failed"],
        planner_prompt_version=None,
        writer_prompt_version="writer-v1",
    )
    (directory / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    if status is GenerationStatus.COMPLETED and output is not None:
        (directory / "output.md").write_text(output, encoding="utf-8")
    if status is GenerationStatus.FAILED:
        (directory / "writer_output.md").write_text(
            "Unapproved writer candidate",
            encoding="utf-8",
        )
    return directory


def test_history_summaries_are_newest_first_bounded_and_skip_output_payloads(
    tmp_path: Path,
) -> None:
    root = tmp_path / "generations"
    write_archive(
        root,
        timestamp="20261010T010203Z",
        request_id="request-1",
        output="SENSITIVE OUTPUT BODY",
    )
    write_archive(
        root,
        timestamp="20261011T010203Z",
        request_id="request-2",
        output="OTHER OUTPUT BODY",
    )
    write_archive(
        root,
        timestamp="20261012T010203Z",
        request_id="request-3",
        output="LATEST OUTPUT BODY",
    )

    summaries = GenerationHistoryStore(root).list_archives(limit=2)

    assert [item.request_id for item in summaries] == ["request-3", "request-2"]
    assert summaries[0].instruction == "Create lecture 4 about generic methods."
    assert summaries[0].task is TaskType.LECTURE
    assert summaries[0].profile is GenerationProfileName.DEEP
    assert summaries[0].status is GenerationStatus.COMPLETED
    assert summaries[0].has_output is True
    assert summaries[0].created_at == datetime(2026, 10, 12, 1, 2, 3, tzinfo=UTC)
    assert "LATEST OUTPUT BODY" not in summaries[0].model_dump_json()


def test_history_api_defaults_to_the_100_newest_archives(tmp_path: Path) -> None:
    root = tmp_path / "generations"
    for index in range(105):
        timestamp = f"20261012T{index // 3600:02d}{index // 60 % 60:02d}{index % 60:02d}Z"
        directory = root / f"{timestamp}-request-{index}-a1b2c3"
        directory.mkdir(parents=True)
        (directory / "request.json").write_text(
            json.dumps(
                {
                    "request_id": f"request-{index}",
                    "instruction": f"Instruction {index}",
                    "workspace_id": None,
                }
            ),
            encoding="utf-8",
        )
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "request_id": f"request-{index}",
                    "task": "lecture",
                    "profile": "deep",
                    "status": "completed",
                }
            ),
            encoding="utf-8",
        )
    service = FakeService()
    service.artifact_root = root
    response = TestClient(create_app(service)).get("/api/history")

    assert response.status_code == 200
    summaries = response.json()
    assert len(summaries) == 100
    assert summaries[0]["request_id"] == "request-104"
    assert summaries[-1]["request_id"] == "request-5"
    assert service.calls == []


def test_history_supports_existing_archive_shape_and_hides_failed_candidates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "generations"
    failed = write_archive(
        root,
        timestamp="20261012T010203Z",
        request_id="request-failed",
        status=GenerationStatus.FAILED,
        output=None,
    )
    request_path = failed / "request.json"
    older_request = json.loads(request_path.read_text(encoding="utf-8"))
    older_request.pop("workspace_id")
    request_path.write_text(json.dumps(older_request), encoding="utf-8")
    store = GenerationHistoryStore(root)

    listed = store.list_archives()
    opened = store.load_archive(failed.name)

    assert len(listed) == 1
    assert listed[0].has_output is False
    assert listed[0].workspace_id is None
    assert opened.output is None
    assert opened.result.final_output is None
    assert (failed / "writer_output.md").is_file()


def test_history_skips_malformed_directories_and_rejects_malformed_detail(
    tmp_path: Path,
) -> None:
    root = tmp_path / "generations"
    valid = write_archive(root, timestamp="20261012T010203Z", request_id="request-1")
    malformed = root / "20261012T020203Z-malformed-a1b2c3"
    malformed.mkdir()
    (malformed / "request.json").write_text("{}", encoding="utf-8")
    (malformed / "result.json").write_text("{", encoding="utf-8")
    store = GenerationHistoryStore(root)

    assert [item.archive_id for item in store.list_archives()] == [valid.name]
    with pytest.raises(InvalidArchiveError):
        store.load_archive(malformed.name)


def test_archive_resolution_rejects_traversal_without_touching_sibling_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "generations"
    root.mkdir()
    outside = tmp_path / "sentinel.txt"
    outside.write_text("keep", encoding="utf-8")
    store = GenerationHistoryStore(root)

    for archive_id in ("..", "../sentinel.txt", "..%2fsentinel.txt", str(outside)):
        with pytest.raises(ArchiveNotFoundError):
            store.delete_archive(archive_id)
    assert outside.read_text(encoding="utf-8") == "keep"


def test_archive_delete_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "generations"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    link = root / "20261012T010203Z-link-a1b2c3"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(ArchiveNotFoundError):
        GenerationHistoryStore(root).delete_archive(link.name)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_history_api_is_registry_independent_and_workspace_copies_survive_archive_delete(
    tmp_path: Path,
) -> None:
    service = FakeService()
    service.artifact_root = tmp_path / "generations"
    service.workspaces = WorkspaceService(database_path=tmp_path / "workspaces.sqlite")
    workspace = service.workspaces.store.create_workspace("History test")
    archive = write_archive(
        service.artifact_root,
        timestamp="20261012T010203Z",
        request_id="request-saved",
    )
    failed = write_archive(
        service.artifact_root,
        timestamp="20261012T020203Z",
        request_id="request-failed",
        status=GenerationStatus.FAILED,
        output=None,
    )
    client = TestClient(create_app(service))

    history = client.get("/api/history?limit=100")
    assert history.status_code == 200
    entries = history.json()
    assert [item["archive_id"] for item in entries] == [failed.name, archive.name]
    assert entries[1]["instruction"] == "Create lecture 4 about generic methods."
    assert entries[1]["has_output"] is True
    assert "output" not in entries[1]
    assert "diagnostics" not in entries[1]

    opened = client.get(f"/api/history/{archive.name}")
    assert opened.status_code == 200
    assert opened.json()["output"] == "Approved archive output"
    assert opened.json()["sources"] == {"retrieved": 1, "assembled": 1}
    assert opened.json()["saved_run"].endswith(archive.name)
    failed_open = client.get(f"/api/history/{failed.name}")
    assert failed_open.status_code == 200
    assert failed_open.json()["status"] == "failed"
    assert failed_open.json()["output"] is None

    saved = client.post(
        f"/api/history/{archive.name}/workspace",
        json={
            "workspace_id": workspace.id,
            "title": "Lecture 4",
            "folder_id": None,
        },
    )
    assert saved.status_code == 201
    saved_item = saved.json()
    assert saved_item["role"] == "history"
    assert saved_item["source_request_id"] == "request-saved"
    assert saved_item["content"] == "Approved archive output"

    rejected = client.post(
        f"/api/history/{failed.name}/workspace",
        json={"workspace_id": workspace.id, "title": "Failed run"},
    )
    assert rejected.status_code == 409
    assert service.calls == []

    deleted = client.delete(f"/api/history/{archive.name}")
    assert deleted.status_code == 204
    assert not archive.exists()
    assert client.delete(f"/api/history/{archive.name}").status_code == 404
    workspace_after_delete = client.get(f"/api/workspaces/{workspace.id}")
    assert workspace_after_delete.status_code == 200
    assert workspace_after_delete.json()["items"][0]["id"] == saved_item["id"]
    assert workspace_after_delete.json()["items"][0]["role"] == "history"
    assert client.delete("/api/history/..%2F..%2Fsentinel").status_code == 404
    assert service.calls == []


def test_history_api_rejects_invalid_limit_and_unknown_archives(tmp_path: Path) -> None:
    service = FakeService()
    service.artifact_root = tmp_path / "generations"
    client = TestClient(create_app(service))

    assert client.get("/api/history?limit=0").status_code == 422
    assert client.get("/api/history?limit=501").status_code == 422
    assert client.get("/api/history/not-an-archive").status_code == 404
    assert client.delete("/api/history/not-an-archive").status_code == 404


def test_history_detail_reports_malformed_archive_without_exposing_filesystem_details(
    tmp_path: Path,
) -> None:
    service = FakeService()
    service.artifact_root = tmp_path / "generations"
    malformed = service.artifact_root / "20261012T010203Z-malformed-a1b2c3"
    malformed.mkdir(parents=True)
    (malformed / "request.json").write_text("{", encoding="utf-8")
    (malformed / "result.json").write_text("{}", encoding="utf-8")

    response = TestClient(create_app(service)).get(f"/api/history/{malformed.name}")

    assert response.status_code == 409
    assert response.json()["detail"] == "The generation archive is incomplete or malformed"
    assert str(tmp_path) not in response.text
    assert service.calls == []
