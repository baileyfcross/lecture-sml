"""FastAPI application exposing the shared Lecture SLM generation service."""

import asyncio
import json
import logging
from collections import OrderedDict
from collections.abc import AsyncIterator
from pathlib import Path
from threading import Lock
from typing import NoReturn

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from lecture_slm.api.models import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ProfileSummary,
    StageProgress,
    WorkspaceFolderUpdate,
    WorkspaceHistorySave,
    WorkspaceItemUpdate,
)
from lecture_slm.generation.models import ProgressEvent
from lecture_slm.generation.service import GenerationService
from lecture_slm.inference.ollama_client import OllamaClient, OllamaError
from lecture_slm.schemas.dataset import TaskType
from lecture_slm.workspaces.models import (
    WorkspaceCreate,
    WorkspaceDetail,
    WorkspaceFolder,
    WorkspaceFolderInput,
    WorkspaceItem,
    WorkspaceItemInput,
    WorkspaceItemRole,
    WorkspaceSummary,
    WorkspaceUpdate,
)
from lecture_slm.workspaces.storage import (
    WorkspaceConflictError,
    WorkspaceNotFoundError,
    WorkspaceStore,
)

logger = logging.getLogger(__name__)


class RunRegistry:
    def __init__(self, capacity: int = 100) -> None:
        if capacity <= 0:
            raise ValueError("run registry capacity must be positive")
        self.capacity = capacity
        self._runs: OrderedDict[str, GenerateResponse] = OrderedDict()
        self._lock = Lock()

    def add(self, response: GenerateResponse) -> None:
        with self._lock:
            self._runs[response.request_id] = response
            self._runs.move_to_end(response.request_id)
            while len(self._runs) > self.capacity:
                self._runs.popitem(last=False)

    def get(self, request_id: str) -> GenerateResponse | None:
        with self._lock:
            return self._runs.get(request_id)


def create_app(
    service: GenerationService,
    *,
    max_retained_runs: int = 100,
    frontend_dist: Path | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Lecture SLM Local API",
        description="Local HTTP interface to the existing Lecture SLM generation pipeline.",
        version="0.1.0",
    )
    registry = RunRegistry(max_retained_runs)
    app.state.generation_service = service
    app.state.run_registry = registry
    static_root = frontend_dist or Path(__file__).parents[3] / "web" / "dist"
    index_file = static_root / "index.html"
    assets_dir = static_root / "assets"

    def workspace_store() -> WorkspaceStore:
        return service.workspaces.store

    def raise_workspace_http_error(error: Exception) -> NoReturn:
        if isinstance(error, WorkspaceNotFoundError):
            raise HTTPException(status_code=404, detail="Workspace record not found") from error
        if isinstance(error, WorkspaceConflictError):
            raise HTTPException(status_code=409, detail=str(error)) from error
        raise error

    if index_file.is_file() and assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/", include_in_schema=False, response_model=None)
    async def frontend() -> Response:
        if index_file.is_file():
            return FileResponse(index_file)
        return PlainTextResponse(
            "Lecture SLM web UI has not been built. Run `cd web && npm run build`."
        )

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        reachable = False
        model_available = False
        try:
            models = await asyncio.to_thread(
                OllamaClient(service.model_config.inference.host, timeout=2.0).available_models
            )
            reachable = True
            model_available = service.model_config.ollama_name in models
        except OllamaError:
            logger.info("Ollama health check did not succeed", exc_info=True)

        index_available: bool | None = None
        index_error = None
        try:
            index_available = await asyncio.to_thread(service.knowledge_index_available)
        except (OSError, ValueError) as error:
            index_available = False
            index_error = str(error)
        return HealthResponse(
            generation_ready=model_available,
            ollama_reachable=reachable,
            model_available=model_available,
            model=service.model_config.ollama_name,
            knowledge_index_available=index_available,
            knowledge_index_error=index_error,
        )

    @app.get("/api/tasks")
    async def tasks() -> dict[str, list[str]]:
        return {"tasks": [task.value for task in TaskType]}

    @app.get("/api/profiles")
    async def profiles() -> dict[str, list[ProfileSummary]]:
        return {"profiles": ProfileSummary.from_profiles(service.profiles)}

    @app.get("/api/workspaces")
    async def list_workspaces() -> list[WorkspaceSummary]:
        return await asyncio.to_thread(workspace_store().list_workspaces)

    @app.post("/api/workspaces", status_code=201)
    async def create_workspace(payload: WorkspaceCreate) -> WorkspaceSummary:
        return await asyncio.to_thread(
            workspace_store().create_workspace,
            payload.name,
            payload.description,
        )

    @app.get("/api/workspaces/{workspace_id}")
    async def get_workspace(workspace_id: str) -> WorkspaceDetail:
        try:
            return await asyncio.to_thread(workspace_store().get_workspace, workspace_id)
        except WorkspaceNotFoundError as error:
            raise_workspace_http_error(error)

    @app.patch("/api/workspaces/{workspace_id}")
    async def update_workspace(
        workspace_id: str,
        payload: WorkspaceUpdate,
    ) -> WorkspaceSummary:
        try:
            return await asyncio.to_thread(
                workspace_store().update_workspace,
                workspace_id,
                **payload.model_dump(exclude_unset=True),
                description_is_set="description" in payload.model_fields_set,
            )
        except (WorkspaceNotFoundError, WorkspaceConflictError) as error:
            raise_workspace_http_error(error)

    @app.delete("/api/workspaces/{workspace_id}", status_code=204)
    async def delete_workspace(workspace_id: str) -> Response:
        try:
            await asyncio.to_thread(workspace_store().delete_workspace, workspace_id)
        except WorkspaceNotFoundError as error:
            raise_workspace_http_error(error)
        return Response(status_code=204)

    @app.post("/api/workspaces/{workspace_id}/folders", status_code=201)
    async def create_workspace_folder(
        workspace_id: str,
        payload: WorkspaceFolderInput,
    ) -> WorkspaceFolder:
        try:
            return await asyncio.to_thread(
                workspace_store().create_folder,
                workspace_id,
                payload.name,
                payload.parent_id,
            )
        except (WorkspaceNotFoundError, WorkspaceConflictError) as error:
            raise_workspace_http_error(error)

    @app.patch("/api/workspaces/{workspace_id}/folders/{folder_id}")
    async def update_workspace_folder(
        workspace_id: str,
        folder_id: str,
        payload: WorkspaceFolderUpdate,
    ) -> WorkspaceFolder:
        try:
            store = workspace_store()
            detail = await asyncio.to_thread(store.get_workspace, workspace_id)
            folder = next((item for item in detail.folders if item.id == folder_id), None)
            if folder is None:
                raise HTTPException(status_code=404, detail="Workspace record not found")
            return await asyncio.to_thread(
                store.update_folder,
                workspace_id,
                folder_id,
                name=payload.name or folder.name,
                parent_id=payload.parent_id,
                move="parent_id" in payload.model_fields_set,
            )
        except WorkspaceNotFoundError as error:
            raise_workspace_http_error(error)
        except WorkspaceConflictError as error:
            raise_workspace_http_error(error)

    @app.delete("/api/workspaces/{workspace_id}/folders/{folder_id}", status_code=204)
    async def delete_workspace_folder(workspace_id: str, folder_id: str) -> Response:
        try:
            await asyncio.to_thread(workspace_store().delete_folder, workspace_id, folder_id)
        except (WorkspaceNotFoundError, WorkspaceConflictError) as error:
            raise_workspace_http_error(error)
        return Response(status_code=204)

    @app.post("/api/workspaces/{workspace_id}/items", status_code=201)
    async def create_workspace_item(
        workspace_id: str,
        payload: WorkspaceItemInput,
    ) -> WorkspaceItem:
        try:
            return await asyncio.to_thread(
                workspace_store().create_item,
                workspace_id,
                payload,
            )
        except (WorkspaceNotFoundError, WorkspaceConflictError) as error:
            raise_workspace_http_error(error)

    @app.patch("/api/workspaces/{workspace_id}/items/{item_id}")
    async def update_workspace_item(
        workspace_id: str,
        item_id: str,
        payload: WorkspaceItemUpdate,
    ) -> WorkspaceItem:
        try:
            return await asyncio.to_thread(
                workspace_store().update_item,
                workspace_id,
                item_id,
                payload.model_dump(exclude_unset=True),
            )
        except (WorkspaceNotFoundError, WorkspaceConflictError) as error:
            raise_workspace_http_error(error)

    @app.delete("/api/workspaces/{workspace_id}/items/{item_id}", status_code=204)
    async def delete_workspace_item(workspace_id: str, item_id: str) -> Response:
        try:
            await asyncio.to_thread(
                workspace_store().delete_item,
                workspace_id,
                item_id,
            )
        except WorkspaceNotFoundError as error:
            raise_workspace_http_error(error)
        return Response(status_code=204)

    @app.post("/api/workspaces/{workspace_id}/history", status_code=201)
    async def save_workspace_history(
        workspace_id: str,
        payload: WorkspaceHistorySave,
    ) -> WorkspaceItem:
        completed_run = registry.get(payload.request_id)
        if completed_run is None:
            raise HTTPException(status_code=404, detail="Unknown API request ID")
        if completed_run.status.value != "completed" or not completed_run.output:
            raise HTTPException(
                status_code=409,
                detail="Only completed, approved output can be saved to Workspace History",
            )
        try:
            return await asyncio.to_thread(
                workspace_store().create_item,
                workspace_id,
                WorkspaceItemInput(
                    title=payload.title,
                    folder_id=payload.folder_id,
                    role=WorkspaceItemRole.HISTORY,
                    content=completed_run.output,
                    source_request_id=payload.request_id,
                ),
            )
        except (WorkspaceNotFoundError, WorkspaceConflictError) as error:
            raise_workspace_http_error(error)

    @app.post("/api/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest) -> GenerateResponse:
        if payload.workspace_id is not None:
            try:
                await asyncio.to_thread(
                    service.workspaces.store.get_workspace,
                    payload.workspace_id,
                )
            except WorkspaceNotFoundError as error:
                raise_workspace_http_error(error)
        try:
            execution = await asyncio.to_thread(
                service.generate,
                payload.to_generation_request(service.profiles.default_profile),
                retrieval=payload.retrieval_options(),
                save_run=payload.save_run,
            )
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        response = GenerateResponse.from_execution(execution)
        registry.add(response)
        return response

    @app.post("/api/generate/stream")
    async def generate_stream(payload: GenerateRequest) -> StreamingResponse:
        if payload.workspace_id is not None:
            try:
                await asyncio.to_thread(
                    service.workspaces.store.get_workspace,
                    payload.workspace_id,
                )
            except WorkspaceNotFoundError as error:
                raise_workspace_http_error(error)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[
            tuple[str, GenerateResponse | StageProgress | dict[str, str] | None]
        ] = asyncio.Queue()

        def on_progress(event: ProgressEvent) -> None:
            progress = StageProgress(
                stage=event.stage.value,
                message=event.message,
                elapsed_seconds=event.elapsed_seconds,
                generated_tokens=event.generated_tokens,
                tokens_per_second=event.tokens_per_second,
                estimate_seconds_remaining=event.estimate_seconds_remaining,
                estimate_rate_source=event.estimate_rate_source,
                estimate_is_approximate=event.estimate_is_approximate,
            )
            loop.call_soon_threadsafe(queue.put_nowait, ("progress", progress))

        async def execute() -> None:
            try:
                execution = await asyncio.to_thread(
                    service.generate,
                    payload.to_generation_request(service.profiles.default_profile),
                    retrieval=payload.retrieval_options(),
                    save_run=payload.save_run,
                    on_progress=on_progress,
                )
                response = GenerateResponse.from_execution(execution)
                registry.add(response)
                await queue.put(("result", response))
            except Exception as error:
                logger.exception("Streaming generation request failed")
                await queue.put(
                    (
                        "error",
                        {
                            "error_type": type(error).__name__,
                            "detail": "The generation request could not be executed.",
                        },
                    )
                )
            finally:
                await queue.put(("end", None))

        async def events() -> AsyncIterator[str]:
            asyncio.create_task(execute())
            while True:
                event_name, payload_value = await queue.get()
                if event_name == "end":
                    break
                if isinstance(payload_value, (GenerateResponse, StageProgress)):
                    data = payload_value.model_dump_json()
                else:
                    data = json.dumps(payload_value, ensure_ascii=False)
                yield f"event: {event_name}\ndata: {data}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs/{request_id}", response_model=GenerateResponse)
    async def get_run(request_id: str) -> GenerateResponse:
        response = registry.get(request_id)
        if response is None:
            raise HTTPException(status_code=404, detail="Unknown API request ID")
        return response

    return app
