"""FastAPI application exposing the shared Lecture SLM generation service."""

import asyncio
import json
import logging
from collections import OrderedDict
from collections.abc import AsyncIterator
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from lecture_slm.api.models import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ProfileSummary,
    StageProgress,
)
from lecture_slm.generation.models import ProgressEvent
from lecture_slm.generation.service import GenerationService
from lecture_slm.inference.ollama_client import OllamaClient, OllamaError
from lecture_slm.schemas.dataset import TaskType

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

    @app.post("/api/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest) -> GenerateResponse:
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
