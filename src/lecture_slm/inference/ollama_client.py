"""Small native Ollama API client."""

from dataclasses import dataclass
from typing import Any

import httpx


class OllamaError(RuntimeError):
    """Base error for Ollama failures."""


class OllamaConnectionError(OllamaError):
    """Raised when the configured Ollama server cannot be reached."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an invalid or unsuccessful response."""


@dataclass(frozen=True)
class ChatResponse:
    """Relevant response data returned by Ollama."""

    model: str
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_ns: int | None = None


class OllamaClient:
    """Non-streaming client for Ollama's native ``/api/chat`` endpoint."""

    def __init__(self, host: str, *, timeout: float = 120.0) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        *,
        model: str,
        user_message: str,
        system_message: str | None = None,
        context: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": user_message})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        try:
            response = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
        except httpx.ConnectError as error:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.host}. Is the server running?"
            ) from error
        except httpx.TimeoutException as error:
            raise OllamaConnectionError(
                f"Ollama request timed out after {self.timeout} seconds"
            ) from error
        except httpx.HTTPError as error:
            raise OllamaResponseError(f"Ollama request failed: {error}") from error

        try:
            data = response.json()
            content = data["message"]["content"]
            return ChatResponse(
                model=str(data.get("model", model)),
                content=str(content),
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
                total_duration_ns=data.get("total_duration"),
            )
        except (TypeError, KeyError, ValueError) as error:
            raise OllamaResponseError("Ollama returned an unexpected response shape") from error
