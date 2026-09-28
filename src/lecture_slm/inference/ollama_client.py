"""Small native Ollama API client."""

from dataclasses import dataclass

import httpx


class OllamaError(RuntimeError):
    """Base error for Ollama failures."""


class OllamaConnectionError(OllamaError):
    """Raised when the configured Ollama server cannot be reached."""


class OllamaTimeoutError(OllamaError):
    """Raised when an Ollama request exceeds its timeout."""


class OllamaModelNotFoundError(OllamaError):
    """Raised when the configured model is not available on the server."""


class OllamaHTTPError(OllamaError):
    """Raised when Ollama returns an unsuccessful HTTP status."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns a malformed response."""


class OllamaIncompleteResponseError(OllamaError):
    """Raised when Ollama did not finish generating a response."""


@dataclass(frozen=True)
class ChatResponse:
    """Relevant response data returned by Ollama."""

    model: str
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_duration_ns: int | None = None
    completion_reason: str | None = None
    thinking_content: str | None = None


class OllamaClient:
    """Non-streaming client for Ollama's native ``/api/chat`` endpoint."""

    def __init__(self, host: str, *, timeout: float = 120.0) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout

    def available_models(self) -> set[str]:
        """Return model names reported by Ollama's native tags endpoint."""

        data = self._request_json("GET", "/api/tags")
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise OllamaResponseError("Ollama returned an unexpected model-list response")
        names: set[str] = set()
        for entry in data["models"]:
            if not isinstance(entry, dict):
                raise OllamaResponseError("Ollama returned an invalid model-list entry")
            name = entry.get("name")
            if not isinstance(name, str):
                raise OllamaResponseError("Ollama returned a model entry without a name")
            names.add(name)
        return names

    def ensure_model_available(self, model: str) -> None:
        """Raise a specific error when the requested model is not installed."""

        if model not in self.available_models():
            raise OllamaModelNotFoundError(
                f"Configured model '{model}' is not available at {self.host}"
            )

    def chat(
        self,
        *,
        model: str,
        user_message: str,
        system_message: str | None = None,
        context: str | None = None,
        options: dict[str, str | int | float | bool] | None = None,
        think: bool | None = None,
        keep_alive: str | int | None = None,
        allow_empty_content: bool = False,
    ) -> ChatResponse:
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": user_message})
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        if think is not None:
            payload["think"] = think
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        data = self._request_json("POST", "/api/chat", payload=payload)
        if not isinstance(data, dict):
            raise OllamaResponseError("Ollama returned an unexpected chat response")
        if data.get("done") is False:
            raise OllamaIncompleteResponseError("Ollama did not complete model generation")
        if data.get("done") is not True:
            raise OllamaResponseError("Ollama response is missing a valid completion status")
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaResponseError("Ollama returned an unexpected message response")
        content = message["content"]
        thinking_content = message.get("thinking")
        if thinking_content is not None and not isinstance(thinking_content, str):
            raise OllamaResponseError("Ollama response field 'message.thinking' must be a string")
        if not content.strip() and not allow_empty_content and not (thinking_content or "").strip():
            raise OllamaIncompleteResponseError("Ollama completed without generating content")
        return ChatResponse(
            model=self._string_value(data, "model", model),
            content=content,
            prompt_tokens=self._optional_integer(data, "prompt_eval_count"),
            completion_tokens=self._optional_integer(data, "eval_count"),
            total_duration_ns=self._optional_integer(data, "total_duration"),
            load_duration_ns=self._optional_integer(data, "load_duration"),
            prompt_eval_duration_ns=self._optional_integer(data, "prompt_eval_duration"),
            eval_duration_ns=self._optional_integer(data, "eval_duration"),
            completion_reason=self._optional_string(data, "done_reason"),
            thinking_content=thinking_content,
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
    ) -> object:
        try:
            if payload is None:
                response = httpx.request(method, f"{self.host}{path}", timeout=self.timeout)
            else:
                response = httpx.request(
                    method,
                    f"{self.host}{path}",
                    json=payload,
                    timeout=self.timeout,
                )
            response.raise_for_status()
        except httpx.ConnectError as error:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.host}. Is the server running?"
            ) from error
        except httpx.TimeoutException as error:
            raise OllamaTimeoutError(
                f"Ollama request to {path} timed out after {self.timeout} seconds"
            ) from error
        except httpx.HTTPStatusError as error:
            raise OllamaHTTPError(
                f"Ollama returned HTTP {error.response.status_code} for {path}"
            ) from error
        except httpx.HTTPError as error:
            raise OllamaHTTPError(f"Ollama request to {path} failed: {error}") from error
        try:
            return response.json()
        except (ValueError, UnicodeDecodeError) as error:
            raise OllamaResponseError("Ollama returned invalid JSON") from error

    @staticmethod
    def _string_value(data: dict[str, object], key: str, default: str) -> str:
        value = data.get(key, default)
        if not isinstance(value, str):
            raise OllamaResponseError(f"Ollama response field '{key}' must be a string")
        return value

    @staticmethod
    def _optional_integer(data: dict[str, object], key: str) -> int | None:
        value = data.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise OllamaResponseError(
                f"Ollama response field '{key}' must be a non-negative integer"
            )
        return value

    @staticmethod
    def _optional_string(data: dict[str, object], key: str) -> str | None:
        value = data.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise OllamaResponseError(f"Ollama response field '{key}' must be a string")
        return value
