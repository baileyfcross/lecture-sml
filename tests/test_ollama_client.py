import runpy
import sys
from pathlib import Path

import httpx
import pytest

from lecture_slm.inference.ollama_client import (
    OllamaClient,
    OllamaHTTPError,
    OllamaIncompleteResponseError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
)

ROOT = Path(__file__).parents[1]
MODEL_CONFIG = ROOT / "configs/models/qwen35-9b.yaml"


class StubResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://ollama.example")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("request failed", request=request, response=response)

    def json(self) -> object:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def tags_response(*model_names: str) -> StubResponse:
    return StubResponse({"models": [{"name": name} for name in model_names]})


def chat_response(**overrides: object) -> StubResponse:
    payload: dict[str, object] = {
        "model": "model-x",
        "message": {"content": "OK"},
        "done": True,
        "total_duration": 1_020_000_000,
        "load_duration": 6_000_000,
        "prompt_eval_duration": 690_000_000,
        "eval_duration": 310_000_000,
        "prompt_eval_count": 12,
        "eval_count": 1,
    }
    payload.update(overrides)
    return StubResponse(payload)


def test_health_cli_sends_bounded_native_request_and_reports_timings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[tuple[str, str, dict[str, object]]] = []
    responses = [tags_response("qwen3.5:9b"), chat_response()]

    def fake_request(method: str, url: str, **kwargs: object) -> StubResponse:
        requests.append((method, url, kwargs))
        return responses.pop(0)

    monkeypatch.setattr("lecture_slm.inference.ollama_client.httpx.request", fake_request)
    monkeypatch.setattr(sys, "argv", ["test_ollama.py", "--config", str(MODEL_CONFIG)])
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.example:11434/")

    namespace = runpy.run_path(str(ROOT / "scripts/test_ollama.py"))
    main = namespace["main"]
    assert callable(main)
    assert main() == 0

    assert [(method, url) for method, url, _ in requests] == [
        ("GET", "http://ollama.example:11434/api/tags"),
        ("POST", "http://ollama.example:11434/api/chat"),
    ]
    assert all(request[2]["timeout"] == 30.0 for request in requests)
    chat_payload = requests[1][2]["json"]
    assert isinstance(chat_payload, dict)
    assert chat_payload["model"] == "qwen3.5:9b"
    assert chat_payload["stream"] is False
    assert chat_payload["think"] is False
    assert chat_payload["keep_alive"] == "10m"
    assert chat_payload["options"] == {"num_predict": 8, "num_ctx": 2048}
    assert chat_payload["messages"][-1]["content"] == "Reply with only the word OK."

    output = capsys.readouterr().out
    assert "Ollama smoke test passed." in output
    assert "Total duration: 1.02 s" in output
    assert "Load duration: 0.01 s" in output
    assert "Prompt processing: 0.69 s" in output
    assert "Generation: 0.31 s" in output
    assert "Prompt tokens: 12" in output
    assert "Generated tokens: 1" in output


def test_missing_model_reports_expected_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lecture_slm.inference.ollama_client.httpx.request",
        lambda *args, **kwargs: tags_response("another-model"),
    )
    with pytest.raises(OllamaModelNotFoundError, match="model-x"):
        OllamaClient("http://ollama.example").ensure_model_available("model-x")


def test_timeout_has_a_distinct_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout_request(*args: object, **kwargs: object) -> None:
        raise httpx.ReadTimeout("slow response")

    monkeypatch.setattr("lecture_slm.inference.ollama_client.httpx.request", timeout_request)
    with pytest.raises(OllamaTimeoutError, match="timed out"):
        OllamaClient("http://ollama.example", timeout=30).available_models()


def test_http_error_has_a_distinct_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lecture_slm.inference.ollama_client.httpx.request",
        lambda *args, **kwargs: StubResponse({}, status_code=500),
    )
    with pytest.raises(OllamaHTTPError, match="HTTP 500"):
        OllamaClient("http://ollama.example").available_models()


def test_malformed_response_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lecture_slm.inference.ollama_client.httpx.request",
        lambda *args, **kwargs: StubResponse({"unexpected": True}),
    )
    with pytest.raises(OllamaResponseError, match="model-list response"):
        OllamaClient("http://ollama.example").available_models()


def test_incomplete_generation_has_a_distinct_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lecture_slm.inference.ollama_client.httpx.request",
        lambda *args, **kwargs: chat_response(done=False),
    )
    with pytest.raises(OllamaIncompleteResponseError, match="did not complete"):
        OllamaClient("http://ollama.example").chat(model="model-x", user_message="Hi")


def test_invalid_json_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lecture_slm.inference.ollama_client.httpx.request",
        lambda *args, **kwargs: StubResponse(ValueError("invalid json")),
    )
    with pytest.raises(OllamaResponseError, match="invalid JSON"):
        OllamaClient("http://ollama.example").available_models()
