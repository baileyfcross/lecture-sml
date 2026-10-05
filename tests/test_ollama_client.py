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
    assert all(
        request[2]["timeout"] == namespace["SMOKE_TEST_TIMEOUT_SECONDS"] for request in requests
    )
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
    smoke_test_timeout = runpy.run_path(str(ROOT / "scripts/test_ollama.py"))[
        "SMOKE_TEST_TIMEOUT_SECONDS"
    ]
    with pytest.raises(OllamaTimeoutError, match="timed out"):
        OllamaClient("http://ollama.example", timeout=smoke_test_timeout).available_models()


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


def test_chat_parses_ollama_timing_and_token_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    response_payload = {
        "model": "model-x",
        "message": {"content": "OK"},
        "done": True,
        "total_duration": 1_020_000_000,
        "load_duration": 6_000_000,
        "prompt_eval_duration": 690_000_000,
        "eval_duration": 310_000_000,
        "prompt_eval_count": 19,
        "eval_count": 2,
    }
    monkeypatch.setattr(
        "lecture_slm.inference.ollama_client.httpx.request",
        lambda *args, **kwargs: StubResponse(response_payload),
    )

    result = OllamaClient("http://ollama.example").chat(
        model="model-x",
        user_message="Reply OK.",
    )

    assert result.total_duration_ns == 1_020_000_000
    assert result.load_duration_ns == 6_000_000
    assert result.prompt_eval_duration_ns == 690_000_000
    assert result.eval_duration_ns == 310_000_000
    assert result.prompt_tokens == 19
    assert result.completion_tokens == 2


def test_chat_parses_completion_reason_and_thinking_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lecture_slm.inference.ollama_client.httpx.request",
        lambda *args, **kwargs: StubResponse(
            {
                "model": "model-x",
                "message": {"content": "Answer", "thinking": "Reasoning trace"},
                "done": True,
                "done_reason": "length",
                "eval_count": 256,
            }
        ),
    )

    result = OllamaClient("http://ollama.example").chat(
        model="model-x",
        user_message="Explain DNS.",
    )

    assert result.completion_reason == "length"
    assert result.thinking_content == "Reasoning trace"


def test_benchmark_can_preserve_empty_final_content_with_thinking_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lecture_slm.inference.ollama_client.httpx.request",
        lambda *args, **kwargs: StubResponse(
            {
                "model": "model-x",
                "message": {"content": "", "thinking": "private reasoning"},
                "done": True,
                "done_reason": "length",
                "eval_count": 256,
                "eval_duration": 10_000_000_000,
            }
        ),
    )

    response = OllamaClient("http://ollama.example").chat(
        model="model-x",
        user_message="Benchmark prompt.",
        think=True,
        allow_empty_content=True,
    )

    assert response.content == ""
    assert response.thinking_content == "private reasoning"
    assert response.completion_reason == "length"
    assert response.completion_tokens == 256


def test_empty_final_content_is_still_rejected_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lecture_slm.inference.ollama_client.httpx.request",
        lambda *args, **kwargs: StubResponse(
            {
                "model": "model-x",
                "message": {"content": ""},
                "done": True,
                "done_reason": "stop",
            }
        ),
    )
    with pytest.raises(OllamaIncompleteResponseError):
        OllamaClient("http://ollama.example").chat(
            model="model-x",
            user_message="Normal request.",
        )


def test_native_chat_forwards_json_schema_format(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    response_payload = {
        "model": "model-x",
        "message": {"content": "{}"},
        "done": True,
    }

    def fake_request(method: str, url: str, **kwargs: object) -> StubResponse:
        captured.update(kwargs)
        return StubResponse(response_payload)

    monkeypatch.setattr("lecture_slm.inference.ollama_client.httpx.request", fake_request)
    response_schema = {"type": "object", "required": ["task"]}
    OllamaClient("http://ollama.example").chat(
        model="model-x",
        user_message="Return a plan.",
        format=response_schema,
    )

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["format"] == response_schema
