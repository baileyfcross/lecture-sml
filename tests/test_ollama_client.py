import httpx
import pytest

from lecture_slm.inference.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
)


class StubResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


def test_chat_uses_native_ollama_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    request: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> StubResponse:
        request.update({"url": url, **kwargs})
        return StubResponse({"model": "model-x", "message": {"content": "Hello"}})

    monkeypatch.setattr("lecture_slm.inference.ollama_client.httpx.post", fake_post)
    response = OllamaClient("http://ollama.example/").chat(
        model="model-x",
        system_message="Be concise.",
        context="Prior lesson: variables.",
        user_message="Explain functions.",
    )

    assert request["url"] == "http://ollama.example/api/chat"
    payload = request["json"]
    assert isinstance(payload, dict)
    assert payload["stream"] is False
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == ["system", "user", "user"]
    assert response.model == "model-x"
    assert response.content == "Hello"


def test_chat_maps_connection_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_post(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("lecture_slm.inference.ollama_client.httpx.post", fail_post)
    with pytest.raises(OllamaConnectionError, match="Could not connect"):
        OllamaClient("http://localhost:11434").chat(model="model-x", user_message="Hi")


def test_chat_rejects_malformed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lecture_slm.inference.ollama_client.httpx.post",
        lambda *args, **kwargs: StubResponse({"unexpected": True}),
    )
    with pytest.raises(OllamaResponseError, match="unexpected response shape"):
        OllamaClient("http://localhost:11434").chat(model="model-x", user_message="Hi")
