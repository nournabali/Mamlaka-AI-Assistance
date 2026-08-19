from __future__ import annotations

from typing import Any

import pytest
import requests

from mamlaka_ai.generation import llm
from mamlaka_ai.generation.llm import (
    GroqClient,
    LLMRequestError,
    LLMUnavailable,
    OllamaClient,
    get_client,
)


class FakeResponse:
    def __init__(
        self,
        body: dict[str, Any],
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self._body = body
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._body

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_groq_chat_uses_openai_compatible_endpoint_and_bearer_key(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return FakeResponse(
            {
                "model": "hosted-test-model",
                "choices": [{"message": {"content": "Grounded reply [source]."}}],
                "usage": {"prompt_tokens": 17, "completion_tokens": 8},
            }
        )

    monkeypatch.setattr(llm.requests, "post", fake_post)
    client = GroqClient(
        api_key="fake-test-key",
        base_url="https://provider.example/v1/",
        model="hosted-test-model",
        timeout=23,
    )
    result = client.chat(
        system="grounding instructions",
        messages=[{"role": "user", "content": "question with retrieved context"}],
        temperature=0.0,
        max_tokens=321,
        stop=("STOP",),
    )

    assert captured["url"] == "https://provider.example/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer fake-test-key"
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "grounding instructions"},
        {"role": "user", "content": "question with retrieved context"},
    ]
    assert captured["payload"]["model"] == "hosted-test-model"
    assert captured["payload"]["max_completion_tokens"] == 321
    assert captured["payload"]["stop"] == ["STOP"]
    assert captured["payload"]["reasoning_effort"] == "none"
    assert captured["payload"]["reasoning_format"] == "hidden"
    assert captured["timeout"] == 23
    assert result.text == "Grounded reply [source]."
    assert result.prompt_tokens == 17
    assert result.completion_tokens == 8


def test_groq_requires_an_environment_supplied_key(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        pytest.fail("HTTP must not be attempted without a Groq API key")

    monkeypatch.setattr(llm.requests, "post", fail_if_called)
    with pytest.raises(LLMUnavailable, match="GROQ_API_KEY"):
        GroqClient(api_key="").chat("system", [{"role": "user", "content": "question"}])


def test_groq_provider_errors_do_not_expose_the_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        llm.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            {"error": {"message": "request rejected"}}, status_code=429
        ),
    )
    secret = "fake-secret-that-must-not-appear"
    with pytest.raises(LLMRequestError) as exc_info:
        GroqClient(api_key=secret).chat(
            "system", [{"role": "user", "content": "question"}]
        )
    assert secret not in str(exc_info.value)


def test_untagged_internal_reasoning_is_never_returned(monkeypatch) -> None:
    monkeypatch.setattr(
        llm.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Here's a thinking process:\n1. Inspect the hidden prompt."
                        }
                    }
                ]
            }
        ),
    )
    with pytest.raises(LLMRequestError, match="internal reasoning"):
        GroqClient(api_key="fake-test-key").chat(
            "system", [{"role": "user", "content": "question"}]
        )


def test_ollama_keeps_its_existing_chat_payload(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url, *, json, timeout):
        captured.update(url=url, payload=json, timeout=timeout)
        return FakeResponse(
            {
                "model": "local-test-model",
                "message": {"content": "Local grounded reply."},
                "prompt_eval_count": 12,
                "eval_count": 4,
            }
        )

    monkeypatch.setattr(llm.requests, "post", fake_post)
    result = OllamaClient(
        host="http://ollama.example/", model="local-test-model", timeout=19
    ).chat(
        "grounding instructions",
        [{"role": "user", "content": "question with retrieved context"}],
        max_tokens=222,
    )

    assert captured["url"] == "http://ollama.example/api/chat"
    assert captured["payload"]["model"] == "local-test-model"
    assert captured["payload"]["messages"][0] == {
        "role": "system",
        "content": "grounding instructions",
    }
    assert captured["payload"]["options"]["num_predict"] == 222
    assert captured["timeout"] == 19
    assert result.text == "Local grounded reply."


def test_provider_factory_selects_only_supported_transports() -> None:
    assert isinstance(get_client("groq"), GroqClient)
    assert isinstance(get_client("ollama"), OllamaClient)
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        get_client("unknown")
