"""LLM clients for Groq and Ollama."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence

import requests

from mamlaka_ai.config import settings


class LLMError(RuntimeError):
    """Base class for language-model transport failures."""


class LLMUnavailable(LLMError):
    """The configured provider, credentials, endpoint, or model is unavailable."""


class LLMRequestError(LLMError):
    """The provider was reached but returned an error or unusable payload."""


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ORPHAN_THINK_RE = re.compile(r"^\s*<think>.*?(?:</think>)?", re.DOTALL | re.IGNORECASE)
_EXPOSED_REASONING_RE = re.compile(
    r"^\s*(?:"
    r"here(?:'s| is) (?:a|the|my) (?:thinking|reasoning) process"
    r"|(?:let me|let's) (?:analy[sz]e|reason|think)"
    r"|(?:analysis|thinking|reasoning)\s*:"
    r")",
    re.IGNORECASE,
)


def strip_reasoning(text: str) -> str:
    """Remove visible reasoning blocks from model output."""
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    if "<think>" in cleaned.lower():
        cleaned = _ORPHAN_THINK_RE.sub("", cleaned)
    return cleaned.strip()


def _final_output(text: str, provider: str) -> str:
    """Reject output that exposes untagged reasoning."""
    cleaned = strip_reasoning(text)
    if _EXPOSED_REASONING_RE.match(cleaned):
        raise LLMRequestError(
            f"{provider} returned internal reasoning instead of a final answer. Please retry."
        )
    return cleaned


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration_ms: int | None = None


class LLMClient(Protocol):
    """Interface shared by LLM providers."""

    provider: str
    model: str

    def chat(
        self,
        system: str,
        messages: Sequence[Dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] = (),
    ) -> LLMResponse: ...

    def health(self) -> Dict[str, Any]: ...


def _error_detail(response: requests.Response) -> str:
    try:
        body = response.json()
        detail = (body.get("error") or {}).get("message")
        if detail:
            return str(detail)[:400]
    except (ValueError, AttributeError, TypeError):
        pass
    return response.text.strip()[:400] or f"HTTP {response.status_code}"


class GroqClient:
    """Groq's OpenAI-compatible hosted Chat Completions transport."""

    provider = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.api_key = settings.groq_api_key if api_key is None else api_key.strip()
        self.base_url = (base_url or settings.groq_base_url).rstrip("/")
        self.host = self.base_url  # compatibility for diagnostics/evaluator output
        self.model = model or settings.llm_model
        self.timeout = timeout or settings.llm_timeout

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def list_models(self) -> List[str]:
        if not self.api_key:
            return []
        try:
            response = requests.get(
                f"{self.base_url}/models", headers=self._headers, timeout=10
            )
            response.raise_for_status()
            return [str(item.get("id", "")) for item in response.json().get("data", [])]
        except (requests.RequestException, ValueError, AttributeError):
            return []

    def is_available(self) -> bool:
        return bool(self.api_key and self.list_models())

    def health(self) -> Dict[str, Any]:
        models = self.list_models()
        return {
            "provider": self.provider,
            "endpoint": self.base_url,
            "model": self.model,
            "configured": bool(self.api_key),
            "reachable": bool(models),
            "model_available": self.model in models,
            "available_models": models,
        }

    def chat(
        self,
        system: str,
        messages: Sequence[Dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] = (),
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMUnavailable(
                "GROQ_API_KEY is not configured. Add it to .env or the deployment secrets."
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_completion_tokens": max_tokens or settings.llm_max_tokens,
            "top_p": 1.0,
            "stream": False,
        }
        if settings.llm_disable_thinking:
            # Ask Qwen to return only the final answer.
            payload["reasoning_effort"] = "none"
            payload["reasoning_format"] = "hidden"
        if stop:
            payload["stop"] = list(stop)

        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise LLMRequestError(
                f"Groq did not respond within {self.timeout}s."
            ) from exc
        except requests.RequestException as exc:
            raise LLMUnavailable(
                f"Cannot reach Groq at {self.base_url} ({exc.__class__.__name__})."
            ) from exc

        if response.status_code in {401, 403}:
            raise LLMUnavailable(
                "Groq rejected GROQ_API_KEY. Check the deployment secret and try again."
            )
        if response.status_code == 404:
            raise LLMUnavailable(
                f"Groq model '{self.model}' is unavailable. Check LLM_MODEL."
            )
        if not response.ok:
            raise LLMRequestError(f"Groq error: {_error_detail(response)}")

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("Groq returned an unusable JSON response.") from exc

        usage = body.get("usage") or {}
        return LLMResponse(
            text=_final_output(str(content or ""), "Groq"),
            model=str(body.get("model") or self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


class OllamaClient:
    """Optional local Ollama transport implementing the same client interface."""

    provider = "ollama"

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.host = (host or settings.ollama_base_url).rstrip("/")
        self.base_url = self.host
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.llm_timeout

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.host}/api/version", timeout=5)
            return response.ok
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            return [str(model.get("name", "")) for model in response.json().get("models", [])]
        except (requests.RequestException, ValueError, AttributeError):
            return []

    def model_installed(self) -> bool:
        installed = self.list_models()
        if self.model in installed:
            return True
        base = self.model.split(":", 1)[0]
        return any(name.split(":", 1)[0] == base for name in installed)

    def health(self) -> Dict[str, Any]:
        reachable = self.is_available()
        models = self.list_models() if reachable else []
        return {
            "provider": self.provider,
            "endpoint": self.host,
            "model": self.model,
            "configured": True,
            "reachable": reachable,
            "model_available": self.model_installed() if reachable else False,
            "available_models": models,
        }

    def chat(
        self,
        system: str,
        messages: Sequence[Dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] = (),
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "options": {
                "temperature": settings.llm_temperature if temperature is None else temperature,
                "num_ctx": settings.llm_num_ctx,
                "num_predict": max_tokens or settings.llm_max_tokens,
                "top_p": 1.0,
                "seed": 0,
            },
        }
        if stop:
            payload["options"]["stop"] = list(stop)
        if settings.llm_disable_thinking:
            payload["think"] = False

        try:
            response = requests.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
        except requests.Timeout as exc:
            raise LLMRequestError(
                f"Ollama did not respond within {self.timeout}s."
            ) from exc
        except requests.RequestException as exc:
            raise LLMUnavailable(
                f"Cannot reach Ollama at {self.host} ({exc.__class__.__name__})."
            ) from exc

        if response.status_code == 404:
            raise LLMUnavailable(
                f"Ollama model '{self.model}' is unavailable. Run: ollama pull {self.model}"
            )
        if not response.ok:
            detail = _error_detail(response)
            if "think" in detail.lower() and payload.pop("think", None) is not None:
                try:
                    retry = requests.post(
                        f"{self.host}/api/chat", json=payload, timeout=self.timeout
                    )
                except requests.RequestException as exc:
                    raise LLMRequestError(f"Ollama retry failed: {exc}") from exc
                if retry.ok:
                    response = retry
                else:
                    raise LLMRequestError(f"Ollama error: {detail}")
            else:
                raise LLMRequestError(f"Ollama error: {detail}")

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMRequestError("Ollama returned a non-JSON response.") from exc

        content = (body.get("message") or {}).get("content", "")
        duration = body.get("total_duration")
        return LLMResponse(
            text=_final_output(content, "Ollama"),
            model=str(body.get("model") or self.model),
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
            duration_ms=int(duration / 1_000_000) if isinstance(duration, int) else None,
        )


def get_client(provider: str | None = None) -> LLMClient:
    """Create the configured LLM client."""
    selected = (provider or settings.llm_provider).strip().lower()
    if selected == "groq":
        return GroqClient()
    if selected == "ollama":
        return OllamaClient()
    raise ValueError(
        f"Unsupported LLM_PROVIDER '{selected}'. Supported providers: groq, ollama."
    )
