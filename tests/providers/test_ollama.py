"""Tests for ``reviewer.providers.ollama``.

The provider accepts an injected ``client`` so tests can swap in a fake
without monkeypatching SDK internals. Fake responses use plain dicts —
the provider is documented as tolerant of both dict and attribute-style
objects and we exercise both here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import openai
import pytest

from khonliang_reviewer import ReviewRequest
from reviewer.providers.ollama import (
    OllamaHealthcheckError,
    OllamaProvider,
    OllamaProviderConfig,
)


SUCCESS_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1776660000,
    "model": "qwen3.5",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "summary": "Ollama review summary.",
                        "findings": [
                            {
                                "severity": "concern",
                                "title": "No test for empty input",
                                "body": "Empty-string path isn't covered.",
                                "category": "testing",
                                "path": "pkg/mod.py",
                                "line": 42,
                                "suggestion": None,
                            }
                        ],
                    }
                ),
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 120,
        "completion_tokens": 80,
        "total_tokens": 200,
    },
}


class _FakeCreate:
    """Records the last kwargs passed to ``chat.completions.create``."""

    def __init__(
        self,
        *,
        response: Any = None,
        raises: BaseException | None = None,
    ):
        self._response = response
        self._raises = raises
        self.last_call: dict[str, Any] | None = None

    async def __call__(self, **kwargs: Any) -> Any:
        self.last_call = kwargs
        if self._raises is not None:
            raise self._raises
        return self._response


class _FakeModelsList:
    def __init__(self, raises: BaseException | None = None):
        self._raises = raises
        self.called = 0

    async def __call__(self) -> Any:
        self.called += 1
        if self._raises is not None:
            raise self._raises
        return {"data": [{"id": "qwen3.5"}]}


def _make_client(
    *,
    response: Any = None,
    raises: BaseException | None = None,
    models_list_raises: BaseException | None = None,
) -> SimpleNamespace:
    """Assemble a fake client with the minimal nested shape the provider uses."""
    create = _FakeCreate(response=response, raises=raises)
    models_list = _FakeModelsList(raises=models_list_raises)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        models=SimpleNamespace(list=models_list),
        _create=create,
        _models_list=models_list,
    )


def _make_request(**overrides: Any) -> ReviewRequest:
    base: dict[str, Any] = {
        "kind": "pr_diff",
        "content": "diff --git a/x b/x\n@@ -1 +1 @@\n-old\n+new\n",
        "instructions": "Review for correctness.",
        "context": {"repo_profile": "python async bus service"},
        "metadata": {"repo": "tolldog/example", "pr_number": 42},
        "request_id": "req-ollama-1",
    }
    base.update(overrides)
    return ReviewRequest(**base)


def _api_error(cls: type, message: str = "boom") -> BaseException:
    """Construct an openai exception without relying on internal signatures."""
    try:
        return cls(message)  # type: ignore[call-arg]
    except TypeError:
        # Many openai error classes demand structured args; fall back to
        # a plain subclass whose __init__ is forgiving.
        class _E(cls):  # type: ignore[misc, valid-type]
            def __init__(self, msg: str):
                Exception.__init__(self, msg)

        return _E(message)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_success_response_produces_posted_review():
    client = _make_client(response=SUCCESS_RESPONSE)
    provider = OllamaProvider(client=client)

    result = await provider.review(_make_request())

    assert result.disposition == "posted"
    assert result.backend == "ollama"
    assert result.model == "qwen3.5"
    assert result.summary == "Ollama review summary."
    assert len(result.findings) == 1
    assert result.findings[0].severity == "concern"
    assert result.findings[0].line == 42
    assert result.usage is not None
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 80
    assert result.usage.cache_read_tokens == 0
    assert result.usage.cache_creation_tokens == 0
    # Ollama does not report per-call cost. Pricing layer fills later.
    assert result.usage.estimated_api_cost_usd == 0.0
    assert result.usage.repo == "tolldog/example"
    assert result.usage.pr_number == 42


async def test_prompt_passed_in_messages_and_includes_schema():
    client = _make_client(response=SUCCESS_RESPONSE)
    await OllamaProvider(client=client).review(_make_request())

    call = client._create.last_call
    assert call is not None
    assert call["model"] == "qwen3.5"
    assert call["response_format"] == {"type": "json_object"}
    messages = call["messages"]
    assert len(messages) == 1
    prompt = messages[0]["content"]
    assert "Review for correctness." in prompt
    assert "diff --git" in prompt
    # schema is embedded inline because Ollama's OpenAI-compat surface
    # uses json_object mode (not strict schema validation)
    assert '"severity"' in prompt


async def test_model_override_from_request_metadata():
    client = _make_client(response=SUCCESS_RESPONSE)
    request = _make_request(metadata={"model": "kimi-k2.5:cloud", "repo": "x/y"})

    result = await OllamaProvider(client=client).review(request)

    assert result.model == "kimi-k2.5:cloud"
    assert client._create.last_call["model"] == "kimi-k2.5:cloud"


async def test_config_default_used_when_request_has_no_model():
    client = _make_client(response=SUCCESS_RESPONSE)
    provider = OllamaProvider(
        OllamaProviderConfig(default_model="glm-4.7-flash"),
        client=client,
    )

    result = await provider.review(_make_request())

    assert result.model == "glm-4.7-flash"


async def test_attribute_style_response_supported():
    """openai SDK returns pydantic models; the extractor must tolerate both."""
    message = SimpleNamespace(
        content=SUCCESS_RESPONSE["choices"][0]["message"]["content"],
        role="assistant",
    )
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=80, total_tokens=200)
    response = SimpleNamespace(
        choices=[choice],
        usage=usage,
        model="qwen3.5",
    )
    client = _make_client(response=response)

    result = await OllamaProvider(client=client).review(_make_request())

    assert result.disposition == "posted"
    assert result.summary == "Ollama review summary."
    assert result.usage is not None
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 80


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


async def test_connection_error_errored():
    client = _make_client(raises=_api_error(openai.APIConnectionError))
    result = await OllamaProvider(client=client).review(_make_request())
    assert result.disposition == "errored"
    assert result.error_category == "backend_error"
    assert "unreachable" in result.error


async def test_timeout_error_errored():
    client = _make_client(raises=_api_error(openai.APITimeoutError))
    provider = OllamaProvider(
        OllamaProviderConfig(timeout_seconds=30.0),
        client=client,
    )
    result = await provider.review(_make_request())
    assert result.disposition == "errored"
    assert result.error_category == "backend_error"
    assert "timed out" in result.error


async def test_authentication_error_errored_auth_category():
    client = _make_client(raises=_api_error(openai.AuthenticationError))
    result = await OllamaProvider(client=client).review(_make_request())
    assert result.disposition == "errored"
    assert result.error_category == "auth_not_provisioned"


async def test_not_found_error_errored():
    client = _make_client(raises=_api_error(openai.NotFoundError))
    result = await OllamaProvider(client=client).review(_make_request())
    assert result.disposition == "errored"
    assert result.error_category == "backend_error"
    assert "not found" in result.error


async def test_generic_api_error_errored():
    client = _make_client(raises=_api_error(openai.APIError))
    result = await OllamaProvider(client=client).review(_make_request())
    assert result.disposition == "errored"
    assert result.error_category == "backend_error"


# ---------------------------------------------------------------------------
# Defensive envelope parsing
# ---------------------------------------------------------------------------


async def test_empty_choices_errored():
    response = dict(SUCCESS_RESPONSE)
    response["choices"] = []
    client = _make_client(response=response)
    result = await OllamaProvider(client=client).review(_make_request())
    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "no message content" in result.error


async def test_non_json_content_errored():
    response = dict(SUCCESS_RESPONSE)
    # deep-copy the choices to avoid mutating the fixture
    response["choices"] = [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "plain text, not json"},
            "finish_reason": "stop",
        }
    ]
    client = _make_client(response=response)
    result = await OllamaProvider(client=client).review(_make_request())
    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "not JSON" in result.error


async def test_non_object_json_content_errored():
    response = dict(SUCCESS_RESPONSE)
    response["choices"] = [
        {
            "index": 0,
            "message": {"role": "assistant", "content": json.dumps(["a", "b"])},
            "finish_reason": "stop",
        }
    ]
    client = _make_client(response=response)
    result = await OllamaProvider(client=client).review(_make_request())
    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "not a JSON object" in result.error


async def test_findings_filters_non_dict_items():
    response = dict(SUCCESS_RESPONSE)
    response["choices"] = [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "summary": "mixed",
                        "findings": [
                            {"severity": "nit", "title": "keep", "body": "ok"},
                            "rogue",
                            None,
                            {"severity": "comment", "title": "also keep", "body": "ok"},
                        ],
                    }
                ),
            },
            "finish_reason": "stop",
        }
    ]
    client = _make_client(response=response)
    result = await OllamaProvider(client=client).review(_make_request())
    assert result.disposition == "posted"
    assert [f.title for f in result.findings] == ["keep", "also keep"]


async def test_missing_usage_tokens_zero():
    response = dict(SUCCESS_RESPONSE)
    response = dict(response)
    response.pop("usage", None)
    client = _make_client(response=response)
    result = await OllamaProvider(client=client).review(_make_request())
    assert result.disposition == "posted"
    assert result.usage is not None
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0


async def test_string_token_values_coerced():
    response = dict(SUCCESS_RESPONSE)
    response["usage"] = {
        "prompt_tokens": "100",
        "completion_tokens": "42.0",
    }
    client = _make_client(response=response)
    result = await OllamaProvider(client=client).review(_make_request())
    assert result.usage is not None
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 42


# ---------------------------------------------------------------------------
# healthcheck
# ---------------------------------------------------------------------------


async def test_healthcheck_success():
    client = _make_client()
    await OllamaProvider(client=client).healthcheck()
    assert client._models_list.called == 1


async def test_healthcheck_unreachable_raises():
    client = _make_client(models_list_raises=_api_error(openai.APIConnectionError))
    with pytest.raises(OllamaHealthcheckError) as excinfo:
        await OllamaProvider(client=client).healthcheck()
    assert "not reachable" in str(excinfo.value)


async def test_healthcheck_error_is_runtime_error():
    """Callers can catch RuntimeError broadly."""
    assert issubclass(OllamaHealthcheckError, RuntimeError)


# ---------------------------------------------------------------------------
# Provider identity
# ---------------------------------------------------------------------------


async def test_provider_name_is_ollama():
    assert OllamaProvider.name == "ollama"
    assert OllamaProvider(client=_make_client()).name == "ollama"
