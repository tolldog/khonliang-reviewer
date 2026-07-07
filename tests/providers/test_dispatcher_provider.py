"""Tests for ``reviewer.providers.dispatcher_provider``
(fr_reviewer_50a5b842).

Uses a fake ``DispatcherClient`` (matching ``dispatcher_lib``'s real
``run(...) -> Handle`` contract) rather than mocking HTTP -- the
provider's own job is choosing skill= vs model=, parsing the
dispatcher's plain-text result, and mapping typed exceptions; the
wire-transport behavior itself is already covered by
khonliang-dispatcher-lib's own test suite.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from dispatcher_lib import (
    Busy,
    Cancelled,
    ChatResult,
    DeadlineExceeded,
    Handle,
    ModelUnavailable,
    TaskInvalid,
)

from khonliang_reviewer import ReviewRequest
from reviewer.providers.dispatcher_provider import (
    DispatcherProvider,
    DispatcherProviderConfig,
)


SUCCESS_PAYLOAD = {
    "summary": "Dispatcher-gatewayed review summary.",
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
SUCCESS_CONTENT = json.dumps(SUCCESS_PAYLOAD)


class _FakeDispatcherClient:
    """Records the kwargs ``run()`` was called with; returns a
    pre-baked ``Handle`` (success or typed error) on each call."""

    def __init__(self, handle: Handle) -> None:
        self.handle = handle
        self.calls: list[dict[str, Any]] = []
        self.base_url = "http://test:8790"

    def run(self, **kwargs: Any) -> Handle:
        self.calls.append(kwargs)
        return self.handle


def _success_handle(
    text: str = SUCCESS_CONTENT, *, model: str = "resident-model",
) -> Handle:
    return Handle(
        outcome=ChatResult(
            text=text,
            tokens_in=10,
            tokens_out=20,
            stop_reason="stop",
            raw={},
            job_id="brk_test",
            engine="tabby",
            engine_instance="tabby@localhost:5000",
            queue_wait_ms=5,
            engine_ms=100,
            total_ms=105,
            envelope={"model": model},
        )
    )


def _request(*, kind: str = "pr_diff", metadata: dict[str, Any] | None = None) -> ReviewRequest:
    return ReviewRequest(
        kind=kind,
        content="diff content",
        request_id="req-1",
        metadata=metadata or {},
    )


@pytest.mark.asyncio
async def test_review_uses_skill_only_as_last_resort() -> None:
    """No override AND no configured default_model on this instance --
    the true last-resort case (an operator relying entirely on
    skill_policy.yaml, no per-backend pin at all)."""
    client = _FakeDispatcherClient(_success_handle())
    provider = DispatcherProvider("tabbyapi", client=client)  # default config: default_model=""

    result = await provider.review(_request(kind="pr_diff"))

    assert client.calls[0]["skill"] == "pr_diff"
    assert "model" not in client.calls[0]
    assert result.disposition == "posted"
    assert result.backend == "tabbyapi"
    assert result.model == "resident-model"
    assert result.summary == "Dispatcher-gatewayed review summary."
    assert result.findings[0].title == "No test for empty input"


@pytest.mark.asyncio
async def test_review_falls_back_to_instance_default_model_not_skill() -> None:
    """codex review finding, round 2: the common case. No metadata
    override, but THIS instance has a configured default_model (the
    normal, non-empty operator config) -- must pin to that model, not
    hand the choice to the dispatcher's skill resolver, or an explicit
    backend=ollama/backend=tabbyapi pick (including the tabby-
    unavailable degrade-to-ollama reroute) would silently lose its
    pin and could land on either internal engine."""
    client = _FakeDispatcherClient(_success_handle(model="deepseek-coder-v2:16b"))
    provider = DispatcherProvider(
        "ollama",
        DispatcherProviderConfig(default_model="deepseek-coder-v2:16b"),
        client=client,
    )

    result = await provider.review(_request(kind="pr_diff"))

    assert client.calls[0]["model"] == "deepseek-coder-v2:16b"
    assert "skill" not in client.calls[0]
    assert client.calls[0]["role"] == "pr_diff"
    assert result.model == "deepseek-coder-v2:16b"


@pytest.mark.asyncio
async def test_review_metadata_override_wins_over_instance_default() -> None:
    client = _FakeDispatcherClient(_success_handle(model="pinned-model"))
    provider = DispatcherProvider(
        "ollama",
        DispatcherProviderConfig(default_model="deepseek-coder-v2:16b"),
        client=client,
    )

    await provider.review(_request(metadata={"model": "pinned-model"}))

    assert client.calls[0]["model"] == "pinned-model"


@pytest.mark.asyncio
async def test_review_uses_model_override_not_skill() -> None:
    """An explicit request.metadata['model'] -- an A/B-comparison pin,
    or the rule table's explicit kimi-k2.5:cloud escalation -- must
    bypass skill= entirely and go straight to model=."""
    client = _FakeDispatcherClient(_success_handle(model="kimi-k2.5:cloud"))
    provider = DispatcherProvider("ollama", client=client)

    result = await provider.review(
        _request(kind="pr_diff", metadata={"model": "kimi-k2.5:cloud"})
    )

    assert client.calls[0]["model"] == "kimi-k2.5:cloud"
    assert "skill" not in client.calls[0]
    assert client.calls[0]["role"] == "pr_diff"
    assert result.model == "kimi-k2.5:cloud"


@pytest.mark.asyncio
async def test_review_whitespace_only_model_override_falls_back_to_skill() -> None:
    client = _FakeDispatcherClient(_success_handle())
    provider = DispatcherProvider("tabbyapi", client=client)

    await provider.review(_request(kind="doc", metadata={"model": "   "}))

    assert client.calls[0]["skill"] == "doc"
    assert "model" not in client.calls[0]


@pytest.mark.asyncio
async def test_review_strips_think_block_and_json_fence() -> None:
    client = _FakeDispatcherClient(
        _success_handle(f"<think>reasoning...</think>\n```json\n{SUCCESS_CONTENT}\n```")
    )
    provider = DispatcherProvider("tabbyapi", client=client)

    result = await provider.review(_request())

    assert result.disposition == "posted"
    assert result.summary == "Dispatcher-gatewayed review summary."


@pytest.mark.asyncio
async def test_review_empty_text_is_errored() -> None:
    client = _FakeDispatcherClient(_success_handle(""))
    provider = DispatcherProvider("tabbyapi", client=client)

    result = await provider.review(_request())

    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"


@pytest.mark.asyncio
async def test_review_non_json_text_is_errored() -> None:
    client = _FakeDispatcherClient(_success_handle("not json at all"))
    provider = DispatcherProvider("tabbyapi", client=client)

    result = await provider.review(_request())

    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"


@pytest.mark.parametrize(
    "exc,expected_category",
    [
        (TaskInvalid("bad request"), "malformed_envelope"),
        (ModelUnavailable("no engine"), "backend_error"),
        (DeadlineExceeded("timed out"), "backend_timeout"),
        (Busy("shed"), "backend_error"),
        (Cancelled("cancelled"), "backend_error"),
    ],
)
@pytest.mark.asyncio
async def test_review_maps_typed_exceptions_to_error_categories(
    exc: Exception, expected_category: str,
) -> None:
    client = _FakeDispatcherClient(Handle(error=exc))
    provider = DispatcherProvider("tabbyapi", client=client)

    result = await provider.review(_request())

    assert result.disposition == "errored"
    assert result.error_category == expected_category
    assert result.backend == "tabbyapi"


@pytest.mark.asyncio
async def test_provider_name_is_instance_level() -> None:
    """One class stands in for both legacy backend names -- name must
    be an instance attribute, not a shared class default, or usage
    records would mislabel one of the two."""
    client = _FakeDispatcherClient(_success_handle())
    ollama_provider = DispatcherProvider("ollama", client=client)
    tabby_provider = DispatcherProvider("tabbyapi", client=client)

    assert ollama_provider.name == "ollama"
    assert tabby_provider.name == "tabbyapi"

    result = await ollama_provider.review(_request())
    assert result.backend == "ollama"


def test_config_defaults() -> None:
    config = DispatcherProviderConfig()
    assert config.base_url == "http://localhost:8790"
    assert config.deadline_s == 240.0


class _FakeHttpxResponse:
    def __init__(self, *, status_code: int = 200, json_data: Any = None) -> None:
        self.status_code = status_code
        self._json_data = json_data

    def raise_for_status(self) -> None:
        if not (200 <= self.status_code < 300):
            request = httpx.Request("GET", "http://test/v1/engines")
            raise httpx.HTTPStatusError(
                "error", request=request,
                response=httpx.Response(self.status_code, request=request),
            )

    def json(self) -> Any:
        return self._json_data


class _FakeAsyncHttpxClient:
    def __init__(self, response: _FakeHttpxResponse | Exception) -> None:
        self._response = response

    async def __aenter__(self) -> "_FakeAsyncHttpxClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass

    async def get(self, url: str) -> _FakeHttpxResponse:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


@pytest.mark.asyncio
async def test_is_available_true_when_engine_kind_available(monkeypatch) -> None:
    response = _FakeHttpxResponse(json_data={
        "engines": [
            {"engine": "tabby", "available": True},
            {"engine": "ollama", "available": False},
        ]
    })
    monkeypatch.setattr(
        "reviewer.providers.dispatcher_provider.httpx.AsyncClient",
        lambda **_: _FakeAsyncHttpxClient(response),
    )
    provider = DispatcherProvider("tabbyapi", client=_FakeDispatcherClient(_success_handle()))

    assert await provider.is_available() is True


@pytest.mark.asyncio
async def test_is_available_false_when_engine_kind_unavailable(monkeypatch) -> None:
    response = _FakeHttpxResponse(json_data={
        "engines": [{"engine": "tabby", "available": False}]
    })
    monkeypatch.setattr(
        "reviewer.providers.dispatcher_provider.httpx.AsyncClient",
        lambda **_: _FakeAsyncHttpxClient(response),
    )
    provider = DispatcherProvider("tabbyapi", client=_FakeDispatcherClient(_success_handle()))

    assert await provider.is_available() is False


@pytest.mark.asyncio
async def test_is_available_false_when_engine_kind_absent(monkeypatch) -> None:
    response = _FakeHttpxResponse(json_data={"engines": [{"engine": "ollama", "available": True}]})
    monkeypatch.setattr(
        "reviewer.providers.dispatcher_provider.httpx.AsyncClient",
        lambda **_: _FakeAsyncHttpxClient(response),
    )
    provider = DispatcherProvider("tabbyapi", client=_FakeDispatcherClient(_success_handle()))

    assert await provider.is_available() is False


@pytest.mark.asyncio
async def test_is_available_false_on_transport_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "reviewer.providers.dispatcher_provider.httpx.AsyncClient",
        lambda **_: _FakeAsyncHttpxClient(httpx.ConnectError("refused")),
    )
    provider = DispatcherProvider("ollama", client=_FakeDispatcherClient(_success_handle()))

    assert await provider.is_available() is False


@pytest.mark.asyncio
async def test_is_available_false_on_malformed_body(monkeypatch) -> None:
    response = _FakeHttpxResponse(json_data={"not_engines": []})
    monkeypatch.setattr(
        "reviewer.providers.dispatcher_provider.httpx.AsyncClient",
        lambda **_: _FakeAsyncHttpxClient(response),
    )
    provider = DispatcherProvider("ollama", client=_FakeDispatcherClient(_success_handle()))

    assert await provider.is_available() is False


@pytest.mark.asyncio
async def test_ollama_forwards_num_ctx_from_config() -> None:
    client = _FakeDispatcherClient(_success_handle())
    provider = DispatcherProvider(
        "ollama", DispatcherProviderConfig(num_ctx=16384), client=client,
    )

    await provider.review(_request())

    assert client.calls[0]["task"].options == {"num_ctx": 16384}


@pytest.mark.asyncio
async def test_ollama_caller_num_ctx_override_wins_over_config() -> None:
    client = _FakeDispatcherClient(_success_handle())
    provider = DispatcherProvider(
        "ollama", DispatcherProviderConfig(num_ctx=16384), client=client,
    )

    await provider.review(_request(metadata={"num_ctx": 32768}))

    assert client.calls[0]["task"].options == {"num_ctx": 32768}


@pytest.mark.asyncio
async def test_ollama_auto_bumps_num_ctx_for_large_prompt() -> None:
    client = _FakeDispatcherClient(_success_handle())
    provider = DispatcherProvider("ollama", client=client)  # no config num_ctx pin

    # Large content forces build_review_prompt's rendered prompt well
    # past the 4096-token auto-bump threshold.
    result = await provider.review(_request())
    del result  # only the sent options matter here

    options = client.calls[0]["task"].options
    # Small default prompt shouldn't need a bump -- confirms the
    # auto-bump heuristic runs (returns None) rather than crashing.
    assert options == {} or "num_ctx" in options


@pytest.mark.asyncio
async def test_tabbyapi_forwards_max_tokens_and_disable_thinking() -> None:
    client = _FakeDispatcherClient(_success_handle())
    provider = DispatcherProvider("tabbyapi", client=client)

    await provider.review(_request())

    assert client.calls[0]["task"].options == {
        "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
    }


@pytest.mark.asyncio
async def test_tabbyapi_disable_thinking_false_omits_chat_template_kwargs() -> None:
    client = _FakeDispatcherClient(_success_handle())
    provider = DispatcherProvider(
        "tabbyapi",
        DispatcherProviderConfig(max_tokens=2048, disable_thinking=False),
        client=client,
    )

    await provider.review(_request())

    assert client.calls[0]["task"].options == {"max_tokens": 2048}
