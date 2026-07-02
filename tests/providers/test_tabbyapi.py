"""Tests for ``reviewer.providers.tabbyapi``.

The provider talks to TabbyAPI's OpenAI-compatible
``/v1/chat/completions`` via an injected ``http_client``. Behaviors
verified live against the real engine on 2026-07-02 and pinned here:

- ``chat_template_kwargs: {"enable_thinking": false}`` is the switch
  that suppresses the Qwen3 ``<think>`` preamble; the parser still
  defensively strips a leading think-block AND a whole-body markdown
  fence (``response_format`` is accepted by the server but does not
  grammar-constrain decoding).
- ``"usage": null`` on non-streaming completions — the usage record
  tolerates it with zero token counts.
- No retry on timeout: the model is resident (no cold start), and a
  retry would compound past the caller's degrade-to-skip budget
  (dog_d6895752).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from khonliang_reviewer import ReviewRequest
from reviewer.providers.tabbyapi import (
    TabbyAPIAuthError,
    TabbyAPIHealthcheckError,
    TabbyAPIProvider,
    TabbyAPIProviderConfig,
    _clean_content,
)


SUCCESS_PAYLOAD = {
    "summary": "Tabby review summary.",
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


def _openai_response(
    content: str,
    *,
    usage: Any = None,
    model: str = "Qwen3-14B-exl3-6bpw",
) -> dict[str, Any]:
    """OpenAI chat-completions body as the live TabbyAPI build shapes it.

    ``usage`` defaults to ``None`` deliberately — that is what the real
    server returns on non-streaming completions (verified live), so the
    default-path tests exercise the null-usage tolerance.
    """
    return {
        "id": "cmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": usage,
    }


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: Any = None,
        json_exc: BaseException | None = None,
        request_method: str = "POST",
        request_url: str = "http://localhost:5000/v1/chat/completions",
    ):
        self.status_code = status_code
        self._json_data = json_data
        self._json_exc = json_exc
        self._request_method = request_method
        self._request_url = request_url

    def json(self) -> Any:
        if self._json_exc is not None:
            raise self._json_exc
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request(self._request_method, self._request_url)
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=request, response=response
            )


class _FakeHttpClient:
    """Async httpx-shaped fake recording POST payloads and counting calls."""

    def __init__(
        self,
        *,
        post_response: _FakeResponse | None = None,
        post_raises: BaseException | None = None,
        get_response: _FakeResponse | None = None,
        get_raises: BaseException | None = None,
    ):
        self._post_response = post_response
        self._post_raises = post_raises
        self._get_response = get_response
        self._get_raises = get_raises
        self.post_calls = 0
        self.last_post_url: str | None = None
        self.last_post_json: dict[str, Any] | None = None

    async def post(self, url: str, *, json: Any = None, timeout: Any = None):
        self.post_calls += 1
        self.last_post_url = url
        self.last_post_json = json
        if self._post_raises is not None:
            raise self._post_raises
        return self._post_response or _FakeResponse(json_data={})

    async def get(self, url: str, *, timeout: Any = None):
        if self._get_raises is not None:
            raise self._get_raises
        return self._get_response or _FakeResponse(
            json_data={}, request_method="GET", request_url=url
        )


def _request(**metadata: Any) -> ReviewRequest:
    return ReviewRequest(
        kind="pr_diff",
        content="diff --git a/x b/x\n+code\n",
        metadata=metadata,
        request_id="req-tabby-test",
    )


def _provider(client: _FakeHttpClient, **config: Any) -> TabbyAPIProvider:
    cfg = TabbyAPIProviderConfig(
        api_key="test-key", default_model="Qwen3-14B-exl3-6bpw", **config
    )
    return TabbyAPIProvider(cfg, http_client=client)


# ---------------------------------------------------------------------------
# Success path + payload shape
# ---------------------------------------------------------------------------


async def test_success_parses_findings_and_null_usage():
    client = _FakeHttpClient(
        post_response=_FakeResponse(json_data=_openai_response(SUCCESS_CONTENT))
    )
    result = await _provider(client).review(_request())

    assert result.disposition == "posted"
    assert result.backend == "tabbyapi"
    assert result.model == "Qwen3-14B-exl3-6bpw"
    assert result.summary == "Tabby review summary."
    assert len(result.findings) == 1
    assert result.findings[0].severity == "concern"
    assert result.findings[0].line == 42
    # Live server returns usage=null; tolerated with honest zeros.
    assert result.usage is not None
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0
    assert result.usage.estimated_api_cost_usd == 0.0


async def test_usage_populated_when_server_reports_it():
    client = _FakeHttpClient(
        post_response=_FakeResponse(
            json_data=_openai_response(
                SUCCESS_CONTENT,
                usage={"prompt_tokens": 321, "completion_tokens": 55},
            )
        )
    )
    result = await _provider(client).review(_request())
    assert result.usage.input_tokens == 321
    assert result.usage.output_tokens == 55


async def test_payload_shape_disables_thinking_and_caps_tokens():
    client = _FakeHttpClient(
        post_response=_FakeResponse(json_data=_openai_response(SUCCESS_CONTENT))
    )
    await _provider(client).review(_request())

    assert client.last_post_url == "http://localhost:5000/v1/chat/completions"
    payload = client.last_post_json
    assert payload["model"] == "Qwen3-14B-exl3-6bpw"
    assert payload["stream"] is False
    assert payload["max_tokens"] == 4096
    # The verified-live switch for the Qwen3 <think> preamble.
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


async def test_disable_thinking_config_off_omits_template_kwargs():
    client = _FakeHttpClient(
        post_response=_FakeResponse(json_data=_openai_response(SUCCESS_CONTENT))
    )
    await _provider(client, disable_thinking=False).review(_request())
    assert "chat_template_kwargs" not in client.last_post_json


async def test_model_override_via_metadata():
    client = _FakeHttpClient(
        post_response=_FakeResponse(json_data=_openai_response(SUCCESS_CONTENT))
    )
    result = await _provider(client).review(_request(model="Other-Model"))
    assert client.last_post_json["model"] == "Other-Model"
    assert result.model == "Other-Model"


# ---------------------------------------------------------------------------
# Thinking-model / fenced-output hygiene
# ---------------------------------------------------------------------------


async def test_leading_think_block_is_stripped():
    content = f"<think>\nreasoning about the diff...\n</think>\n\n{SUCCESS_CONTENT}"
    client = _FakeHttpClient(
        post_response=_FakeResponse(json_data=_openai_response(content))
    )
    result = await _provider(client).review(_request())
    assert result.disposition == "posted"
    assert result.summary == "Tabby review summary."


async def test_fenced_json_is_unwrapped():
    content = f"```json\n{SUCCESS_CONTENT}\n```"
    client = _FakeHttpClient(
        post_response=_FakeResponse(json_data=_openai_response(content))
    )
    result = await _provider(client).review(_request())
    assert result.disposition == "posted"
    assert len(result.findings) == 1


async def test_empty_think_block_then_fence_both_stripped():
    # The exact shape /no_think produces live: an EMPTY think block, and
    # (with response_format set) a fenced body after it.
    content = f"<think>\n\n</think>\n\n```json\n{SUCCESS_CONTENT}\n```"
    client = _FakeHttpClient(
        post_response=_FakeResponse(json_data=_openai_response(content))
    )
    result = await _provider(client).review(_request())
    assert result.disposition == "posted"


def test_clean_content_preserves_think_mention_inside_body():
    # Only a LEADING think-block is stripped; a finding whose body quotes
    # "<think>" text mid-payload must survive byte-identical.
    payload = json.dumps({"summary": "mentions <think> in prose", "findings": []})
    assert _clean_content(payload) == payload


# ---------------------------------------------------------------------------
# Error dispositions
# ---------------------------------------------------------------------------


async def test_timeout_is_backend_timeout_with_no_retry():
    client = _FakeHttpClient(post_raises=httpx.ConnectTimeout("slow"))
    result = await _provider(client).review(_request())
    assert result.disposition == "errored"
    assert result.error_category == "backend_timeout"
    # ONE attempt only — no cold-start retry on a resident engine
    # (dog_d6895752: retries compound past the degrade-to-skip budget).
    assert client.post_calls == 1


async def test_auth_rejection_is_auth_not_provisioned():
    client = _FakeHttpClient(post_response=_FakeResponse(status_code=401))
    result = await _provider(client).review(_request())
    assert result.disposition == "errored"
    assert result.error_category == "auth_not_provisioned"


async def test_http_500_is_backend_error():
    client = _FakeHttpClient(post_response=_FakeResponse(status_code=500))
    result = await _provider(client).review(_request())
    assert result.error_category == "backend_error"


async def test_unreachable_is_backend_error():
    client = _FakeHttpClient(post_raises=httpx.ConnectError("refused"))
    result = await _provider(client).review(_request())
    assert result.error_category == "backend_error"


async def test_non_json_content_is_malformed_envelope():
    client = _FakeHttpClient(
        post_response=_FakeResponse(
            json_data=_openai_response("I am not JSON at all.")
        )
    )
    result = await _provider(client).review(_request())
    assert result.error_category == "malformed_envelope"


async def test_missing_content_is_malformed_envelope():
    client = _FakeHttpClient(
        post_response=_FakeResponse(json_data={"choices": []})
    )
    result = await _provider(client).review(_request())
    assert result.error_category == "malformed_envelope"


# ---------------------------------------------------------------------------
# Binary-questions parity
# ---------------------------------------------------------------------------


def _full_verdicts() -> list[dict[str, Any]]:
    from reviewer.providers._prompt import BINARY_QUESTION_DIMENSIONS

    return [
        {
            "dimension": dimension,
            "question": question,
            "answer": True,
            "explanation": "grounded",
        }
        for dimension, question in BINARY_QUESTION_DIMENSIONS
    ]


async def test_binary_questions_verdicts_parsed_with_full_coverage():
    payload = dict(SUCCESS_PAYLOAD, verdicts=_full_verdicts())
    client = _FakeHttpClient(
        post_response=_FakeResponse(json_data=_openai_response(json.dumps(payload)))
    )
    result = await _provider(client).review(
        _request(_khonliang_binary_questions=True)
    )
    assert result.disposition == "posted"
    assert len(result.verdicts) == 6


async def test_binary_questions_incomplete_coverage_is_malformed_envelope():
    payload = dict(SUCCESS_PAYLOAD, verdicts=_full_verdicts()[:3])
    client = _FakeHttpClient(
        post_response=_FakeResponse(json_data=_openai_response(json.dumps(payload)))
    )
    result = await _provider(client).review(
        _request(_khonliang_binary_questions=True)
    )
    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "binary-questions contract" in result.error


async def test_unsolicited_verdicts_dropped_on_holistic_review():
    payload = dict(SUCCESS_PAYLOAD, verdicts=_full_verdicts())
    client = _FakeHttpClient(
        post_response=_FakeResponse(json_data=_openai_response(json.dumps(payload)))
    )
    result = await _provider(client).review(_request())
    assert result.disposition == "posted"
    assert result.verdicts == []


# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------


async def test_healthcheck_passes_on_200():
    client = _FakeHttpClient(
        get_response=_FakeResponse(json_data={"data": []}, request_method="GET")
    )
    await _provider(client).healthcheck()  # no raise


async def test_healthcheck_auth_error_on_401():
    client = _FakeHttpClient(
        get_response=_FakeResponse(status_code=401, request_method="GET")
    )
    with pytest.raises(TabbyAPIAuthError):
        await _provider(client).healthcheck()


async def test_healthcheck_unreachable_raises_base_error():
    client = _FakeHttpClient(get_raises=httpx.ConnectError("refused"))
    with pytest.raises(TabbyAPIHealthcheckError):
        await _provider(client).healthcheck()
