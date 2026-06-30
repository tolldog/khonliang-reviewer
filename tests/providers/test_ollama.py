"""Tests for ``reviewer.providers.ollama``.

The provider talks to Ollama's **native** ``/api/chat`` endpoint via an
injected ``http_client`` so tests swap in a fake without monkeypatching
transport internals. The native ``/v1`` shim silently drops
``options.num_ctx`` (bug_reviewer_832a909b); these tests assert the
native payload shape that actually honors it, plus the native->OpenAI
response mapping the parsers consume.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from khonliang_reviewer import ReviewRequest
from reviewer.providers.ollama import (
    OllamaAuthError,
    OllamaHealthcheckError,
    OllamaProvider,
    OllamaProviderConfig,
)


SUCCESS_CONTENT = json.dumps(
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
)


def _native_response(
    content: Any,
    *,
    prompt_eval: Any = 120,
    eval_count: Any = 80,
    model: str = "qwen2.5-coder:14b",
) -> dict[str, Any]:
    """Build a native ``/api/chat`` (non-streaming) response body.

    Native fields differ from the OpenAI shape: the assistant text lives
    at ``message.content`` and token counts at ``prompt_eval_count`` /
    ``eval_count``. ``None`` for either count omits that key from the
    native body; ``_native_to_compat`` then surfaces it as a ``None``
    token value, which ``_build_usage`` / ``_safe_int`` coerce to 0
    (exercising the zero/absent-token path).
    """
    body: dict[str, Any] = {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
    }
    if prompt_eval is not None:
        body["prompt_eval_count"] = prompt_eval
    if eval_count is not None:
        body["eval_count"] = eval_count
    return body


NATIVE_SUCCESS = _native_response(SUCCESS_CONTENT)


class _FakeResponse:
    """Minimal stand-in for ``httpx.Response``.

    ``raise_for_status`` raises a real ``httpx.HTTPStatusError`` for any
    4xx/5xx so the provider's status-error handling runs against the
    genuine exception type.
    """

    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: Any = None,
        json_exc: BaseException | None = None,
        request_method: str = "POST",
        request_url: str = "http://localhost:11434/api/chat",
    ):
        self.status_code = status_code
        self._json_data = json_data
        self._json_exc = json_exc
        # The originating request is reflected in the raised
        # HTTPStatusError so a GET /api/tags failure isn't mislabeled as
        # a POST /api/chat one.
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
    """Async httpx-shaped fake recording the last POST and GET."""

    def __init__(
        self,
        *,
        post_response: _FakeResponse | None = None,
        post_raises: BaseException | None = None,
        get_response: _FakeResponse | None = None,
        get_raises: BaseException | None = None,
    ):
        self._post_response = (
            post_response
            if post_response is not None
            else _FakeResponse(json_data=NATIVE_SUCCESS)
        )
        self._post_raises = post_raises
        self._get_response = (
            get_response
            if get_response is not None
            else _FakeResponse(json_data={"models": []})
        )
        self._get_raises = get_raises
        self.last_post: dict[str, Any] | None = None
        self.last_get_url: str | None = None
        self.get_calls = 0

    async def post(self, url: str, *, json: Any = None, timeout: Any = None) -> Any:
        self.last_post = {"url": url, "json": json, "timeout": timeout}
        if self._post_raises is not None:
            raise self._post_raises
        return self._post_response

    async def get(self, url: str, *args: Any, **kwargs: Any) -> Any:
        self.get_calls += 1
        self.last_get_url = url
        if self._get_raises is not None:
            raise self._get_raises
        return self._get_response


def _make_http(**kwargs: Any) -> _FakeHttpClient:
    return _FakeHttpClient(**kwargs)


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


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_success_response_produces_posted_review():
    http = _make_http()
    provider = OllamaProvider(http_client=http)

    result = await provider.review(_make_request())

    assert result.disposition == "posted"
    assert result.backend == "ollama"
    assert result.model == "qwen2.5-coder:14b"
    assert result.summary == "Ollama review summary."
    assert len(result.findings) == 1
    assert result.findings[0].severity == "concern"
    assert result.findings[0].line == 42
    assert result.usage is not None
    # native prompt_eval_count / eval_count -> input/output tokens
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 80
    assert result.usage.cache_read_tokens == 0
    assert result.usage.cache_creation_tokens == 0
    assert result.usage.estimated_api_cost_usd == 0.0
    assert result.usage.repo == "tolldog/example"
    assert result.usage.pr_number == 42


async def test_posts_to_native_api_chat_endpoint():
    """The fix: requests go to the native /api/chat route (base_url's
    /v1 suffix stripped), since the /v1 shim ignores num_ctx."""
    http = _make_http()
    await OllamaProvider(http_client=http).review(_make_request())

    assert http.last_post is not None
    assert http.last_post["url"] == "http://localhost:11434/api/chat"


async def test_native_payload_shape_messages_format_stream_and_schema():
    http = _make_http()
    await OllamaProvider(http_client=http).review(_make_request())

    payload = http.last_post["json"]
    assert payload["model"] == "qwen2.5-coder:14b"
    assert payload["stream"] is False
    # JSON-constrained decoding is the effective default (native honors
    # it, unlike the old /v1 response_format).
    assert payload["format"] == "json"
    messages = payload["messages"]
    assert len(messages) == 1
    prompt = messages[0]["content"]
    assert "Review for correctness." in prompt
    assert "diff --git" in prompt
    assert '"severity"' in prompt


async def test_model_override_from_request_metadata():
    http = _make_http()
    request = _make_request(metadata={"model": "kimi-k2.5:cloud", "repo": "x/y"})

    result = await OllamaProvider(http_client=http).review(request)

    assert result.model == "kimi-k2.5:cloud"
    assert http.last_post["json"]["model"] == "kimi-k2.5:cloud"


async def test_config_default_used_when_request_has_no_model():
    http = _make_http()
    provider = OllamaProvider(
        OllamaProviderConfig(default_model="glm-4.7-flash"),
        http_client=http,
    )

    result = await provider.review(_make_request())

    assert result.model == "glm-4.7-flash"


# ---------------------------------------------------------------------------
# Error paths (httpx-shaped)
# ---------------------------------------------------------------------------


async def test_connection_error_errored():
    http = _make_http(post_raises=httpx.ConnectError("no route"))
    result = await OllamaProvider(http_client=http).review(_make_request())
    assert result.disposition == "errored"
    assert result.error_category == "backend_error"
    assert "unreachable" in result.error


async def test_timeout_error_errored():
    http = _make_http(post_raises=httpx.ReadTimeout("slow"))
    provider = OllamaProvider(
        OllamaProviderConfig(timeout_seconds=30.0),
        http_client=http,
    )
    result = await provider.review(_make_request())
    assert result.disposition == "errored"
    # Timeouts carry the dedicated ``backend_timeout`` category (distinct from
    # the generic ``backend_error``) so the sign-off path can degrade them to a
    # non-blocking "review-skipped" rather than a hard gate failure.
    assert result.error_category == "backend_timeout"
    assert "timed out" in result.error


async def test_auth_rejection_errored_auth_category():
    http = _make_http(post_response=_FakeResponse(status_code=401))
    result = await OllamaProvider(http_client=http).review(_make_request())
    assert result.disposition == "errored"
    assert result.error_category == "auth_not_provisioned"


async def test_not_found_status_errored():
    http = _make_http(post_response=_FakeResponse(status_code=404))
    result = await OllamaProvider(http_client=http).review(_make_request())
    assert result.disposition == "errored"
    assert result.error_category == "backend_error"
    assert "404" in result.error


async def test_generic_5xx_status_errored():
    http = _make_http(post_response=_FakeResponse(status_code=500))
    result = await OllamaProvider(http_client=http).review(_make_request())
    assert result.disposition == "errored"
    assert result.error_category == "backend_error"


async def test_non_json_body_errored():
    bad = _FakeResponse(json_exc=json.JSONDecodeError("bad", "doc", 0))
    http = _make_http(post_response=bad)
    result = await OllamaProvider(http_client=http).review(_make_request())
    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"


# ---------------------------------------------------------------------------
# Defensive envelope parsing (native body -> compat -> parser)
# ---------------------------------------------------------------------------


async def test_missing_message_content_errored():
    http = _make_http(post_response=_FakeResponse(json_data=_native_response(None)))
    result = await OllamaProvider(http_client=http).review(_make_request())
    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "no message content" in result.error


async def test_non_json_content_errored():
    http = _make_http(
        post_response=_FakeResponse(
            json_data=_native_response("plain text, not json")
        )
    )
    result = await OllamaProvider(http_client=http).review(_make_request())
    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "not JSON" in result.error


async def test_non_object_json_content_errored():
    http = _make_http(
        post_response=_FakeResponse(
            json_data=_native_response(json.dumps(["a", "b"]))
        )
    )
    result = await OllamaProvider(http_client=http).review(_make_request())
    assert result.disposition == "errored"
    assert result.error_category == "malformed_envelope"
    assert "not a JSON object" in result.error


async def test_section_field_parses_for_artifact_findings():
    # Artifact reviews anchor findings to a named section (fr_reviewer_19c871ab).
    content = json.dumps(
        {
            "summary": "spec review",
            "findings": [
                {
                    "severity": "concern",
                    "title": "Acceptance untestable",
                    "body": "no observable signal",
                    "section": "§Acceptance Criteria",
                },
                # non-string section is coerced to None at the boundary
                {"severity": "nit", "title": "t", "body": "b", "section": 123},
            ],
        }
    )
    http = _make_http(post_response=_FakeResponse(json_data=_native_response(content)))
    result = await OllamaProvider(http_client=http).review(_make_request())
    assert result.findings[0].section == "§Acceptance Criteria"
    assert result.findings[0].line is None
    assert result.findings[1].section is None


async def test_findings_filters_non_dict_items():
    content = json.dumps(
        {
            "summary": "mixed",
            "findings": [
                {"severity": "nit", "title": "keep", "body": "ok"},
                "rogue",
                None,
                {"severity": "comment", "title": "also keep", "body": "ok"},
            ],
        }
    )
    http = _make_http(post_response=_FakeResponse(json_data=_native_response(content)))
    result = await OllamaProvider(http_client=http).review(_make_request())
    assert result.disposition == "posted"
    assert [f.title for f in result.findings] == ["keep", "also keep"]


async def test_missing_usage_tokens_zero():
    http = _make_http(
        post_response=_FakeResponse(
            json_data=_native_response(SUCCESS_CONTENT, prompt_eval=None, eval_count=None)
        )
    )
    result = await OllamaProvider(http_client=http).review(_make_request())
    assert result.disposition == "posted"
    assert result.usage is not None
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0


async def test_string_token_values_coerced():
    http = _make_http(
        post_response=_FakeResponse(
            json_data=_native_response(SUCCESS_CONTENT, prompt_eval="100", eval_count="42.0")
        )
    )
    result = await OllamaProvider(http_client=http).review(_make_request())
    assert result.usage is not None
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 42


# ---------------------------------------------------------------------------
# healthcheck (native /api/tags)
# ---------------------------------------------------------------------------


async def test_healthcheck_success():
    http = _make_http()
    await OllamaProvider(http_client=http).healthcheck()
    assert http.get_calls == 1
    assert http.last_get_url == "http://localhost:11434/api/tags"


async def test_healthcheck_unreachable_raises():
    http = _make_http(get_raises=httpx.ConnectError("no route"))
    with pytest.raises(OllamaHealthcheckError) as excinfo:
        await OllamaProvider(http_client=http).healthcheck()
    assert "not reachable" in str(excinfo.value)
    assert not isinstance(excinfo.value, OllamaAuthError)


async def test_healthcheck_auth_rejection_raises_auth_error():
    http = _make_http(
        get_response=_FakeResponse(
            status_code=401,
            request_method="GET",
            request_url="http://localhost:11434/api/tags",
        )
    )
    with pytest.raises(OllamaAuthError) as excinfo:
        await OllamaProvider(http_client=http).healthcheck()
    assert "rejected credentials" in str(excinfo.value)


async def test_healthcheck_generic_status_falls_back_to_healthcheck_failed():
    http = _make_http(
        get_response=_FakeResponse(
            status_code=500,
            request_method="GET",
            request_url="http://localhost:11434/api/tags",
        )
    )
    with pytest.raises(OllamaHealthcheckError) as excinfo:
        await OllamaProvider(http_client=http).healthcheck()
    assert "healthcheck failed" in str(excinfo.value)
    assert not isinstance(excinfo.value, OllamaAuthError)


async def test_healthcheck_error_is_runtime_error():
    """Callers can catch RuntimeError broadly — covers both subclasses."""
    assert issubclass(OllamaHealthcheckError, RuntimeError)
    assert issubclass(OllamaAuthError, OllamaHealthcheckError)
    assert issubclass(OllamaAuthError, RuntimeError)


# ---------------------------------------------------------------------------
# Provider identity + native base-url derivation
# ---------------------------------------------------------------------------


async def test_provider_name_is_ollama():
    assert OllamaProvider.name == "ollama"
    assert OllamaProvider(http_client=_make_http()).name == "ollama"


def test_auth_headers_builds_bearer_for_real_key():
    from reviewer.providers.ollama import _auth_headers

    assert _auth_headers("secret-token") == {"Authorization": "Bearer secret-token"}
    # Default placeholder still produces a (harmless) header — parity with
    # the old openai client which sent api_key as the bearer token.
    assert _auth_headers("ollama") == {"Authorization": "Bearer ollama"}
    # Surrounding whitespace is stripped from the token, not carried into
    # the Bearer value (which would fail auth on an otherwise-valid key).
    assert _auth_headers("  secret  ") == {"Authorization": "Bearer secret"}
    # Blank / None → no header (don't send "Bearer " with nothing).
    assert _auth_headers("") == {}
    assert _auth_headers("   ") == {}
    assert _auth_headers(None) == {}


async def test_provider_threads_api_key_as_bearer_on_real_client():
    """A real httpx client carries Authorization from config.api_key so
    auth-required Ollama deployments are fixable via config (the 401/403
    paths become reachable)."""
    provider = OllamaProvider(OllamaProviderConfig(api_key="secret-token"))
    try:
        # httpx header names are case-insensitive.
        assert provider._http.headers.get("authorization") == "Bearer secret-token"
    finally:
        await provider._http.aclose()


async def test_provider_blank_api_key_sends_no_auth_header():
    provider = OllamaProvider(OllamaProviderConfig(api_key="  "))
    try:
        assert "authorization" not in provider._http.headers
    finally:
        await provider._http.aclose()


def test_native_base_url_strips_v1_suffix():
    from reviewer.providers.ollama import _native_base_url

    assert _native_base_url("http://localhost:11434/v1") == "http://localhost:11434"
    assert _native_base_url("http://localhost:11434/v1/") == "http://localhost:11434"
    # Already native — only trailing slash trimmed.
    assert _native_base_url("http://host:11434") == "http://host:11434"
    assert _native_base_url("http://host:11434/") == "http://host:11434"
    # A non-/v1 path is left intact (minus trailing slash).
    assert _native_base_url("http://host/ollama") == "http://host/ollama"
    # Surrounding whitespace (e.g. a trailing newline from env/file) is
    # stripped before the /v1 suffix check.
    assert _native_base_url("http://localhost:11434/v1\n") == "http://localhost:11434"
    assert _native_base_url("  http://localhost:11434/v1  ") == "http://localhost:11434"


async def test_custom_base_url_native_route():
    http = _make_http()
    provider = OllamaProvider(
        OllamaProviderConfig(base_url="http://gpu-box:11434/v1"),
        http_client=http,
    )
    await provider.review(_make_request())
    assert http.last_post["url"] == "http://gpu-box:11434/api/chat"


# ---------------------------------------------------------------------------
# num_ctx auto-sizing (regression for bug_reviewer_663d0d62)
# ---------------------------------------------------------------------------


def test_suggest_num_ctx_returns_none_for_small_prompt():
    """Prompts that fit Ollama's 4096 default should not override num_ctx."""
    from reviewer.providers.ollama import _suggest_num_ctx

    # Budget = ceil(bytes/2.3) + 1024 headroom + 256 template. 6000 bytes
    # → ceil(2609) + 1280 = 3889 ≤ 4096 → fits default → None.
    assert _suggest_num_ctx("x" * 6000) is None
    assert _suggest_num_ctx("hi") is None


def test_suggest_num_ctx_snug_rounds_up_to_granularity():
    """Larger prompts get a window snugly above their worst-case token
    count, rounded up to the 2048 granularity — not a coarse power-of-two
    leap. Sizing uses the bytes/2.3 worst-case floor + 1024 headroom.

    window = ceil(bytes / 2.3) + 1024 headroom + 256 template, then ceil
    to 2048. "x" is one UTF-8 byte, so bytes == char count.
    """
    from reviewer.providers.ollama import _suggest_num_ctx

    # 15_000/2.3=6522 +1024+256=7802 → ceil to 2048 → 8192
    assert _suggest_num_ctx("x" * 15_000) == 8192
    # 30_000/2.3=13_044 +1280=14_324 → 14_336
    assert _suggest_num_ctx("x" * 30_000) == 14_336
    # 80_000/2.3=34_783 +1280=36_063 → 36_864
    assert _suggest_num_ctx("x" * 80_000) == 36_864
    # 150_000/2.3=65_218 +1280=66_498 → 67_584
    assert _suggest_num_ctx("x" * 150_000) == 67_584


def test_suggest_num_ctx_covers_worst_case_density():
    """The sized window must hold a real 2.3-bytes/token tokenization of
    the prompt plus the response headroom — regression guard for codex's
    P1 (snug rounding eroding the slack the old ladder gave token-dense
    diffs). Includes codex's concrete 11_777-byte counterexample, which
    must NOT truncate."""
    import math

    from reviewer.providers.ollama import (
        _CHAT_TEMPLATE_TOKEN_OVERHEAD,
        _RESPONSE_TOKEN_HEADROOM,
        _suggest_num_ctx,
    )

    # codex's concrete case: bytes/2.3 needs 5121 + 1024 = 6145 tokens;
    # the window must be ≥ that (the 1.3x factor returned 6144 → truncate).
    assert _suggest_num_ctx("x" * 11_777) == 8192
    # codex's exact-granularity-boundary case: bytes/2.3 + 1024 == 6144
    # exactly, so without the template allowance the window was 6144 with
    # zero room for /api/chat's control tokens. Must clear the boundary.
    assert _suggest_num_ctx("x" * 11_774) == 8192

    # General property: the window holds the worst-case prompt + response
    # headroom + chat-template overhead across a range of prompt sizes.
    for n in (10_000, 11_774, 11_777, 20_000, 50_000, 123_456):
        prompt = "x" * n
        window = _suggest_num_ctx(prompt)
        budget = (
            math.ceil(len(prompt.encode("utf-8")) / 2.3)
            + _RESPONSE_TOKEN_HEADROOM
            + _CHAT_TEMPLATE_TOKEN_OVERHEAD
        )
        assert window >= budget, (n, window, budget)


def test_suggest_num_ctx_caps_at_max():
    from reviewer.providers.ollama import _suggest_num_ctx

    assert _suggest_num_ctx("x" * 1_000_000) == 131072


# ---------------------------------------------------------------------------
# num_ctx resolution order (fr_reviewer_2c751c3b)
# ---------------------------------------------------------------------------


def test_resolve_num_ctx_caller_metadata_wins_over_auto_bump():
    from reviewer.providers.ollama import OllamaProviderConfig, _resolve_num_ctx

    request = ReviewRequest(
        kind="pr_diff",
        content="x" * 200_000,
        metadata={"num_ctx": 16384},
        request_id="req-test",
    )
    cfg = OllamaProviderConfig()
    assert _resolve_num_ctx(request, cfg, "x" * 200_000) == 16384


def test_resolve_num_ctx_caller_wins_over_config():
    from reviewer.providers.ollama import OllamaProviderConfig, _resolve_num_ctx

    request = ReviewRequest(
        kind="pr_diff",
        content="x" * 1000,
        metadata={"num_ctx": 16384},
        request_id="req-test",
    )
    cfg = OllamaProviderConfig(num_ctx=8192)
    assert _resolve_num_ctx(request, cfg, "x" * 1000) == 16384


def test_resolve_num_ctx_config_default_used_when_no_caller_override():
    from reviewer.providers.ollama import OllamaProviderConfig, _resolve_num_ctx

    request = ReviewRequest(
        kind="pr_diff",
        content="x" * 1000,
        metadata={},
        request_id="req-test",
    )
    cfg = OllamaProviderConfig(num_ctx=32768)
    assert _resolve_num_ctx(request, cfg, "x" * 1000) == 32768


def test_resolve_num_ctx_falls_through_to_auto_bump():
    from reviewer.providers.ollama import OllamaProviderConfig, _resolve_num_ctx

    big_prompt = "x" * 30_000
    request = ReviewRequest(
        kind="pr_diff", content=big_prompt, metadata={}, request_id="req-test"
    )
    cfg = OllamaProviderConfig()
    # 30_000/3=10_000; *1.3=13_000 +1024=14_024 → snug-rounded to 14_336.
    assert _resolve_num_ctx(request, cfg, big_prompt) == 14_336


def test_resolve_num_ctx_falls_through_for_small_prompt():
    from reviewer.providers.ollama import OllamaProviderConfig, _resolve_num_ctx

    request = ReviewRequest(kind="pr_diff", content="hi", metadata={}, request_id="req-test")
    cfg = OllamaProviderConfig()
    assert _resolve_num_ctx(request, cfg, "hi") is None


def test_resolve_num_ctx_ignores_invalid_caller_value():
    from reviewer.providers.ollama import OllamaProviderConfig, _resolve_num_ctx

    cfg = OllamaProviderConfig(num_ctx=8192)
    big = "x" * 1000

    for bad_value in (None, "16384", 16384.5, True, -100, 0):
        request = ReviewRequest(
            kind="pr_diff",
            content=big,
            metadata={"num_ctx": bad_value},
            request_id="req-test",
        )
        assert _resolve_num_ctx(request, cfg, big) == 8192


async def test_caller_num_ctx_threads_to_native_options():
    """A caller-supplied num_ctx lands on the native payload as
    ``options.num_ctx`` — the wire-level contract that actually bites
    (unlike the dropped /v1 extra_body)."""
    http = _make_http()
    provider = OllamaProvider(http_client=http)
    await provider.review(_make_request(metadata={"num_ctx": 16384}))

    assert http.last_post["json"].get("options") == {"num_ctx": 16384}


async def test_config_num_ctx_threads_to_native_options_when_no_caller_override():
    http = _make_http()
    provider = OllamaProvider(OllamaProviderConfig(num_ctx=8192), http_client=http)
    await provider.review(_make_request(metadata={"repo": "tolldog/example"}))

    assert http.last_post["json"].get("options") == {"num_ctx": 8192}


# ---------------------------------------------------------------------------
# Format resolution (fr_reviewer_d8556085)
# ---------------------------------------------------------------------------


def test_resolve_format_caller_metadata_wins_over_config():
    from reviewer.providers.ollama import OllamaProviderConfig, _resolve_format

    cfg = OllamaProviderConfig(format="some-other-format")
    request = ReviewRequest(
        kind="pr_diff",
        content="x",
        metadata={"format": "json"},
        request_id="req-test",
    )
    assert _resolve_format(request, cfg) == "json"


def test_resolve_format_config_default_used_when_no_caller_override():
    from reviewer.providers.ollama import OllamaProviderConfig, _resolve_format

    cfg = OllamaProviderConfig(format="json")
    request = ReviewRequest(
        kind="pr_diff",
        content="x",
        metadata={},
        request_id="req-test",
    )
    assert _resolve_format(request, cfg) == "json"


def test_resolve_format_falls_through_to_none_when_neither_set():
    """``_resolve_format`` returns None when unset; the provider then
    applies ``"json"`` as the effective default (asserted separately in
    test_native_payload_shape_*)."""
    from reviewer.providers.ollama import OllamaProviderConfig, _resolve_format

    cfg = OllamaProviderConfig()
    request = ReviewRequest(
        kind="pr_diff",
        content="x",
        metadata={},
        request_id="req-test",
    )
    assert _resolve_format(request, cfg) is None


def test_resolve_format_ignores_invalid_caller_value():
    from reviewer.providers.ollama import OllamaProviderConfig, _resolve_format

    cfg = OllamaProviderConfig(format="json")

    for bad_value in (None, 1, 0, True, False, "", "   ", [], {}, 1.5):
        request = ReviewRequest(
            kind="pr_diff",
            content="x",
            metadata={"format": bad_value},
            request_id="req-test",
        )
        assert _resolve_format(request, cfg) == "json", (
            f"bad value {bad_value!r} should fall through to config"
        )


def test_resolve_format_strips_whitespace_from_valid_value():
    from reviewer.providers.ollama import OllamaProviderConfig, _resolve_format

    cfg = OllamaProviderConfig()
    request = ReviewRequest(
        kind="pr_diff",
        content="x",
        metadata={"format": "  json  "},
        request_id="req-test",
    )
    assert _resolve_format(request, cfg) == "json"


def test_resolve_model_treats_whitespace_override_as_unset():
    from reviewer.providers.ollama import _resolve_model

    # whitespace-only override → fall back to default, not a bogus model.
    req_ws = ReviewRequest(
        kind="pr_diff", content="x", metadata={"model": "   "}, request_id="r"
    )
    assert _resolve_model(req_ws, "qwen2.5-coder:14b") == "qwen2.5-coder:14b"
    # surrounding whitespace on a real value is stripped.
    req_pad = ReviewRequest(
        kind="pr_diff", content="x", metadata={"model": "  glm-4.7-flash  "}, request_id="r"
    )
    assert _resolve_model(req_pad, "qwen2.5-coder:14b") == "glm-4.7-flash"


async def test_caller_format_threads_to_native_payload():
    """A caller-supplied ``format`` lands as a top-level native field."""
    http = _make_http()
    provider = OllamaProvider(http_client=http)
    await provider.review(_make_request(metadata={"format": "custom-grammar"}))

    assert http.last_post["json"]["format"] == "custom-grammar"


async def test_config_format_threads_to_native_payload_when_no_caller_override():
    http = _make_http()
    provider = OllamaProvider(OllamaProviderConfig(format="json"), http_client=http)
    await provider.review(_make_request(metadata={"repo": "tolldog/example"}))

    assert http.last_post["json"]["format"] == "json"


async def test_format_and_num_ctx_compose_in_native_payload():
    """``options.num_ctx`` (context window) and top-level ``format``
    (grammar) are independent native fields and must compose."""
    http = _make_http()
    provider = OllamaProvider(
        OllamaProviderConfig(num_ctx=8192, format="json"),
        http_client=http,
    )
    await provider.review(_make_request(metadata={}))

    payload = http.last_post["json"]
    assert payload.get("options") == {"num_ctx": 8192}
    assert payload["format"] == "json"


async def test_review_omits_options_and_defaults_format_for_small_prompt():
    """When the prompt fits the default window, no ``options`` key is
    sent (num_ctx left at the model default); ``format`` defaults to
    ``"json"``."""
    http = _make_http()
    await OllamaProvider(http_client=http).review(_make_request())

    payload = http.last_post["json"]
    assert "options" not in payload, (
        "small prompts should not override num_ctx; default 4096 is fine"
    )
    assert payload["format"] == "json"


async def test_review_passes_num_ctx_for_large_prompt():
    """Large prompts must carry a native ``options.num_ctx`` override so
    Ollama doesn't silently truncate at 4096."""
    http = _make_http()
    big_diff = "diff --git a/f b/f\n" + ("+padding line content\n" * 1500)
    await OllamaProvider(http_client=http).review(_make_request(content=big_diff))

    payload = http.last_post["json"]
    assert "options" in payload
    num_ctx = payload["options"]["num_ctx"]
    assert num_ctx > 4096
    assert num_ctx % 2048 == 0
    assert num_ctx <= 131072


def test_suggest_num_ctx_uses_ceiling_division_at_boundary():
    """The override decision and the sizing share one budget, so the
    None/override boundary is where that single budget crosses 4096."""
    from reviewer.providers.ollama import _suggest_num_ctx

    # 6476 bytes → ceil(6476/2.3)=2816 + 1280 = 4096 → fits default → None.
    assert _suggest_num_ctx("x" * 6476) is None
    # 6477 bytes → ceil(6477/2.3)=2817 + 1280 = 4097 → override, snug 6144.
    # Regression for the decision/sizing inconsistency: this prompt used to
    # return None (typical bytes/3 fit 4096) while its worst-case budget
    # exceeded the default — a silent-truncation gap.
    assert _suggest_num_ctx("x" * 6477) == 6144


def test_suggest_num_ctx_counts_utf8_bytes_for_non_ascii():
    from reviewer.providers.ollama import _suggest_num_ctx

    # "中" is 3 UTF-8 bytes → 15_000 bytes / 3 = 5000 prompt tokens;
    # *1.3=6500 +1024=7524 → snug 8192 (same as the 15_000-"x" case).
    cjk_prompt = "中" * 5000
    assert _suggest_num_ctx(cjk_prompt) == 8192
    assert _suggest_num_ctx("x" * 5000) is None


async def test_review_warns_on_low_output_relative_to_input(caplog):
    """A high-input / low-output review should emit a truncation WARNING
    (the bug_reviewer_663d0d62 signature)."""
    http = _make_http(
        post_response=_FakeResponse(
            json_data=_native_response(
                json.dumps({"summary": "", "findings": []}),
                prompt_eval=5000,
                eval_count=8,
            )
        )
    )

    with caplog.at_level("WARNING"):
        await OllamaProvider(http_client=http).review(_make_request())

    truncation_warnings = [
        r for r in caplog.records if "truncated" in r.getMessage()
    ]
    assert truncation_warnings, (
        "low output_tokens on a large input must surface a WARNING; "
        f"all records: {[r.getMessage() for r in caplog.records]}"
    )


async def test_review_no_warning_on_clean_small_review(caplog):
    """A small input + small output is a legitimate clean review — no warning."""
    http = _make_http(
        post_response=_FakeResponse(
            json_data=_native_response(SUCCESS_CONTENT, prompt_eval=200, eval_count=8)
        )
    )

    with caplog.at_level("WARNING"):
        await OllamaProvider(http_client=http).review(_make_request())

    truncation_warnings = [
        r for r in caplog.records if "truncated" in r.getMessage()
    ]
    assert truncation_warnings == [], (
        "small-input clean review should not fire the truncation heuristic"
    )
