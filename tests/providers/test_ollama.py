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
    OllamaAuthError,
    OllamaHealthcheckError,
    OllamaProvider,
    OllamaProviderConfig,
)


SUCCESS_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1776660000,
    "model": "qwen2.5-coder:14b",
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
        return {"data": [{"id": "qwen2.5-coder:14b"}]}


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
    assert result.model == "qwen2.5-coder:14b"
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
    assert call["model"] == "qwen2.5-coder:14b"
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
        model="qwen2.5-coder:14b",
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
    # reachability failure is NOT an auth error
    assert not isinstance(excinfo.value, OllamaAuthError)


async def test_healthcheck_auth_rejection_raises_auth_error():
    client = _make_client(models_list_raises=_api_error(openai.AuthenticationError))
    with pytest.raises(OllamaAuthError) as excinfo:
        await OllamaProvider(client=client).healthcheck()
    assert "rejected credentials" in str(excinfo.value)


async def test_healthcheck_generic_api_error_falls_back_to_healthcheck_failed():
    client = _make_client(models_list_raises=_api_error(openai.APIError))
    with pytest.raises(OllamaHealthcheckError) as excinfo:
        await OllamaProvider(client=client).healthcheck()
    assert "healthcheck failed" in str(excinfo.value)
    assert not isinstance(excinfo.value, OllamaAuthError)


async def test_healthcheck_error_is_runtime_error():
    """Callers can catch RuntimeError broadly — covers both subclasses."""
    assert issubclass(OllamaHealthcheckError, RuntimeError)
    assert issubclass(OllamaAuthError, OllamaHealthcheckError)
    assert issubclass(OllamaAuthError, RuntimeError)


# ---------------------------------------------------------------------------
# Provider identity
# ---------------------------------------------------------------------------


async def test_provider_name_is_ollama():
    assert OllamaProvider.name == "ollama"
    assert OllamaProvider(client=_make_client()).name == "ollama"


# ---------------------------------------------------------------------------
# num_ctx auto-sizing (regression for bug_reviewer_663d0d62)
# ---------------------------------------------------------------------------


def test_suggest_num_ctx_returns_none_for_small_prompt():
    """Prompts that fit Ollama's 4096 default should not override num_ctx.

    Forcing a non-default ``num_ctx`` on a short prompt would risk
    rejection from local models that don't accept smaller-than-default
    windows, and the override gains nothing.
    """
    from reviewer.providers.ollama import _suggest_num_ctx

    # Default headroom (1024) + len/3 must stay under 4096 to skip override.
    # 9000 chars = 3000 estimated tokens + 1024 headroom = 4024 → still under default.
    assert _suggest_num_ctx("x" * 9000) is None
    # And a tiny prompt obviously fits.
    assert _suggest_num_ctx("hi") is None


def test_suggest_num_ctx_steps_up_through_ladder():
    """Larger prompts should land on standard llama.cpp window sizes."""
    from reviewer.providers.ollama import _suggest_num_ctx

    # ~15000 chars / 3 = 5000 + 1024 = 6024 → first ladder step is 8192.
    assert _suggest_num_ctx("x" * 15_000) == 8192
    # ~30000 chars / 3 = 10000 + 1024 = 11024 → 16384.
    assert _suggest_num_ctx("x" * 30_000) == 16384
    # ~80000 chars / 3 = 26666 + 1024 = 27690 → 32768.
    assert _suggest_num_ctx("x" * 80_000) == 32768
    # ~150000 chars / 3 = 50000 + 1024 = 51024 → 65536.
    assert _suggest_num_ctx("x" * 150_000) == 65536


def test_suggest_num_ctx_caps_at_largest_ladder_step():
    """Beyond 131072 we cap; review should still go through and the
    truncation warning will fire if the model can't actually consume it."""
    from reviewer.providers.ollama import _suggest_num_ctx

    # 1MB of x's: estimated tokens ~= 333334 + 1024 = 334358 → exceeds
    # all ladder entries; cap at top.
    assert _suggest_num_ctx("x" * 1_000_000) == 131072


def test_suggest_num_ctx_uses_ceiling_division_at_boundary():
    """Ceiling division must push borderline prompts over the override
    threshold rather than rounding down and silently truncating.

    With floor division, len=9217 ASCII bytes would estimate 3072
    tokens (9217 // 3 == 3072 because 3*3072 == 9216 < 9217), and
    3072+1024=4096 ≤ 4096 → no override → truncation risk on a prompt
    that's actually one byte over the line. With math.ceil the same
    length produces 3073, and 3073+1024=4097 > 4096 → first ladder
    step (8192). Locks the conservative-bias property the docstring
    promises.
    """
    from reviewer.providers.ollama import _suggest_num_ctx

    # Just at the boundary: 9216 ASCII bytes / 3 = 3072 tokens exactly,
    # plus 1024 = 4096 → no override (still inside default).
    assert _suggest_num_ctx("x" * 9216) is None
    # One byte over: ceil pushes to 3073 tokens + 1024 = 4097 → first
    # ladder step.
    assert _suggest_num_ctx("x" * 9217) == 8192


def test_suggest_num_ctx_counts_utf8_bytes_for_non_ascii():
    """Multi-byte scripts must be measured in UTF-8 bytes, not chars.

    A CJK character is 3 bytes in UTF-8 and tokenizes to roughly one
    token in most modern tokenizers, so ``len(prompt)`` (character
    count) would dramatically underestimate the token count and let a
    truncation-prone prompt slip through with no override. Byte
    counting keeps the estimator uniform across scripts.
    """
    from reviewer.providers.ollama import _suggest_num_ctx

    # 5000 CJK characters = 15000 UTF-8 bytes ≈ 5000 tokens estimated
    # (15000 / 3) + 1024 = 6024 → first ladder step (8192).
    cjk_prompt = "中" * 5000
    assert _suggest_num_ctx(cjk_prompt) == 8192
    # Same character count in ASCII (5000 chars = 5000 bytes ≈ 1667
    # tokens + 1024 headroom = 2691 tokens estimated) fits in default
    # → no override. Confirms the byte-vs-char distinction matters at
    # the relevant size band.
    assert _suggest_num_ctx("x" * 5000) is None


async def test_review_omits_extra_body_for_small_prompt():
    """When the prompt fits the default window the SDK call carries no extra_body."""
    client = _make_client(response=SUCCESS_RESPONSE)
    await OllamaProvider(client=client).review(_make_request())

    last = client._create.last_call
    assert last is not None
    assert "extra_body" not in last, (
        "small prompts should not override num_ctx; default 4096 is fine"
    )


async def test_review_passes_num_ctx_for_large_prompt():
    """Large prompts should land a num_ctx override under extra_body.options."""
    client = _make_client(response=SUCCESS_RESPONSE)

    # ~30KB of diff content overflows the default 4096-token window.
    big_diff = "diff --git a/f b/f\n" + ("+padding line content\n" * 1500)
    await OllamaProvider(client=client).review(
        _make_request(content=big_diff)
    )

    last = client._create.last_call
    assert last is not None
    assert "extra_body" in last, (
        "large prompts must override num_ctx so Ollama doesn't silently truncate"
    )
    assert "options" in last["extra_body"]
    num_ctx = last["extra_body"]["options"]["num_ctx"]
    # Exact step depends on the prompt-builder overhead; assert it's
    # strictly above the default and on the ladder.
    assert num_ctx > 4096
    assert num_ctx in {8192, 16384, 32768, 65536, 131072}


async def test_review_warns_on_low_output_relative_to_input(caplog):
    """A high-input / low-output review should emit a truncation WARNING.

    Reproduces the bug_reviewer_663d0d62 signature: review returned
    ``output_tokens=8`` with ``disposition=posted`` on a large input.
    The warning gives operators a chance to notice the silent
    truncation even when the rule-table / num_ctx-auto-size didn't
    catch it.
    """
    response = dict(SUCCESS_RESPONSE)
    response["usage"] = {
        "prompt_tokens": 5000,  # large input
        "completion_tokens": 8,  # tiny output — truncation signature
        "total_tokens": 5008,
    }
    # Replace message content with empty findings so the result still
    # round-trips cleanly; the warning is independent of finding shape.
    response["choices"] = [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": json.dumps({"summary": "", "findings": []}),
            },
            "finish_reason": "stop",
        }
    ]

    client = _make_client(response=response)

    with caplog.at_level("WARNING"):
        await OllamaProvider(client=client).review(_make_request())

    truncation_warnings = [
        r for r in caplog.records if "truncated" in r.getMessage()
    ]
    assert truncation_warnings, (
        "low output_tokens on a large input must surface a WARNING; "
        f"all records: {[r.getMessage() for r in caplog.records]}"
    )


async def test_review_no_warning_on_clean_small_review(caplog):
    """A small input + small output is a legitimate clean review — no warning."""
    response = dict(SUCCESS_RESPONSE)
    response["usage"] = {
        "prompt_tokens": 200,
        "completion_tokens": 8,
        "total_tokens": 208,
    }
    client = _make_client(response=response)

    with caplog.at_level("WARNING"):
        await OllamaProvider(client=client).review(_make_request())

    truncation_warnings = [
        r for r in caplog.records if "truncated" in r.getMessage()
    ]
    assert truncation_warnings == [], (
        "small-input clean review should not fire the truncation heuristic"
    )
