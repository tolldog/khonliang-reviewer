"""Tests for ``SelectorConfig.default_models`` per-backend lookup.

Pins the resolution order ``ProviderSelector.select`` follows when
``model is None``:

1. Non-empty ``default_models[chosen_backend]`` wins.
2. Empty-string entry in ``default_models`` is treated as "unset at
   the selector layer" and falls through (does NOT short-circuit
   selection with the empty string).
3. Legacy ``default_model`` is consulted only when the chosen
   backend matches ``default_backend``.
4. Otherwise the empty-string sentinel ("provider applies its own
   default") is returned.

Caller-supplied non-empty ``model`` always wins over the table.
Caller-supplied ``model=""`` is preserved (explicit "let provider
decide") and not coerced to a config default.
"""

from __future__ import annotations

import pytest
from khonliang_reviewer import (
    ReviewProvider,
    ReviewRequest,
    ReviewResult,
    UsageEvent,
)

from reviewer.selector import (
    ProviderSelector,
    SelectorConfig,
    UnknownBackendError,
)


class _FakeProvider(ReviewProvider):
    def __init__(self, name: str):
        self.name = name

    async def review(self, request: ReviewRequest) -> ReviewResult:  # pragma: no cover
        return ReviewResult(
            request_id=request.request_id,
            summary="",
            findings=[],
            usage=UsageEvent(
                timestamp=0.0,
                backend=self.name,
                model="",
                input_tokens=0,
                output_tokens=0,
                duration_ms=0,
            ),
        )


def _selector(*, default_models: dict[str, str], default_backend: str = "ollama",
              default_model: str = "qwen2.5-coder:14b") -> ProviderSelector:
    return ProviderSelector(
        {
            "ollama": _FakeProvider("ollama"),
            "claude_cli": _FakeProvider("claude_cli"),
            "codex_cli": _FakeProvider("codex_cli"),
            "gh_copilot": _FakeProvider("gh_copilot"),
        },
        SelectorConfig(
            default_backend=default_backend,
            default_model=default_model,
            default_models=default_models,
        ),
    )


def test_default_models_dict_wins_for_explicit_backend():
    """Non-empty ``default_models[chosen_backend]`` is the per-backend
    default — wins over the legacy global ``default_model`` even when
    the chosen backend differs from ``default_backend``.
    """
    sel = _selector(
        default_models={"claude_cli": "sonnet", "ollama": "qwen2.5-coder:14b"},
    )
    _, model = sel.select(backend="claude_cli")
    assert model == "sonnet"


def test_empty_string_in_default_models_falls_through_not_short_circuit():
    """An empty-string value in ``default_models`` is "unset at the
    selector layer" — the lookup falls through to the legacy default
    or the empty-string sentinel rather than emitting ``--model ""``
    to the binary.

    Concretely: with ``default_models={"codex_cli": ""}`` and a caller
    asking for codex_cli without a model, selection should land on
    the empty-string sentinel (provider applies its own default),
    not on the literal empty string read from the dict.
    """
    sel = _selector(
        default_models={"codex_cli": "", "ollama": "qwen2.5-coder:14b"},
    )
    _, model = sel.select(backend="codex_cli")
    # codex_cli != default_backend (ollama), no per-backend default
    # → empty-string sentinel.
    assert model == ""


def test_legacy_default_model_still_works_paired_with_default_backend():
    """Legacy single-string ``default_model`` keeps working — when the
    chosen backend matches ``default_backend`` and ``default_models``
    has nothing for it, the legacy field is consulted.
    """
    sel = _selector(
        default_models={},  # nothing per-backend
        default_backend="ollama",
        default_model="qwen2.5-coder:14b",
    )
    _, model = sel.select(backend="ollama")
    assert model == "qwen2.5-coder:14b"


def test_legacy_default_model_not_applied_to_non_default_backend():
    """The legacy ``default_model`` is paired with ``default_backend``;
    applying it to a different backend would send (e.g.) an Ollama
    model id into the Claude binary. Empty-string sentinel returned
    instead.

    This is the failure mode the FR is closing for the cross-backend
    path; the only failure mode that REMAINS without a per-backend
    dict is the *default-backend* path covered in
    ``test_per_backend_dict_closes_default_backend_misconfiguration``.
    """
    sel = _selector(
        default_models={},
        default_backend="ollama",
        default_model="qwen2.5-coder:14b",
    )
    _, model = sel.select(backend="claude_cli")
    assert model == ""


def test_per_backend_dict_closes_default_backend_misconfiguration():
    """The actual remaining bug per MS-D rev6: when
    ``default_provider="claude_cli"`` AND ``default_model`` still
    carries the bundled Ollama-shaped id, a request paired with the
    default backend would pipe the wrong model into Claude. With a
    per-backend dict overriding it, the right Claude-shaped model
    surfaces.
    """
    sel = _selector(
        default_models={"claude_cli": "sonnet"},
        default_backend="claude_cli",
        default_model="qwen2.5-coder:14b",  # legacy global default, Ollama-shaped
    )
    _, model = sel.select(backend="claude_cli")
    # Without the per-backend dict this would resolve to
    # "qwen2.5-coder:14b" via the legacy path. With it, "sonnet".
    assert model == "sonnet"


def test_caller_supplied_model_always_wins():
    """Caller-supplied non-empty ``model`` overrides everything —
    rule-table-supplied model + per-backend default + legacy default.
    """
    sel = _selector(
        default_models={"claude_cli": "sonnet"},
        default_backend="ollama",
        default_model="qwen2.5-coder:14b",
    )
    _, model = sel.select(backend="claude_cli", model="opus")
    assert model == "opus"


def test_caller_supplied_empty_model_preserved_not_coerced():
    """``model=""`` is the explicit "let the provider apply its own
    default" sentinel from the existing contract. The new
    ``default_models`` dict must NOT silently override it back to a
    config default — otherwise a caller asking for "no model" would
    inherit an unrelated dict entry.
    """
    sel = _selector(
        default_models={"claude_cli": "sonnet"},
        default_backend="claude_cli",
    )
    _, model = sel.select(backend="claude_cli", model="")
    assert model == ""


def test_unknown_backend_raises():
    """Unknown backends still raise — the new dict doesn't silently
    register backends just because they appear in ``default_models``.
    """
    sel = _selector(default_models={"phantom_backend": "x"})
    with pytest.raises(UnknownBackendError):
        sel.select(backend="phantom_backend")


def test_default_models_field_default_is_empty_dict():
    """``SelectorConfig()`` constructs with an empty per-backend dict
    so existing tests / configs that only set ``default_model`` keep
    working without changes (backwards compat).
    """
    cfg = SelectorConfig()
    assert cfg.default_models == {}
    assert cfg.default_model == "qwen2.5-coder:14b"
    assert cfg.default_backend == "ollama"


def test_no_caller_backend_uses_default_backend_with_per_backend_dict():
    """When the caller supplies neither backend nor model, the chosen
    backend is ``default_backend`` and the model comes from the
    per-backend dict if available.
    """
    sel = _selector(
        default_models={"ollama": "custom-ollama-model"},
        default_backend="ollama",
    )
    _, model = sel.select()
    assert model == "custom-ollama-model"
