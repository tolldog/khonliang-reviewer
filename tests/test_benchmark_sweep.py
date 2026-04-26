"""Tests for ``reviewer.tools.benchmark_sweep``.

The harness runs every registered (backend, model) pair through a
real :class:`ReviewProvider`. Tests inject a fake provider per
backend so no real subprocess / HTTP traffic happens — the harness
itself (filtering, artifact writing, summary rendering, error
handling) is what's under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from khonliang_reviewer import (
    ReviewFinding,
    ReviewProvider,
    ReviewRequest,
    ReviewResult,
    UsageEvent,
)

from reviewer.registry import ProviderRegistry
from reviewer.tools import benchmark_sweep


# ---------------------------------------------------------------------------
# Fake provider
# ---------------------------------------------------------------------------


class _FakeProvider(ReviewProvider):
    """Provider that records the request it saw and returns a canned result."""

    def __init__(
        self,
        backend: str,
        *,
        result: ReviewResult | None = None,
        raises: BaseException | None = None,
    ):
        self.name = backend
        self._result = result
        self._raises = raises
        self.last_request: ReviewRequest | None = None
        self.call_count = 0

    async def review(self, request: ReviewRequest) -> ReviewResult:
        self.last_request = request
        self.call_count += 1
        if self._raises is not None:
            raise self._raises
        assert self._result is not None
        return self._result


def _make_result(
    *,
    backend: str = "fake",
    model: str = "fake-model",
    findings: list[ReviewFinding] | None = None,
    summary: str = "ok",
    disposition: str = "posted",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> ReviewResult:
    return ReviewResult(
        request_id="req-1",
        summary=summary,
        findings=findings or [],
        disposition=disposition,  # type: ignore[arg-type]
        backend=backend,
        model=model,
        usage=UsageEvent(
            timestamp=1.0,
            backend=backend,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=42,
        ),
    )


# ---------------------------------------------------------------------------
# load_diff
# ---------------------------------------------------------------------------


def test_load_diff_defaults_to_bundled_reference():
    diff, label = benchmark_sweep.load_diff(None)
    assert diff.startswith("diff "), "bundled diff should start with a `diff` header"
    assert "bus_lib_pr14.diff" in label


def test_load_diff_accepts_path(tmp_path):
    p = tmp_path / "tiny.diff"
    p.write_text("diff --git a/x b/x\n@@\n-old\n+new\n")
    diff, label = benchmark_sweep.load_diff(str(p))
    assert "old" in diff and "new" in diff
    assert label.startswith("path:")


def test_load_diff_rejects_unresolvable_source():
    with pytest.raises(RuntimeError) as excinfo:
        benchmark_sweep.load_diff("/this/path/does/not/exist")
    assert "did not resolve" in str(excinfo.value)


def test_fetch_pr_diff_timeout_surfaces_runtime_error(monkeypatch):
    """``subprocess.TimeoutExpired`` from ``gh pr diff`` becomes a
    friendly :class:`RuntimeError` so the harness's documented
    contract — "fetch errors as a friendly RuntimeError" — covers
    the slow-network case, not just the missing-binary case.
    """
    import subprocess

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=60)

    monkeypatch.setattr(benchmark_sweep.subprocess, "run", _raise_timeout)

    with pytest.raises(RuntimeError) as excinfo:
        benchmark_sweep._fetch_pr_diff("owner/repo", 42)

    msg = str(excinfo.value)
    assert "timed out" in msg
    assert "--diff" in msg


def test_load_diff_bundled_missing_raises_runtime_error_with_hint(monkeypatch):
    """Catch resource-load failure on the bundled-diff path, surface a hint.

    Otherwise an install missing the ``benchmark_data/`` payload
    would raise a raw ``FileNotFoundError`` from
    ``importlib.resources.read_text`` — confusing for the operator
    who can't see why the harness failed without the package_data
    context.
    """
    monkeypatch.setattr(
        benchmark_sweep, "_DEFAULT_DIFF_NAME", "definitely-not-bundled.diff"
    )

    with pytest.raises(RuntimeError) as excinfo:
        benchmark_sweep.load_diff(None)

    msg = str(excinfo.value)
    assert "unreadable" in msg
    assert "package-data" in msg
    assert "--diff" in msg


# ---------------------------------------------------------------------------
# _safe_artifact_name
# ---------------------------------------------------------------------------


def test_safe_artifact_name_strips_problematic_chars():
    name = benchmark_sweep._safe_artifact_name(
        "ollama", "kimi-k2.5:cloud", "result.json"
    )
    # Colons / brackets are filesystem-fragile across OSes.
    assert ":" not in name
    assert "[" not in name and "]" not in name
    # Backend + model + suffix all surface so artifacts stay
    # debuggable from the file name alone.
    assert "ollama" in name
    assert "kimi-k2.5" in name
    # Suffix preserved as the final extension (the disambiguating
    # hash slots between the stem and the extension so the file is
    # still recognized as JSON by tools that switch on suffix).
    assert name.endswith(".json")


def test_safe_artifact_name_disambiguates_sanitization_collisions():
    """Two raw model ids that sanitize to the same string must still
    produce distinct filenames — otherwise ``summary.jsonl`` would
    point at an artifact silently overwritten by a sibling row.
    """
    a = benchmark_sweep._safe_artifact_name(
        "ollama", "kimi-k2.5:cloud", "result.json"
    )
    b = benchmark_sweep._safe_artifact_name(
        "ollama", "kimi-k2/5/cloud", "result.json"
    )
    assert a != b


def test_safe_artifact_name_handles_empty_model():
    name = benchmark_sweep._safe_artifact_name("claude_cli", "", "result.json")
    # Empty model becomes ``default`` so two backend rows both with
    # an empty model don't collide on disk.
    assert "default" in name


# ---------------------------------------------------------------------------
# Sweep end-to-end
# ---------------------------------------------------------------------------


async def test_run_writes_summary_and_artifacts(tmp_path):
    """End-to-end: registry with two fakes → summary.jsonl + REPORT.md +
    per-row artifacts."""
    fake_a = _FakeProvider(
        "backend_a",
        result=_make_result(
            backend="backend_a",
            model="model-a",
            findings=[
                ReviewFinding(severity="concern", title="C", body="b"),
                ReviewFinding(severity="nit", title="N", body="b"),
            ],
        ),
    )
    fake_b = _FakeProvider(
        "backend_b",
        result=_make_result(backend="backend_b", model="model-b"),
    )
    registry = ProviderRegistry()
    registry.register(fake_a, default_model="model-a", declared_models=["model-a"])
    registry.register(fake_b, default_model="model-b", declared_models=["model-b"])

    out_dir = tmp_path / "sweep"
    jsonl_path, report_path, rows = await benchmark_sweep.run(
        diff_source=None,
        output_dir=out_dir,
        backends=[],
        models=[],
        kind="pr_diff",
        instructions="test",
        registry=registry,
    )

    assert jsonl_path.exists()
    assert report_path.exists()
    # Rows are in registration order.
    assert [r.backend for r in rows] == ["backend_a", "backend_b"]
    # Per-row artifacts written.
    artifacts_dir = out_dir / "artifacts"
    assert artifacts_dir.is_dir()
    artifact_files = list(artifacts_dir.glob("*.json"))
    assert len(artifact_files) == 2

    # summary.jsonl is one record per row.
    contents = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 2
    parsed = [json.loads(line) for line in contents]
    assert parsed[0]["backend"] == "backend_a"
    assert parsed[0]["finding_count_concern"] == 1
    assert parsed[0]["finding_count_nit"] == 1
    assert parsed[1]["backend"] == "backend_b"

    # REPORT.md surfaces both backends.
    md = report_path.read_text(encoding="utf-8")
    assert "| backend_a |" in md
    assert "| backend_b |" in md

    # DIFF_SOURCE label written + diff payload preserved.
    assert (out_dir / "DIFF_SOURCE.txt").read_text().startswith("bundled:")
    assert (out_dir / "diff.patch").read_text(encoding="utf-8").startswith("diff ")


async def test_run_filters_by_backend(tmp_path):
    fake_a = _FakeProvider(
        "backend_a",
        result=_make_result(backend="backend_a", model="x"),
    )
    fake_b = _FakeProvider(
        "backend_b",
        result=_make_result(backend="backend_b", model="y"),
    )
    registry = ProviderRegistry()
    registry.register(fake_a, default_model="x", declared_models=["x"])
    registry.register(fake_b, default_model="y", declared_models=["y"])

    _, _, rows = await benchmark_sweep.run(
        diff_source=None,
        output_dir=tmp_path / "out",
        backends=["backend_a"],
        models=[],
        kind="pr_diff",
        instructions="t",
        registry=registry,
    )

    assert [r.backend for r in rows] == ["backend_a"]
    assert fake_a.call_count == 1
    assert fake_b.call_count == 0


async def test_run_filters_by_model(tmp_path):
    fake = _FakeProvider(
        "backend_a",
        result=_make_result(backend="backend_a", model="x"),
    )
    registry = ProviderRegistry()
    registry.register(
        fake,
        default_model="x",
        declared_models=["x", "y", "z"],
    )

    _, _, rows = await benchmark_sweep.run(
        diff_source=None,
        output_dir=tmp_path / "out",
        backends=[],
        models=["y"],
        kind="pr_diff",
        instructions="t",
        registry=registry,
    )

    # Only the y row survives.
    assert len(rows) == 1
    assert rows[0].model == "y"


async def test_model_filter_excludes_no_declared_backends(tmp_path):
    """``--model X`` must not run backends that have no declared models.

    The empty sentinel row (``model=""``) for backends without
    declared models can't satisfy any explicit model filter. Earlier
    shapes ran the empty-sentinel row regardless, polluting the
    sweep with rows the operator did not request.
    """
    declared = _FakeProvider(
        "declared",
        result=_make_result(backend="declared", model="x"),
    )
    bare = _FakeProvider(
        "bare",
        result=_make_result(backend="bare", model="provider-default"),
    )
    registry = ProviderRegistry()
    registry.register(declared, default_model="x", declared_models=["x"])
    registry.register(bare)  # no declared models

    _, _, rows = await benchmark_sweep.run(
        diff_source=None,
        output_dir=tmp_path / "out",
        backends=[],
        models=["x"],
        kind="pr_diff",
        instructions="t",
        registry=registry,
    )

    # Only the matching declared row survives; the empty-sentinel
    # ``bare`` row is filtered out because ``""`` can't satisfy
    # ``--model x``.
    assert len(rows) == 1
    assert rows[0].backend == "declared"
    assert rows[0].model == "x"
    assert bare.call_count == 0


async def test_run_runs_once_when_registry_has_no_declared_models(tmp_path):
    """A backend with no declared models still gets a single row.

    Otherwise a freshly-registered provider with empty pricing YAML
    would silently disappear from the matrix. The harness uses
    ``model=''`` for that row so the provider's own default model
    resolution kicks in.
    """
    fake = _FakeProvider(
        "lonely",
        result=_make_result(backend="lonely", model="provider-default"),
    )
    registry = ProviderRegistry()
    registry.register(fake)  # no default, no declared

    _, _, rows = await benchmark_sweep.run(
        diff_source=None,
        output_dir=tmp_path / "out",
        backends=[],
        models=[],
        kind="pr_diff",
        instructions="t",
        registry=registry,
    )

    assert len(rows) == 1
    assert rows[0].backend == "lonely"
    assert rows[0].model == ""
    assert fake.call_count == 1


async def test_run_propagates_cancellation_through_broad_except(tmp_path):
    """``asyncio.CancelledError`` must propagate, not get swallowed
    by the broad ``except Exception`` that converts provider failures
    into errored rows. Otherwise a sweep running under an async
    supervisor wouldn't stop promptly when cancellation is requested.
    """
    import asyncio

    cancelled = _FakeProvider("cancelled", raises=asyncio.CancelledError())
    registry = ProviderRegistry()
    registry.register(cancelled, default_model="m", declared_models=["m"])

    with pytest.raises(asyncio.CancelledError):
        await benchmark_sweep.run(
            diff_source=None,
            output_dir=tmp_path / "out",
            backends=[],
            models=[],
            kind="pr_diff",
            instructions="t",
            registry=registry,
        )


async def test_run_captures_provider_exception_as_errored_row(tmp_path):
    """An uncaught provider exception becomes a structured errored row.

    Otherwise a single misbehaving provider could abort the whole
    sweep. The harness wraps each call so one row's failure doesn't
    starve the rest.
    """
    boom = _FakeProvider("boom", raises=RuntimeError("kaboom"))
    fine = _FakeProvider(
        "fine",
        result=_make_result(backend="fine", model="m"),
    )
    registry = ProviderRegistry()
    registry.register(boom, default_model="m", declared_models=["m"])
    registry.register(fine, default_model="m", declared_models=["m"])

    _, _, rows = await benchmark_sweep.run(
        diff_source=None,
        output_dir=tmp_path / "out",
        backends=[],
        models=[],
        kind="pr_diff",
        instructions="t",
        registry=registry,
    )

    assert len(rows) == 2
    boom_row = next(r for r in rows if r.backend == "boom")
    assert boom_row.disposition == "errored"
    assert boom_row.error_category == "harness_exception"
    assert "kaboom" in boom_row.error
    fine_row = next(r for r in rows if r.backend == "fine")
    assert fine_row.disposition == "posted"


async def test_duration_ms_falls_back_to_wall_clock_when_usage_missing(tmp_path):
    """When a provider returns ``usage=None`` or ``duration_ms=0``,
    the harness must still report a non-zero latency from its own
    wall-clock measurement. Otherwise the matrix reports a
    misleading ``0ms`` for that row.
    """
    # Provider returns a result with usage explicitly cleared to
    # mimic a pathological provider that doesn't track latency.
    sentinel_result = ReviewResult(
        request_id="req",
        summary="ok",
        findings=[],
        disposition="posted",
        backend="laggard",
        model="x",
        usage=None,
    )
    provider = _FakeProvider("laggard", result=sentinel_result)

    registry = ProviderRegistry()
    registry.register(provider, default_model="x", declared_models=["x"])

    _, _, rows = await benchmark_sweep.run(
        diff_source=None,
        output_dir=tmp_path / "out",
        backends=[],
        models=[],
        kind="pr_diff",
        instructions="t",
        registry=registry,
    )

    # Wall-clock fallback clamps to a 1ms floor so a row from a
    # provider that didn't track latency never reports a misleading
    # 0ms even on hosts where the int-cast lands at 0.
    assert len(rows) == 1
    assert rows[0].duration_ms >= 1


async def test_run_writes_diff_payload_to_output(tmp_path):
    fake = _FakeProvider(
        "backend_a",
        result=_make_result(backend="backend_a", model="m"),
    )
    registry = ProviderRegistry()
    registry.register(fake, default_model="m", declared_models=["m"])

    out_dir = tmp_path / "sweep"
    custom_diff = tmp_path / "custom.diff"
    custom_diff.write_text("diff --git a/x b/x\n@@\n-old\n+new\n", encoding="utf-8")

    await benchmark_sweep.run(
        diff_source=str(custom_diff),
        output_dir=out_dir,
        backends=[],
        models=[],
        kind="pr_diff",
        instructions="t",
        registry=registry,
    )

    # The provider received the diff verbatim.
    assert fake.last_request is not None
    assert "old" in fake.last_request.content


async def test_run_always_passes_model_key_in_metadata(tmp_path):
    """``ReviewRequest.metadata`` must always carry a ``model`` key,
    even when the harness is running the empty-sentinel row for a
    backend with no declared models. Mirrors the bus/agent path so
    providers that distinguish "key absent" from "present but empty"
    see the same payload here as in production.
    """
    bare = _FakeProvider("bare", result=_make_result(backend="bare", model="x"))
    declared = _FakeProvider(
        "declared", result=_make_result(backend="declared", model="m")
    )
    registry = ProviderRegistry()
    registry.register(bare)  # no declared models → empty-sentinel row
    registry.register(declared, default_model="m", declared_models=["m"])

    await benchmark_sweep.run(
        diff_source=None,
        output_dir=tmp_path / "out",
        backends=[],
        models=[],
        kind="pr_diff",
        instructions="t",
        registry=registry,
    )

    assert bare.last_request is not None
    assert bare.last_request.metadata == {"model": ""}
    assert declared.last_request is not None
    assert declared.last_request.metadata == {"model": "m"}


async def test_run_writes_relative_artifact_paths_in_summary(tmp_path):
    """``summary.jsonl`` should record artifact paths relative to
    ``output_dir`` so the manifest is portable across machines and
    reruns. Otherwise rows would carry absolute tmp paths that mean
    nothing once the sweep directory moves.
    """
    fake = _FakeProvider("backend_a", result=_make_result(backend="backend_a", model="m"))
    boom = _FakeProvider("boom", raises=RuntimeError("kaboom"))
    registry = ProviderRegistry()
    registry.register(fake, default_model="m", declared_models=["m"])
    registry.register(boom, default_model="m", declared_models=["m"])

    out_dir = tmp_path / "sweep"
    jsonl_path, _, rows = await benchmark_sweep.run(
        diff_source=None,
        output_dir=out_dir,
        backends=[],
        models=[],
        kind="pr_diff",
        instructions="t",
        registry=registry,
    )

    for row in rows:
        assert not Path(row.artifact_path).is_absolute()
        assert row.artifact_path.startswith("artifacts/")
        # Resolves against output_dir to a real file.
        assert (out_dir / row.artifact_path).exists()


def test_main_log_level_rejects_unknown_value(capsys):
    """``--log-level`` is constrained to argparse choices so unknown
    strings exit with a friendly error instead of crashing inside
    ``logging.basicConfig`` later.
    """
    with pytest.raises(SystemExit):
        benchmark_sweep.main(["--log-level", "infoo"])
    captured = capsys.readouterr()
    assert "invalid choice" in captured.err
    assert "INFOO" in captured.err


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_render_markdown_handles_empty_rows():
    md = benchmark_sweep._render_markdown([])
    assert "no rows" in md.lower()
    # Empty-rows fallback must NOT inject a malformed table row
    # (mismatched cell count vs the 10-column header). Emit as a
    # paragraph below the table instead.
    lines = [line for line in md.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|")]
    # Header + alignment row only — no data row pretending to be
    # part of the table.
    assert len(table_lines) == 2


def test_render_markdown_includes_severity_split():
    row = benchmark_sweep._BenchmarkRow(
        backend="ollama",
        model="qwen2.5-coder:14b",
        disposition="posted",
        error="",
        error_category="",
        duration_ms=1234,
        input_tokens=500,
        output_tokens=200,
        finding_count_total=3,
        finding_count_nit=1,
        finding_count_comment=1,
        finding_count_concern=1,
        summary_chars=42,
        artifact_path="/tmp/x.json",
    )
    md = benchmark_sweep._render_markdown([row])
    # Counts are present in the table.
    assert " 1 |" in md  # nit / comment / concern columns
    assert "qwen2.5-coder:14b" in md
    assert "1234" in md
