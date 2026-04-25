"""Benchmark sweep harness — calibrate every registered provider+model.

Replaces the ad-hoc 2026-04-22 9-model sweep (see
``project_benchmark_2026_04_22_full`` memory) with a re-runnable
harness. Iterates the canonical :class:`reviewer.registry.ProviderRegistry`
so any backend / model that lights up via the four shipped providers
gets calibrated automatically — the same code path that powers the
``list_models`` MCP skill.

CLI
---

::

    python -m reviewer.tools.benchmark_sweep \\
        --diff <path-or-PR>      # default: bundled bus_lib_pr14.diff
        --output <dir>           # default: ./benchmark-out/<UTC-stamp>/
        --backend <name>         # filter to one backend (repeatable)
        --model <id>             # filter to one model (repeatable)
        --kind pr_diff           # ReviewRequest.kind; defaults to pr_diff
        --instructions <text>    # extra review instructions

Output
------

For each (backend, model) the harness writes a per-model artifact +
a row in a top-level ``summary.jsonl``. A human-friendly markdown
table lands at ``REPORT.md`` once the sweep completes; both are
ready to drop into a PR description / memory entry.

The harness deliberately does NOT call any bus surface — it
constructs a real ``ProviderRegistry`` via
``ReviewerAgent._build_default_registry()`` and exercises each
``ReviewProvider`` in-process. That keeps the calibration matrix
honest about the same code path the bus actually uses while
sidestepping the bus-restart staleness that would otherwise
surface here.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from khonliang_reviewer import ReviewRequest

from reviewer.agent import ReviewerAgent
from reviewer.registry import ProviderRegistration, ProviderRegistry


_DEFAULT_DIFF_PATH = Path(__file__).parent / "benchmark_data" / "bus_lib_pr14.diff"
_PR_REF_PATTERN = re.compile(r"^([\w.-]+)/([\w.-]+)#(\d+)$")
_LOGGER = logging.getLogger("reviewer.tools.benchmark_sweep")


# ----------------------------------------------------------------------
# Result shapes
# ----------------------------------------------------------------------


@dataclass
class _BenchmarkRow:
    """One row per (backend, model) pair the harness exercises."""

    backend: str
    model: str
    disposition: str
    error: str
    error_category: str
    duration_ms: int
    input_tokens: int
    output_tokens: int
    finding_count_total: int
    finding_count_nit: int
    finding_count_comment: int
    finding_count_concern: int
    summary_chars: int
    artifact_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Diff loading
# ----------------------------------------------------------------------


def load_diff(source: str | None) -> tuple[str, str]:
    """Resolve a ``--diff`` argument to ``(diff_text, label)``.

    ``source`` may be:

    - ``None`` / empty → bundled reference diff at
      :data:`_DEFAULT_DIFF_PATH`. Label includes the file name so
      sweep reports stay self-describing.
    - A filesystem path to a unified-diff file.
    - A ``<owner>/<repo>#<num>`` PR reference, fetched via the ``gh``
      CLI (already required for the broader reviewer development
      flow). Surfacing fetch errors as a friendly RuntimeError.
    """
    if not source:
        # Catch ``OSError`` (parent of ``FileNotFoundError``,
        # ``PermissionError``, etc.) and surface a clearer hint than
        # the raw filesystem error. This branch fires when the
        # package was installed without bundling the
        # ``benchmark_data/`` payload — the harness still works but
        # the operator must pass ``--diff <path>`` or a PR
        # reference.
        try:
            diff = _DEFAULT_DIFF_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"bundled reference diff is missing at "
                f"{_DEFAULT_DIFF_PATH}: {exc}. The package may have "
                f"been installed without ``benchmark_data/`` payload "
                f"(check pyproject.toml ``package-data`` includes "
                f"``reviewer/tools/benchmark_data/*.diff``). Pass "
                f"``--diff <path>`` or a ``<owner>/<repo>#<num>`` "
                f"PR reference to bypass."
            ) from exc
        return diff, f"bundled:{_DEFAULT_DIFF_PATH.name}"

    pr_match = _PR_REF_PATTERN.match(source)
    if pr_match:
        owner, repo, num = pr_match.groups()
        diff = _fetch_pr_diff(f"{owner}/{repo}", int(num))
        return diff, f"gh:{owner}/{repo}#{num}"

    path = Path(source)
    if path.is_file():
        return path.read_text(encoding="utf-8"), f"path:{path}"

    raise RuntimeError(
        f"--diff value {source!r} did not resolve to a file path or a "
        f"<owner>/<repo>#<num> PR reference"
    )


def _fetch_pr_diff(repo: str, pr_number: int) -> str:
    """Fetch a unified diff via ``gh pr diff``.

    Uses subprocess rather than githubkit because the harness runs
    standalone — it shouldn't take an HTTP dependency just to read a
    diff that ``gh`` can produce in one process call.
    """
    cmd = ["gh", "pr", "diff", "--repo", repo, str(pr_number)]
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv from internal call
            cmd, capture_output=True, text=True, check=False, timeout=60
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "gh CLI not found on PATH; install it or pass --diff <path>"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh pr diff {shlex.join(cmd[1:])} failed with code "
            f"{proc.returncode}: {proc.stderr.strip()[:500]}"
        )
    if not proc.stdout.strip():
        raise RuntimeError(
            f"gh pr diff {repo}#{pr_number} produced empty output"
        )
    return proc.stdout


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------


def _filter_registry(
    registry: ProviderRegistry,
    *,
    backends: Iterable[str],
    models: Iterable[str],
) -> list[tuple[ProviderRegistration, str]]:
    """Expand the registry into a flat ``[(registration, model)]`` list.

    Every (backend, model) pair we'd actually exercise. When the
    registry has no declared models for a backend, fall back to
    a single sentinel entry with ``model=""`` so the harness still
    runs the provider once (it'll use whatever the provider's own
    default model resolves to).
    """
    backend_filter = {b for b in backends if b}
    model_filter = {m for m in models if m}
    pairs: list[tuple[ProviderRegistration, str]] = []
    for reg in registry.list():
        if backend_filter and reg.backend not in backend_filter:
            continue
        if reg.models:
            for m in reg.models:
                if model_filter and m not in model_filter:
                    continue
                pairs.append((reg, m))
        else:
            # Backend with no declared models. When ``--model`` is
            # set, the empty sentinel can't satisfy any filter — drop
            # the row rather than running an unfiltered provider in
            # what's supposed to be a model-scoped sweep. When
            # ``--model`` is NOT set, run once with the empty
            # sentinel so a freshly-registered backend with empty
            # pricing YAML still appears in the matrix (its own
            # config default_model resolves at provider time).
            if not model_filter:
                pairs.append((reg, ""))
    return pairs


async def _run_one(
    reg: ProviderRegistration,
    model: str,
    *,
    diff: str,
    kind: str,
    instructions: str,
    artifact_dir: Path,
    registry: ProviderRegistry,
) -> _BenchmarkRow:
    """Exercise a single (backend, model) pair and write the artifact."""
    provider = registry.providers.get(reg.backend)
    if provider is None:
        # Defensive — shouldn't happen given _filter_registry comes
        # from registry.list() — but covers a future race where the
        # registry shape evolves.
        return _failed_row(reg.backend, model, "provider missing from registry")

    request = ReviewRequest(
        kind=kind,
        content=diff,
        instructions=instructions,
        metadata={"model": model} if model else {},
        request_id=f"bench-{uuid.uuid4().hex[:8]}",
    )

    start = time.monotonic()
    try:
        result = await provider.review(request)
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        _LOGGER.warning(
            "benchmark sweep: %s/%s raised %s: %s",
            reg.backend,
            model or "<provider-default>",
            type(exc).__name__,
            exc,
        )
        artifact_path = artifact_dir / _safe_artifact_name(reg.backend, model, "exception.txt")
        artifact_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return _BenchmarkRow(
            backend=reg.backend,
            model=model,
            disposition="errored",
            error=f"{type(exc).__name__}: {exc}",
            error_category="harness_exception",
            duration_ms=duration_ms,
            input_tokens=0,
            output_tokens=0,
            finding_count_total=0,
            finding_count_nit=0,
            finding_count_comment=0,
            finding_count_concern=0,
            summary_chars=0,
            artifact_path=str(artifact_path),
        )

    artifact_path = artifact_dir / _safe_artifact_name(reg.backend, model, "result.json")
    artifact_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    severities = {"nit": 0, "comment": 0, "concern": 0}
    for f in result.findings:
        severities[f.severity] = severities.get(f.severity, 0) + 1

    usage = result.usage
    measured_ms = int((time.monotonic() - start) * 1000)
    # Prefer the provider's reported duration (typically more accurate
    # — it strips off our own argv-assembly overhead) but fall back
    # to wall-clock when the provider didn't populate ``usage`` or
    # left ``duration_ms`` at 0. Otherwise rows from providers that
    # don't track latency themselves would show a misleading 0ms.
    if usage and usage.duration_ms:
        duration_ms = usage.duration_ms
    else:
        duration_ms = measured_ms
    return _BenchmarkRow(
        backend=reg.backend,
        model=model,
        disposition=result.disposition,
        error=result.error,
        error_category=result.error_category,
        duration_ms=duration_ms,
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        finding_count_total=len(result.findings),
        finding_count_nit=severities["nit"],
        finding_count_comment=severities["comment"],
        finding_count_concern=severities["concern"],
        summary_chars=len(result.summary or ""),
        artifact_path=str(artifact_path),
    )


def _failed_row(backend: str, model: str, error: str) -> _BenchmarkRow:
    return _BenchmarkRow(
        backend=backend,
        model=model,
        disposition="errored",
        error=error,
        error_category="harness_setup",
        duration_ms=0,
        input_tokens=0,
        output_tokens=0,
        finding_count_total=0,
        finding_count_nit=0,
        finding_count_comment=0,
        finding_count_concern=0,
        summary_chars=0,
        artifact_path="",
    )


def _safe_artifact_name(backend: str, model: str, suffix: str) -> str:
    """Build a stable per-row filename safe across filesystems.

    Replaces ``/``, ``\\``, and other separator-prone characters
    with ``_`` so model ids like ``kimi-k2.5:cloud`` or
    ``claude-opus-4-7[1m]`` don't break path resolution on Linux,
    macOS, or Windows.
    """
    raw = f"{backend}__{model or 'default'}__{suffix}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def write_summary(rows: list[_BenchmarkRow], output_dir: Path) -> tuple[Path, Path]:
    """Write ``summary.jsonl`` + ``REPORT.md``. Returns both paths."""
    jsonl_path = output_dir / "summary.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), sort_keys=True))
            f.write("\n")

    report_path = output_dir / "REPORT.md"
    report_path.write_text(_render_markdown(rows), encoding="utf-8")
    return jsonl_path, report_path


def _render_markdown(rows: list[_BenchmarkRow]) -> str:
    """Render a copy/paste-friendly markdown table.

    Columns picked to mirror the 2026-04-22 sweep matrix shape:
    backend, model, disposition (so errors stand out), duration,
    token counts, finding counts split by severity. Wide enough to
    spot calibration drift; narrow enough to read in a PR
    description.
    """
    lines = [
        "# Benchmark sweep",
        "",
        "| Backend | Model | Disposition | Duration ms | In tokens | Out tokens | Findings (total) | nit | comment | concern |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        disposition = row.disposition
        if row.error_category:
            disposition = f"{disposition} ({row.error_category})"
        lines.append(
            "| {b} | {m} | {d} | {dur} | {it} | {ot} | {ft} | {fn} | {fc} | {fk} |".format(
                b=row.backend,
                m=row.model or "_default_",
                d=disposition,
                dur=row.duration_ms,
                it=row.input_tokens,
                ot=row.output_tokens,
                ft=row.finding_count_total,
                fn=row.finding_count_nit,
                fc=row.finding_count_comment,
                fk=row.finding_count_concern,
            )
        )
    if not rows:
        lines.append("| _no rows — registry was empty after filters_ | | | | | | | | | |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


async def run(
    *,
    diff_source: str | None,
    output_dir: Path,
    backends: Iterable[str],
    models: Iterable[str],
    kind: str,
    instructions: str,
    config_path: str = "",
    registry: ProviderRegistry | None = None,
) -> tuple[Path, Path, list[_BenchmarkRow]]:
    """Run the sweep. Programmatic entry point used by tests + CLI.

    When ``registry`` is supplied (tests), the agent isn't
    instantiated; otherwise we build the canonical registry via
    ``ReviewerAgent._build_default_registry()`` to match what the
    bus skill path would see at boot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    diff, label = load_diff(diff_source)
    (output_dir / "DIFF_SOURCE.txt").write_text(label + "\n", encoding="utf-8")
    (output_dir / "diff.patch").write_text(diff, encoding="utf-8")

    if registry is None:
        agent = ReviewerAgent(
            agent_id="benchmark-sweep",
            bus_url="http://benchmark.invalid",  # never invoked
            config_path=config_path,
        )
        registry = agent._ensure_registry()

    pairs = _filter_registry(registry, backends=backends, models=models)
    rows: list[_BenchmarkRow] = []
    for reg, model in pairs:
        _LOGGER.info(
            "benchmark sweep: starting %s / %s",
            reg.backend,
            model or "<provider-default>",
        )
        row = await _run_one(
            reg,
            model,
            diff=diff,
            kind=kind,
            instructions=instructions,
            artifact_dir=artifact_dir,
            registry=registry,
        )
        rows.append(row)

    jsonl_path, report_path = write_summary(rows, output_dir)
    return jsonl_path, report_path, rows


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reviewer.tools.benchmark_sweep",
        description=(
            "Calibrate every registered review provider+model against a "
            "reference diff. Produces summary.jsonl + REPORT.md plus "
            "per-row JSON artifacts."
        ),
    )
    parser.add_argument(
        "--diff",
        default=None,
        help=(
            "Path to a unified-diff file, OR a `<owner>/<repo>#<num>` PR "
            "reference (fetched via gh). Defaults to the bundled "
            "bus_lib_pr14.diff so reruns are deterministic."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory. Defaults to ./benchmark-out/<UTC-stamp>/."
        ),
    )
    parser.add_argument(
        "--backend",
        action="append",
        default=[],
        help="Filter to a single backend; pass multiple times for >1.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Filter to a single model; pass multiple times for >1.",
    )
    parser.add_argument(
        "--kind",
        default="pr_diff",
        help="ReviewRequest.kind passed to each provider (default: pr_diff).",
    )
    parser.add_argument(
        "--instructions",
        default="Review for correctness and security; flag concerns first.",
        help="Review instructions threaded into ReviewRequest.instructions.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("KHONLIANG_REVIEWER_CONFIG", ""),
        help=(
            "Reviewer config.yaml path. Defaults to "
            "$KHONLIANG_REVIEWER_CONFIG, then to an empty path (which "
            "produces the built-in defaults)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the sweep (DEBUG/INFO/WARNING). Default: INFO.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(message)s")

    output_dir = (
        Path(args.output)
        if args.output
        else Path("benchmark-out") / dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    )

    try:
        jsonl_path, report_path, rows = asyncio.run(
            run(
                diff_source=args.diff,
                output_dir=output_dir,
                backends=args.backend,
                models=args.model,
                kind=args.kind,
                instructions=args.instructions,
                config_path=args.config,
            )
        )
    except RuntimeError as exc:
        _LOGGER.error("benchmark sweep aborted: %s", exc)
        return 2

    _LOGGER.info("benchmark sweep complete: %d rows", len(rows))
    _LOGGER.info("  summary: %s", jsonl_path)
    _LOGGER.info("  report:  %s", report_path)
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry point
    sys.exit(main())
