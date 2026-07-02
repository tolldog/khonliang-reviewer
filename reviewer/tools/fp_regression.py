"""Re-runnable hot-tier false-positive regression check (fr_reviewer_ff923ebf d).

Complements the one-off before/after measurement that closed candidate (d): this
is the *durable* gate. It runs the current reviewer prompt against a small set of
bundled fixtures that reproduce the recorded hot-tier failure modes and checks two
things at once:

- **FP fixtures** (``fp_*.diff`` in :mod:`reviewer.tools.benchmark_data`) reproduce
  the dogfood false-positive shapes — an echo-bait docstring addition
  (``dog_9902f2f9``), a clean literal-duplication consolidation
  (``dog_3891542a`` "Code Repetition" bait), an added-guard diff
  (``fr_reviewer_8d261d32`` imagined-pre-state bait), and a multi-hunk
  import-then-use diff (``dog_05937209`` hunk-isolation bait). A well-calibrated
  reviewer emits **zero** ``concern``-severity findings on these; the check fails
  when the fraction of runs emitting a concern exceeds ``--max-fp-concern-rate``
  (rate-based, so the verdict doesn't depend on ``--runs``). Fixtures with a
  ``_FP_FORBIDDEN`` entry additionally fail on a *forbidden claim* at ANY
  severity — a provably-false statement (e.g. "imported but never used" about
  an import every later hunk uses) is an FP even as a nit.
- **Control fixtures** (``control_*.diff``) carry a genuine introduced defect (a
  removed ``with`` → file-handle leak). The reviewer must **still** flag them — the
  check fails if the control hit-rate drops below ``--min-control-hit-rate``. This
  guards against a calibration that reduces false positives by silencing the model
  entirely.

The provider samples at its default decoding (ollama sets only ``num_ctx``), so a
single run is noisy — default ``--runs 5`` and the thresholds are rate-based.

CLI::

    python -m reviewer.tools.fp_regression --runs 5   # defaults to qwen2.5-coder:14b

Exit code is 0 when every fixture passes, 1 otherwise — usable as a gate. The
default model is **qwen2.5-coder:14b**: the FP-concern behaviour lives on qwen (not
the conservative deepseek default), and qwen reliably flags the control defect
(5/5 in measurement), whereas deepseek under-catches it as a style nit and fails
the control legitimately. See the ``reviewer-fp-calibration-measurement`` memory.

The pure aggregation/verdict functions (:func:`classify_run`, :func:`evaluate`) are
model-free and unit-tested; only :func:`run` touches a live provider.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from importlib import resources
from typing import Iterable

from khonliang_reviewer import ReviewRequest, ReviewResult

_FIXTURE_PACKAGE = "reviewer.tools.benchmark_data"
_FP_PREFIX = "fp_"
_CONTROL_PREFIX = "control_"

#: Case-insensitive *phrases* that identify a control fixture's SEEDED defect in a
#: finding's title/body. A control run only counts as "retained" when it flags
#: *this* defect — not any unrelated nit — so the gate actually catches the
#: regression it exists for (a calibration that silences the real defect but still
#: emits chatter). Phrases are deliberately multi-word/specific, NOT broad single
#: tokens: bare ``close``/``resource`` would substring-match unrelated notes ("names
#: too close together", "resource usage could be improved") and false-green the
#: gate. These phrases all appear verbatim in measured qwen leak findings ("File Not
#: Closed", "resource leaks", "Resource Management Issue", "close the file",
#: "context manager"). Every ``control_*.diff`` MUST have an entry; a control
#: fixture with no expectation is a configuration error (see :func:`load_fp_cases`).
_CONTROL_EXPECTATIONS: dict[str, tuple[str, ...]] = {
    "control_resource_leak": (
        "not closed",
        "never closed",
        "isn't closed",
        "close the file",
        "closing the file",
        "without closing",
        "resource leak",
        "resource management",
        "file descriptor",
        "file handle leak",
        "context manager",
        "with statement",
        "with-statement",
        "unclosed",
    ),
}

#: The FP-bait fixtures the gate is expected to exercise. Declared explicitly (like
#: the control set in _CONTROL_EXPECTATIONS) so packaging drift or a rename that
#: DROPS a specific fixture is caught — a "≥1 of each kind" check would let the
#: gate silently stop covering, e.g., the echo case while still reporting green.
_EXPECTED_FP_FIXTURES: frozenset[str] = frozenset(
    {
        "fp_docstring_prose",
        "fp_consolidate_literals",
        "fp_applied_guard",
        "fp_hunk_isolation",
    }
)

#: Forbidden-claim keywords per FP fixture, matched against findings of ANY
#: severity (same case-insensitive matching as _CONTROL_EXPECTATIONS). The
#: concern-rate metric alone misses factually-false claims emitted at nit /
#: comment severity — dog_05937209's "imported but never used" FP was a nit.
#: A finding on the fixture whose title/body names the forbidden claim is a
#: false positive regardless of severity: the fixture is constructed so the
#: claim is provably wrong (e.g. the import added in hunk 1 is used in every
#: later hunk). Only fixtures with an entry get the extra check.
_FP_FORBIDDEN: dict[str, tuple[str, ...]] = {
    "fp_hunk_isolation": (
        "unused",
        "never used",
        "not used",
        "isn't used",
        "unreferenced",
        "imported but",
        "undefined",
    ),
}


@dataclass(frozen=True)
class FpCase:
    """A regression fixture: an FP-bait or a control diff."""

    name: str
    kind: str  # "fp" | "control"
    diff: str
    #: For control fixtures: keywords identifying the seeded defect. Empty for FP.
    expect_keywords: tuple[str, ...] = ()
    #: For FP fixtures: keywords of the forbidden (provably false) claim, any
    #: severity. Empty when the fixture has no forbidden-claim entry.
    forbid_keywords: tuple[str, ...] = ()


def load_fp_cases() -> list[FpCase]:
    """Load the bundled ``fp_*.diff`` / ``control_*.diff`` fixtures.

    Uses :mod:`importlib.resources` so it works from an installed wheel (the
    ``*.diff`` glob is attached to :mod:`reviewer.tools.benchmark_data` via
    package-data). Files without a recognized prefix (e.g. the benchmark
    reference diff ``bus_lib_pr14.diff``) are ignored.
    """
    cases: list[FpCase] = []
    root = resources.files(_FIXTURE_PACKAGE)
    for entry in root.iterdir():
        fname = entry.name
        if not fname.endswith(".diff"):
            continue
        name = fname[: -len(".diff")]
        forbid: tuple[str, ...] = ()
        if fname.startswith(_FP_PREFIX):
            kind, expect = "fp", ()
            forbid = _FP_FORBIDDEN.get(name, ())
        elif fname.startswith(_CONTROL_PREFIX):
            kind = "control"
            expect = _CONTROL_EXPECTATIONS.get(name)
            if not expect:
                # A control fixture with no seeded-defect keywords cannot be
                # validated — its retention check would pass on any finding,
                # exactly the P1 hole. Refuse rather than silently under-check.
                raise RuntimeError(
                    f"control fixture {name!r} has no _CONTROL_EXPECTATIONS entry"
                )
        else:
            continue
        cases.append(
            FpCase(
                name=name,
                kind=kind,
                diff=entry.read_text(encoding="utf-8"),
                expect_keywords=expect,
                forbid_keywords=forbid,
            )
        )
    cases.sort(key=lambda c: (c.kind, c.name))
    # A gate that silently loses coverage must never report green: validate the
    # FULL expected fixture set by name (not just "≥1 of each kind"), so a missing
    # package-data payload OR a drop/rename of any single recorded fixture raises
    # rather than passing vacuously on the survivors.
    loaded = {c.name for c in cases}
    expected = _EXPECTED_FP_FIXTURES | set(_CONTROL_EXPECTATIONS)
    missing = expected - loaded
    if missing:
        raise RuntimeError(
            f"fp_regression: expected fixtures missing from {_FIXTURE_PACKAGE!r}: "
            f"{sorted(missing)} (loaded {sorted(loaded)}). Bundled *.diff "
            f"package-data missing or a recorded fixture was dropped/renamed."
        )
    return cases


@dataclass
class CaseReport:
    """Aggregated per-fixture outcome across N runs."""

    name: str
    kind: str
    runs: int
    #: Keywords identifying the seeded defect (control fixtures only).
    expect_keywords: tuple[str, ...] = ()
    #: Forbidden-claim keywords (FP fixtures only; see _FP_FORBIDDEN).
    forbid_keywords: tuple[str, ...] = ()
    concern_runs: int = 0  # runs that produced >=1 concern-severity finding
    concern_total: int = 0  # total concern findings across all runs
    finding_runs: int = 0  # runs that produced >=1 finding of any severity
    defect_hit_runs: int = 0  # control runs that flagged the SEEDED defect
    errored_runs: int = 0
    concern_titles: list[str] = field(default_factory=list)
    forbidden_hit_runs: int = 0  # fp runs that emitted the forbidden claim (any severity)
    forbidden_titles: list[str] = field(default_factory=list)


def _finding_hits_defect(finding, keywords: tuple[str, ...]) -> bool:
    """True when a finding's title/body names the seeded defect (case-insensitive)."""
    hay = f"{finding.title or ''} {finding.body or ''}".lower()
    return any(kw in hay for kw in keywords)


def classify_run(report: CaseReport, result: ReviewResult) -> None:
    """Fold a single review ``result`` into ``report`` (pure, no I/O).

    Counts concern-severity findings (the FP metric) and — for control fixtures
    — whether the run flagged the *seeded* defect specifically (matched against
    ``report.expect_keywords``), NOT merely any finding. A control that emits an
    unrelated nit while missing the real defect must not count as retained. An
    errored result (e.g. a timeout) counts as an errored run and contributes no
    findings — neither a false positive nor a caught defect.
    """
    if result.disposition == "errored":
        report.errored_runs += 1
        return
    findings = result.findings or []
    if findings:
        report.finding_runs += 1
    concerns = [f for f in findings if f.severity == "concern"]
    if concerns:
        report.concern_runs += 1
        report.concern_total += len(concerns)
        report.concern_titles.extend((f.title or "(untitled)") for f in concerns)
    if report.kind == "control" and any(
        _finding_hits_defect(f, report.expect_keywords) for f in findings
    ):
        report.defect_hit_runs += 1
    if report.kind == "fp" and report.forbid_keywords:
        # Forbidden-claim check (dog_05937209): the fixture is constructed so
        # this claim is provably false — emitting it at ANY severity is an FP,
        # not just at concern. Reuses the control-keyword matcher.
        hits = [
            f for f in findings if _finding_hits_defect(f, report.forbid_keywords)
        ]
        if hits:
            report.forbidden_hit_runs += 1
            report.forbidden_titles.extend((f.title or "(untitled)") for f in hits)


def evaluate(
    reports: Iterable[CaseReport],
    *,
    max_fp_concern_rate: float,
    min_control_hit_rate: float,
) -> tuple[bool, list[str]]:
    """Turn per-fixture reports into a pass/fail verdict + human lines (pure).

    Both verdicts are **rate-based** over the non-errored runs, so the outcome
    reflects the underlying regression rate rather than the sample size
    (``--runs``): the same behaviour passes/fails identically at N=5 and N=10.

    - An ``fp`` fixture FAILS when the fraction of runs that produced any
      ``concern`` finding exceeds ``max_fp_concern_rate`` (default 0.0 = zero
      tolerance, matching the "zero echo/invert concerns" acceptance).
    - A ``control`` fixture FAILS when the fraction of runs that flagged the
      *seeded defect* falls below ``min_control_hit_rate`` — a real defect was
      silenced.

    Errored runs are excluded from both denominators so a flaky timeout isn't a
    false FP or a false miss; a fixture whose every run errored fails (nothing
    was actually measured).
    """
    ok = True
    lines: list[str] = []
    for r in reports:
        scored = r.runs - r.errored_runs
        if r.kind == "fp":
            rate = (r.concern_runs / scored) if scored else 1.0
            passed = scored > 0 and rate <= max_fp_concern_rate
            detail = (
                f"concern runs {r.concern_runs}/{scored} "
                f"(rate {rate:.2f}, max {max_fp_concern_rate:.2f})"
            )
            if r.concern_titles:
                detail += f" :: {', '.join(sorted(set(r.concern_titles)))}"
            if r.forbid_keywords:
                # Forbidden-claim verdict shares the FP tolerance: the claim is
                # provably false at any severity, so the same zero-tolerance
                # default applies (rate-based, --runs-independent).
                frate = (r.forbidden_hit_runs / scored) if scored else 1.0
                fpassed = scored > 0 and frate <= max_fp_concern_rate
                detail += (
                    f"; forbidden-claim runs {r.forbidden_hit_runs}/{scored} "
                    f"(rate {frate:.2f}, max {max_fp_concern_rate:.2f})"
                )
                if r.forbidden_titles:
                    detail += f" :: {', '.join(sorted(set(r.forbidden_titles)))}"
                passed = passed and fpassed
        else:  # control
            rate = (r.defect_hit_runs / scored) if scored else 0.0
            passed = scored > 0 and rate >= min_control_hit_rate
            detail = (
                f"seeded defect flagged {r.defect_hit_runs}/{scored} runs "
                f"(rate {rate:.2f}, min {min_control_hit_rate:.2f})"
            )
        if r.errored_runs:
            detail += f" [errored {r.errored_runs}/{r.runs}]"
        ok = ok and passed
        lines.append(f"[{'PASS' if passed else 'FAIL'}] {r.kind}:{r.name} — {detail}")
    return ok, lines


async def run(
    *,
    model: str,
    backend: str,
    runs: int,
    config_path: str = "",
    provider=None,
) -> list[CaseReport]:
    """Execute the check live. ``provider`` may be injected for tests; otherwise
    the canonical registry is built via ``ReviewerAgent`` (same path the bus uses).
    """
    if provider is None:
        from reviewer.agent import ReviewerAgent  # cold-path import

        agent = ReviewerAgent(
            agent_id="fp-regression",
            bus_url="http://fp-regression.invalid",  # never invoked
            config_path=config_path,
        )
        provider = agent._ensure_registry().get_provider(backend)
    if provider is None:
        raise SystemExit(f"backend {backend!r} not registered")

    reports: list[CaseReport] = []
    for case in load_fp_cases():
        report = CaseReport(
            name=case.name,
            kind=case.kind,
            runs=runs,
            expect_keywords=case.expect_keywords,
            forbid_keywords=case.forbid_keywords,
        )
        for i in range(runs):
            request = ReviewRequest(
                kind="pr_diff",
                content=case.diff,
                metadata={"model": model},
                request_id=f"fpreg-{case.name}-{i}",
            )
            try:
                result = await provider.review(request)
            except Exception as exc:  # noqa: BLE001 — one bad run shouldn't abort the sweep
                report.errored_runs += 1
                sys.stderr.write(f"  {case.name} run {i}: error {exc}\n")
                continue
            classify_run(report, result)
        reports.append(report)
    return reports


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m reviewer.tools.fp_regression",
        description="Hot-tier false-positive regression check.",
    )
    # Default to qwen: it is both where the FP-concern behaviour lives AND a
    # reliable catcher of the control defect (5/5 runs named the resource leak in
    # measurement). deepseek under-catches real defects — it frames the removed
    # `with` as a style/whitespace nit, so it fails the control legitimately and
    # is the wrong model for this gate (see reviewer-fp-calibration-measurement).
    p.add_argument("--model", default="qwen2.5-coder:14b")
    p.add_argument("--backend", default="ollama")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument(
        "--max-fp-concern-rate",
        type=float,
        default=0.0,
        help="max fraction of runs allowed to emit a concern per FP fixture "
        "(default 0.0 = zero tolerance; rate-based so it's --runs-independent)",
    )
    p.add_argument(
        "--min-control-hit-rate",
        type=float,
        default=0.6,
        help="min fraction of (non-errored) control runs that must flag the defect",
    )
    p.add_argument("--config", default="", help="path to reviewer config.yaml")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    reports = asyncio.run(
        run(
            model=args.model,
            backend=args.backend,
            runs=args.runs,
            config_path=args.config,
        )
    )
    ok, lines = evaluate(
        reports,
        max_fp_concern_rate=args.max_fp_concern_rate,
        min_control_hit_rate=args.min_control_hit_rate,
    )
    print(f"# FP-regression: {args.backend}/{args.model}, N={args.runs}")
    for line in lines:
        print(line)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
