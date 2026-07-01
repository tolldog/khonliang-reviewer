"""Unit tests for the FP-regression check's pure core (no live model)."""

from __future__ import annotations

from khonliang_reviewer import ReviewFinding, ReviewResult

from reviewer.tools.fp_regression import (
    CaseReport,
    classify_run,
    evaluate,
    load_fp_cases,
    run,
)


def _result(findings, *, disposition="posted", error=""):
    return ReviewResult(
        request_id="r",
        summary="",
        findings=findings,
        disposition=disposition,  # type: ignore[arg-type]
        error=error,
        backend="ollama",
        model="m",
    )


def _f(severity, title="t"):
    return ReviewFinding(severity=severity, title=title, body="b")  # type: ignore[arg-type]


# -- fixtures load + classify by prefix -------------------------------


def test_load_fp_cases_picks_up_bundled_fixtures():
    cases = load_fp_cases()
    by_name = {c.name: c for c in cases}
    # Both FP fixtures and the control ship in benchmark_data.
    assert by_name["fp_docstring_prose"].kind == "fp"
    assert by_name["fp_consolidate_literals"].kind == "fp"
    assert by_name["control_resource_leak"].kind == "control"
    # The unrelated benchmark reference diff is NOT picked up.
    assert "bus_lib_pr14" not in by_name
    # Diff bodies are real.
    assert "parse_header" in by_name["fp_docstring_prose"].diff
    assert "open(path)" in by_name["control_resource_leak"].diff


# -- classify_run ------------------------------------------------------


def test_classify_run_counts_concerns_and_findings():
    r = CaseReport(name="x", kind="fp", runs=3)
    classify_run(r, _result([_f("concern", "A"), _f("nit")]))
    classify_run(r, _result([_f("comment")]))
    classify_run(r, _result([]))
    assert r.concern_runs == 1
    assert r.concern_total == 1
    assert r.finding_runs == 2  # first two runs had findings
    assert r.concern_titles == ["A"]
    assert r.errored_runs == 0


def test_classify_run_errored_counts_as_errored_not_finding():
    r = CaseReport(name="x", kind="control", runs=1)
    classify_run(r, _result([], disposition="errored", error="timeout"))
    assert r.errored_runs == 1
    assert r.finding_runs == 0
    assert r.concern_total == 0


# -- evaluate: FP fixtures ---------------------------------------------


def test_evaluate_fp_passes_at_zero_concerns():
    r = CaseReport(name="fp1", kind="fp", runs=5, finding_runs=3)
    ok, lines = evaluate([r], max_fp_concerns=0, min_control_hit_rate=0.6)
    assert ok
    assert "PASS" in lines[0]


def test_evaluate_fp_fails_when_concerns_exceed_limit():
    r = CaseReport(name="fp1", kind="fp", runs=5, concern_runs=2, concern_total=3,
                   concern_titles=["Code Repetition", "Code Repetition"])
    ok, lines = evaluate([r], max_fp_concerns=0, min_control_hit_rate=0.6)
    assert not ok
    assert "FAIL" in lines[0]
    # Offending titles surface for triage.
    assert "Code Repetition" in lines[0]


def test_evaluate_fp_respects_nonzero_tolerance():
    r = CaseReport(name="fp1", kind="fp", runs=5, concern_total=1)
    assert evaluate([r], max_fp_concerns=1, min_control_hit_rate=0.6)[0]
    assert not evaluate([r], max_fp_concerns=0, min_control_hit_rate=0.6)[0]


# -- evaluate: control fixtures ----------------------------------------


def test_evaluate_control_passes_when_defect_retained():
    r = CaseReport(name="c1", kind="control", runs=5, finding_runs=5)
    ok, _ = evaluate([r], max_fp_concerns=0, min_control_hit_rate=0.6)
    assert ok


def test_evaluate_control_fails_when_defect_silenced():
    # Calibration that gags the model: 1/5 runs flag the real defect.
    r = CaseReport(name="c1", kind="control", runs=5, finding_runs=1)
    ok, lines = evaluate([r], max_fp_concerns=0, min_control_hit_rate=0.6)
    assert not ok
    assert "FAIL" in lines[0]


def test_evaluate_control_excludes_errored_runs_from_denominator():
    # 2 flagged / 2 scored (3 errored) → rate 1.0, passes; errors are annotated.
    r = CaseReport(name="c1", kind="control", runs=5, finding_runs=2, errored_runs=3)
    ok, lines = evaluate([r], max_fp_concerns=0, min_control_hit_rate=0.6)
    assert ok
    assert "errored 3/5" in lines[0]


def test_evaluate_control_all_errored_fails():
    r = CaseReport(name="c1", kind="control", runs=5, errored_runs=5)
    ok, _ = evaluate([r], max_fp_concerns=0, min_control_hit_rate=0.6)
    assert not ok


# -- run() with an injected fake provider (no live model) --------------


class _FakeProvider:
    """Returns a scripted result keyed by fixture kind embedded in the diff."""

    async def review(self, request):
        # FP fixtures → clean (no concern); control → a flagged defect.
        if "load_config" in request.content:  # the control fixture body
            return _result([_f("concern", "Resource leak")])
        return _result([_f("nit", "style")])


async def test_run_with_fake_provider_end_to_end():
    reports = await run(model="m", backend="ollama", runs=2, provider=_FakeProvider())
    by = {r.name: r for r in reports}
    # FP fixtures: fake returns only nits → zero concerns → pass.
    assert by["fp_docstring_prose"].concern_total == 0
    assert by["fp_consolidate_literals"].concern_total == 0
    # Control: fake flags it every run → retained.
    assert by["control_resource_leak"].finding_runs == 2
    ok, _ = evaluate(reports, max_fp_concerns=0, min_control_hit_rate=0.6)
    assert ok
