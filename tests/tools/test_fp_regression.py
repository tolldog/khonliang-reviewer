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
    # The control fixture carries seeded-defect keywords (P1 wiring); FP has none.
    assert by_name["control_resource_leak"].expect_keywords  # non-empty
    assert by_name["fp_docstring_prose"].expect_keywords == ()


class _FakeEntry:
    def __init__(self, name, text=""):
        self.name = name
        self._text = text

    def read_text(self, encoding="utf-8"):
        return self._text


class _FakeRoot:
    def __init__(self, entries):
        self._entries = entries

    def iterdir(self):
        return iter(self._entries)


def test_load_fp_cases_raises_on_empty_fixture_set(monkeypatch):
    """P2: a missing package-data payload / drifted prefixes must NOT pass
    vacuously — the loader raises rather than returning an empty (green) set."""
    import pytest

    from reviewer.tools import fp_regression as m

    # Only the unrelated reference diff present → all expected fixtures missing.
    monkeypatch.setattr(
        m.resources, "files", lambda pkg: _FakeRoot([_FakeEntry("bus_lib_pr14.diff")])
    )
    with pytest.raises(RuntimeError, match="expected fixtures missing"):
        m.load_fp_cases()


def test_load_fp_cases_raises_when_one_expected_fixture_dropped(monkeypatch):
    """P2 (round 2): dropping a SINGLE recorded fixture while others survive must
    raise — a '>=1 of each kind' check would silently lose that regression's
    coverage while still reporting green."""
    import pytest

    from reviewer.tools import fp_regression as m

    # Present: the control + one FP fixture; MISSING: fp_docstring_prose.
    monkeypatch.setattr(
        m.resources,
        "files",
        lambda pkg: _FakeRoot(
            [
                _FakeEntry("fp_consolidate_literals.diff", "d"),
                _FakeEntry("control_resource_leak.diff", "d"),
            ]
        ),
    )
    with pytest.raises(RuntimeError, match="fp_docstring_prose"):
        m.load_fp_cases()


def test_load_fp_cases_raises_on_control_without_expectation(monkeypatch):
    """A control fixture with no _CONTROL_EXPECTATIONS entry is refused — else its
    retention check would pass on any finding (the P1 hole)."""
    import pytest

    from reviewer.tools import fp_regression as m

    monkeypatch.setattr(
        m.resources,
        "files",
        lambda pkg: _FakeRoot(
            [_FakeEntry("fp_x.diff", "d"), _FakeEntry("control_unknown.diff", "d")]
        ),
    )
    with pytest.raises(RuntimeError, match="no _CONTROL_EXPECTATIONS"):
        m.load_fp_cases()


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


def test_classify_run_control_hit_requires_seeded_defect():
    """P1 regression: a control run only counts as retained when a finding names
    the SEEDED defect — an unrelated nit must NOT count."""
    r = CaseReport(name="c", kind="control", runs=2, expect_keywords=("leak", "close"))
    # Run 1: flags the leak → defect hit.
    classify_run(r, _result([_f("concern", "Resource leak: file never closed")]))
    # Run 2: emits an unrelated nit, MISSES the leak → finding but no defect hit.
    classify_run(r, _result([_f("nit", "Prefer snake_case")]))
    assert r.finding_runs == 2
    assert r.defect_hit_runs == 1  # only the run that named the defect


def test_classify_run_errored_counts_as_errored_not_finding():
    r = CaseReport(name="x", kind="control", runs=1, expect_keywords=("leak",))
    classify_run(r, _result([], disposition="errored", error="timeout"))
    assert r.errored_runs == 1
    assert r.finding_runs == 0
    assert r.defect_hit_runs == 0


# -- evaluate: FP fixtures ---------------------------------------------


def test_evaluate_fp_passes_at_zero_concern_runs():
    r = CaseReport(name="fp1", kind="fp", runs=5, finding_runs=3, concern_runs=0)
    ok, lines = evaluate([r], max_fp_concern_rate=0.0, min_control_hit_rate=0.6)
    assert ok
    assert "PASS" in lines[0]


def test_evaluate_fp_fails_when_concern_rate_exceeds_limit():
    r = CaseReport(name="fp1", kind="fp", runs=5, concern_runs=2, concern_total=3,
                   concern_titles=["Code Repetition"])
    ok, lines = evaluate([r], max_fp_concern_rate=0.0, min_control_hit_rate=0.6)
    assert not ok
    assert "FAIL" in lines[0]
    assert "Code Repetition" in lines[0]  # offending title surfaces


def test_evaluate_fp_verdict_is_runs_independent():
    """P2: the same concern RATE passes/fails the same way regardless of --runs.
    20% concern rate under a 0.25 tolerance passes at both N=5 and N=10."""
    r5 = CaseReport(name="fp", kind="fp", runs=5, concern_runs=1)   # 0.20
    r10 = CaseReport(name="fp", kind="fp", runs=10, concern_runs=2)  # 0.20
    for r in (r5, r10):
        assert evaluate([r], max_fp_concern_rate=0.25, min_control_hit_rate=0.6)[0]
        assert not evaluate([r], max_fp_concern_rate=0.0, min_control_hit_rate=0.6)[0]


def test_evaluate_fp_all_errored_fails():
    r = CaseReport(name="fp", kind="fp", runs=5, errored_runs=5)
    assert not evaluate([r], max_fp_concern_rate=0.0, min_control_hit_rate=0.6)[0]


# -- evaluate: control fixtures ----------------------------------------


def test_evaluate_control_passes_when_defect_retained():
    r = CaseReport(name="c1", kind="control", runs=5, finding_runs=5, defect_hit_runs=5)
    ok, _ = evaluate([r], max_fp_concern_rate=0.0, min_control_hit_rate=0.6)
    assert ok


def test_evaluate_control_fails_when_defect_silenced_despite_chatter():
    # P1: model emits findings every run (chatter) but only 1/5 name the real
    # defect → the gate must FAIL on defect_hit_runs, not finding_runs.
    r = CaseReport(name="c1", kind="control", runs=5, finding_runs=5, defect_hit_runs=1)
    ok, lines = evaluate([r], max_fp_concern_rate=0.0, min_control_hit_rate=0.6)
    assert not ok
    assert "FAIL" in lines[0]


def test_evaluate_control_excludes_errored_runs_from_denominator():
    # 2 defect-hits / 2 scored (3 errored) → rate 1.0, passes; errors annotated.
    r = CaseReport(name="c1", kind="control", runs=5, defect_hit_runs=2, errored_runs=3)
    ok, lines = evaluate([r], max_fp_concern_rate=0.0, min_control_hit_rate=0.6)
    assert ok
    assert "errored 3/5" in lines[0]


def test_evaluate_control_all_errored_fails():
    r = CaseReport(name="c1", kind="control", runs=5, errored_runs=5)
    ok, _ = evaluate([r], max_fp_concern_rate=0.0, min_control_hit_rate=0.6)
    assert not ok


def test_control_keywords_reject_unrelated_wording():
    """P1 (substring safety): the real leak phrases must match genuine leak
    findings (measured qwen phrasings) but NOT unrelated notes that merely share
    a broad word ("too close together", "resource usage could be improved")."""
    from reviewer.tools.fp_regression import (
        _CONTROL_EXPECTATIONS,
        _finding_hits_defect,
    )

    kw = _CONTROL_EXPECTATIONS["control_resource_leak"]

    def hit(title, body=""):
        return _finding_hits_defect(_f2(title, body), kw)

    # Genuine leak findings (verbatim-ish from measured qwen runs) → match.
    assert hit("File Not Closed", "not closed after reading; resource leaks")
    assert hit("Resource Management Issue", "the file is opened but not closed")
    assert hit("Best Practices", "use a context manager")
    # Unrelated wording that shares only a broad token → must NOT match.
    assert not hit("Naming", "these names are too close together")
    assert not hit("Perf", "resource usage could be improved")
    assert not hit("Style", "prefer snake_case")


def _f2(title, body):
    return ReviewFinding(severity="comment", title=title, body=body)  # type: ignore[arg-type]


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
    # Control: fake names the leak every run → seeded defect retained.
    assert by["control_resource_leak"].finding_runs == 2
    assert by["control_resource_leak"].defect_hit_runs == 2
    ok, _ = evaluate(reports, max_fp_concern_rate=0.0, min_control_hit_rate=0.6)
    assert ok
