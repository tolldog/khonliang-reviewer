"""dropped_findings audit trail recording across distill transforms.

fr_reviewer_de1694a8: the dropping transforms (severity_filter, dedup,
max_findings) record what they removed onto ``ReviewResult.dropped_findings``
so audit / benchmark corpora can recover the raw provider output. The
``audit_corpus`` audience short-circuits the pipeline, so it drops nothing.
"""

from __future__ import annotations

from khonliang_reviewer import ReviewFinding, ReviewResult

from reviewer.distill import run_pipeline
from reviewer.distill.transforms.dedup import apply_dedup
from reviewer.distill.transforms.max_findings import apply_max_findings
from reviewer.distill.transforms.severity_filter import apply_severity_filter
from reviewer.rules.distill import DistillConfig


def _finding(severity: str, title: str, body: str = "b") -> ReviewFinding:
    return ReviewFinding(severity=severity, title=title, body=body)  # type: ignore[arg-type]


def _result(findings: list[ReviewFinding]) -> ReviewResult:
    return ReviewResult(request_id="r", summary="s", findings=findings)


def test_severity_filter_records_dropped():
    out = apply_severity_filter(
        _result([_finding("nit", "n"), _finding("concern", "c")]),
        DistillConfig(severity_floor="concern"),
    )
    assert [f.title for f in out.findings] == ["c"]
    assert [f.title for f in out.dropped_findings] == ["n"]


def test_max_findings_records_overflow():
    fs = [_finding("comment", f"f{i}") for i in range(5)]
    out = apply_max_findings(_result(fs), DistillConfig(max_findings=2))
    assert len(out.findings) == 2
    assert len(out.dropped_findings) == 3


def test_dedup_records_collapsed_duplicates():
    out = apply_dedup(
        _result([_finding("comment", "same", "same"), _finding("comment", "same", "same")]),
        DistillConfig(dedup="exact"),
    )
    assert len(out.findings) == 1
    assert len(out.dropped_findings) == 1


def test_dropped_findings_accumulate_across_pipeline():
    # 1 nit (floored out) + 4 comments capped to 2 → dropped = 1 + 2.
    nit = _finding("nit", "nit")
    comments = [_finding("comment", f"c{i}") for i in range(4)]
    out = run_pipeline(
        _result([nit, *comments]),
        DistillConfig(severity_floor="comment", max_findings=2),
    )
    assert len(out.findings) == 2
    assert len(out.dropped_findings) == 3


def test_audit_corpus_records_no_dropped():
    # Short-circuit: nothing is filtered, so nothing is dropped.
    out = run_pipeline(
        _result([_finding("nit", "n")]),
        DistillConfig(severity_floor="concern", audience="audit_corpus"),
    )
    assert [f.title for f in out.findings] == ["n"]
    assert out.dropped_findings == []


def test_inert_config_records_no_dropped():
    out = run_pipeline(_result([_finding("concern", "c")]), DistillConfig())
    assert out.dropped_findings == []
