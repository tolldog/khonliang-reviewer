"""Unit coverage for the binary-questions verdict helpers
(fr_khonliang-reviewer_a585ea3d): ``parse_verdicts`` (bus/model-boundary
validation of the model-produced ``verdicts`` array) and ``score_verdicts``
(per-dimension + overall fraction-of-True aggregation)."""

from __future__ import annotations

from khonliang_reviewer import Verdict

from reviewer.providers._prompt import parse_verdicts, score_verdicts


def _item(dimension="correctness", question="Is it correct?", answer=True, explanation="ok"):
    return {
        "dimension": dimension,
        "question": question,
        "answer": answer,
        "explanation": explanation,
    }


# -- parse_verdicts ---------------------------------------------------


def test_parse_verdicts_wellformed_payload():
    payload = {
        "summary": "s",
        "verdicts": [
            _item("correctness", "Q1?", True, "e1"),
            _item("tests", "Q2?", False, "e2"),
        ],
    }
    out = parse_verdicts(payload)
    assert len(out) == 2
    assert all(isinstance(v, Verdict) for v in out)
    assert out[0].dimension == "correctness"
    assert out[0].answer is True
    assert out[1].dimension == "tests"
    assert out[1].answer is False
    assert out[1].explanation == "e2"


def test_parse_verdicts_no_key_returns_empty():
    # Holistic-mode payload (no verdicts key) — every provider calls
    # parse_verdicts unconditionally, so this must be safe.
    assert parse_verdicts({"summary": "s", "findings": []}) == []


def test_parse_verdicts_non_list_returns_empty():
    assert parse_verdicts({"verdicts": "nope"}) == []
    assert parse_verdicts({"verdicts": None}) == []
    assert parse_verdicts({"verdicts": {"dimension": "x"}}) == []


def test_parse_verdicts_skips_non_bool_answer():
    # "false" is a string, not a JSON boolean — must be dropped, NOT coerced
    # (bool("false") is True, the footgun this guard closes).
    payload = {
        "verdicts": [
            _item(answer="false"),
            _item(answer=1),
            _item(answer=None),
            _item(dimension="tests", answer=True),
        ]
    }
    out = parse_verdicts(payload)
    assert len(out) == 1
    assert out[0].dimension == "tests"
    assert out[0].answer is True


def test_parse_verdicts_skips_missing_or_bad_fields():
    payload = {
        "verdicts": [
            {"question": "q?", "answer": True},  # missing dimension
            {"dimension": "correctness", "answer": True},  # missing question
            {"dimension": "", "question": "q?", "answer": True},  # empty dimension
            {"dimension": "x", "question": "", "answer": True},  # empty question
            "not-a-dict",
            _item(dimension="clarity"),  # the one valid item
        ]
    }
    out = parse_verdicts(payload)
    assert len(out) == 1
    assert out[0].dimension == "clarity"


def test_parse_verdicts_coerces_missing_explanation_to_empty_str():
    payload = {"verdicts": [{"dimension": "correctness", "question": "q?", "answer": True}]}
    out = parse_verdicts(payload)
    assert len(out) == 1
    assert out[0].explanation == ""


# -- score_verdicts ---------------------------------------------------


def test_score_verdicts_empty_is_zero_overall():
    assert score_verdicts([]) == {"overall": 0.0}


def test_score_verdicts_per_dimension_and_overall():
    verdicts = [
        Verdict(dimension="correctness", question="q", answer=True, explanation=""),
        Verdict(dimension="correctness", question="q", answer=False, explanation=""),
        Verdict(dimension="tests", question="q", answer=True, explanation=""),
        Verdict(dimension="security", question="q", answer=False, explanation=""),
    ]
    scores = score_verdicts(verdicts)
    assert scores["correctness"] == 0.5
    assert scores["tests"] == 1.0
    assert scores["security"] == 0.0
    # overall = 2 True / 4 total
    assert scores["overall"] == 0.5


def test_score_verdicts_all_true_is_one():
    verdicts = [
        Verdict(dimension=d, question="q", answer=True, explanation="")
        for d in ("correctness", "tests", "clarity")
    ]
    scores = score_verdicts(verdicts)
    assert scores["overall"] == 1.0
    assert scores["correctness"] == 1.0
