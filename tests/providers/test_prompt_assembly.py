"""Tests for the repo-prompts merge layer in ``reviewer.providers._prompt``.

These cover the pure function shape — given a :class:`RepoPrompts`
snapshot and an ``example_format``, does the assembled prompt contain
the right pieces in the right order with the right framing? Integration
with the agent's load-and-thread path is covered in
``test_agent_skills.py``; this file is strictly about the assembly
primitive.
"""

from __future__ import annotations

import logging

from khonliang_reviewer import ReviewRequest

from reviewer.config.prompts import RepoPrompts
from reviewer.providers._prompt import (
    BINARY_QUESTION_DIMENSIONS,
    REVIEW_RESPONSE_SCHEMA,
    build_review_prompt,
    classify_diff_content,
    review_response_schema,
)


# -- no repo prompts = pre-FR bytes -----------------------------------


def test_no_repo_prompts_matches_pre_fr_shape():
    """``repo_prompts=None`` reproduces the exact pre-FR prompt bytes.

    Regression guard: callers (tests, providers without repo hints)
    must get identical output before and after the FR lands.
    """
    request = ReviewRequest(kind="pr_diff", content="diff body")
    got = build_review_prompt(request, include_schema=False)

    assert "## Repository System Preamble" not in got
    assert "## Severity Rubric" not in got
    assert "## Examples" not in got
    # Task content still lands at the tail.
    assert got.rstrip().endswith("diff body")


def test_empty_repo_prompts_matches_pre_fr_shape():
    """An explicitly-empty ``RepoPrompts()`` also produces pre-FR bytes.

    The loader's ``is_empty`` shortcut returns True for the default
    snapshot; the prompt assembler must honour it and not emit an
    empty section header.
    """
    request = ReviewRequest(kind="pr_diff", content="x")
    no_prompts = build_review_prompt(request)
    with_empty = build_review_prompt(request, repo_prompts=RepoPrompts())
    assert no_prompts == with_empty


# -- content injection -----------------------------------------------


def test_severity_rubric_appears_in_prompt():
    """AC: rubric text lands in the assembled prompt verbatim."""
    rp = RepoPrompts(severity_rubric="BEHOLD THE RUBRIC")
    prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content="x"),
        repo_prompts=rp,
    )
    assert "BEHOLD THE RUBRIC" in prompt
    assert "## Severity Rubric" in prompt


def test_system_preamble_appears_in_prompt():
    rp = RepoPrompts(system_preamble="BE GENTLE")
    prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content="x"),
        repo_prompts=rp,
    )
    assert "BE GENTLE" in prompt
    assert "## Repository System Preamble" in prompt


# -- per-kind example filtering --------------------------------------


def test_examples_filtered_by_kind():
    """AC: a pr_diff review does not see spec examples."""
    rp = RepoPrompts(
        examples={
            ("pr_diff", "nit"): "DIFF NIT EXAMPLE",
            ("spec", "nit"): "SPEC NIT EXAMPLE",
        }
    )
    pr_prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content="x"),
        repo_prompts=rp,
    )
    assert "DIFF NIT EXAMPLE" in pr_prompt
    assert "SPEC NIT EXAMPLE" not in pr_prompt

    spec_prompt = build_review_prompt(
        ReviewRequest(kind="spec", content="x"),
        repo_prompts=rp,
    )
    assert "SPEC NIT EXAMPLE" in spec_prompt
    assert "DIFF NIT EXAMPLE" not in spec_prompt


# -- vendor wrapping --------------------------------------------------


def test_xml_wrapping_for_anthropic_style():
    """AC: ``example_format='xml'`` wraps examples in ``<example>`` tags."""
    rp = RepoPrompts(examples={("pr_diff", "concern"): "RACE COND"})
    prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content="x"),
        repo_prompts=rp,
        example_format="xml",
    )
    assert '<example severity="concern">' in prompt
    assert "</example>" in prompt
    assert "RACE COND" in prompt


def test_json_wrapping_for_openai_style():
    """AC: ``example_format='json'`` emits a JSON payload per example."""
    rp = RepoPrompts(examples={("pr_diff", "nit"): "TRAILING WHITESPACE"})
    prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content="x"),
        repo_prompts=rp,
        example_format="json",
    )
    # The JSON payload carries both severity and example — check both
    # to catch a regression where one is dropped from the JSON shape.
    assert '"severity": "nit"' in prompt
    assert '"example": "TRAILING WHITESPACE"' in prompt
    assert "```json" in prompt


def test_markdown_wrapping_is_default():
    """AC: missing / unknown ``example_format`` → markdown fence framing."""
    rp = RepoPrompts(examples={("pr_diff", "comment"): "NAMING ISSUE"})
    prompt_no_fmt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content="x"),
        repo_prompts=rp,
    )
    assert "### comment" in prompt_no_fmt
    assert "```\nNAMING ISSUE\n```" in prompt_no_fmt

    # And an unknown format string also falls back to markdown.
    prompt_unknown = build_review_prompt(
        ReviewRequest(kind="pr_diff", content="x"),
        repo_prompts=rp,
        example_format="yamlish-mystery-format",
    )
    assert "### comment" in prompt_unknown
    assert "```\nNAMING ISSUE\n```" in prompt_unknown


def test_json_wrapping_handles_embedded_quotes():
    """A hand-rolled ``{"example": "..."}`` would break on embedded quotes.

    Regression guard: pin that ``json.dumps`` is the escape path.
    """
    rp = RepoPrompts(
        examples={("pr_diff", "nit"): 'line with "quoted" phrase and \\ slash'}
    )
    prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content="x"),
        repo_prompts=rp,
        example_format="json",
    )
    # The embedded quote is escaped; the slash is re-escaped. Both
    # come out valid JSON when the wrapping uses json.dumps.
    assert '\\"quoted\\"' in prompt
    assert "\\\\" in prompt


# -- merge ordering ---------------------------------------------------


def test_merge_order_is_system_then_rubric_then_examples_then_content():
    """AC: assembly order matches the FR (system → rubric → examples → task)."""
    rp = RepoPrompts(
        system_preamble="SYS",
        severity_rubric="RUB",
        examples={("pr_diff", "nit"): "EX"},
    )
    prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content="CONTENT_MARKER"),
        repo_prompts=rp,
    )
    # Extract the byte offset of each anchor and assert ordering.
    sys_at = prompt.index("SYS")
    rub_at = prompt.index("RUB")
    ex_at = prompt.index("EX")
    content_at = prompt.index("CONTENT_MARKER")
    assert sys_at < rub_at < ex_at < content_at


def test_examples_emitted_in_severity_order():
    """AC: examples appear in ``(nit, comment, concern)`` order.

    File-write / dict-iteration order must not leak into the prompt
    — the order-stable rendering makes the merged prompt byte-stable
    across reruns of the same base SHA, which matters for prompt
    caching and for diffing two prompts in dogfood sessions.
    """
    rp = RepoPrompts(
        examples={
            ("pr_diff", "concern"): "C_EX",
            ("pr_diff", "nit"): "N_EX",
            ("pr_diff", "comment"): "M_EX",
        }
    )
    prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content="x"),
        repo_prompts=rp,
    )
    n_at = prompt.index("N_EX")
    m_at = prompt.index("M_EX")
    c_at = prompt.index("C_EX")
    assert n_at < m_at < c_at


def test_repo_prompts_land_before_schema():
    """Repo prompts land before the schema block when the schema is inline.

    Order matters for prompt caching: the response-schema section is
    large and mostly static; repo prompts are repo-specific. Putting
    repo prompts before the schema keeps the per-repo prefix as
    compact as possible for models that support KV-cache reuse.
    """
    rp = RepoPrompts(severity_rubric="RUB")
    prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content="x"),
        repo_prompts=rp,
        include_schema=True,
    )
    rub_at = prompt.index("RUB")
    schema_at = prompt.index("## Response Schema")
    assert rub_at < schema_at


# -- direct-caller fail-open for _render_repo_prompts -----------------


def test_render_repo_prompts_none_returns_empty():
    """Direct call with ``None`` → empty list (pre-existing behaviour).

    Regression guard for the ``None`` branch of the fail-open defense
    so re-ordering the checks doesn't accidentally drop this case.
    """
    from reviewer.providers._prompt import _render_repo_prompts

    assert _render_repo_prompts(None, kind="pr_diff", example_format=None) == []


def test_render_repo_prompts_non_repoprompts_value_returns_empty():
    """Direct call with a non-``RepoPrompts`` value must not crash.

    The agent-layer reserved-metadata strip is the primary defense
    against a caller injecting an arbitrary value into
    ``metadata["_khonliang_repo_prompts"]``. This test covers the
    **fallback** — a provider invoked directly (test, library consumer,
    future call path) with a bogus ``repo_prompts`` argument must
    fail-open to an empty merge section rather than raising.

    A crash here would propagate up through :func:`build_review_prompt`
    and fail the entire review, which is the opposite of the
    graceful-absence contract the rest of the ``.reviewer/`` loader
    carries.
    """
    from reviewer.providers._prompt import _render_repo_prompts

    # str: has no ``.is_empty`` attribute — would AttributeError pre-fix.
    assert (
        _render_repo_prompts("evil-string", kind="pr_diff", example_format=None)
        == []
    )
    # int: also no ``.is_empty``.
    assert _render_repo_prompts(42, kind="pr_diff", example_format=None) == []
    # dict: would respond to ``.is_empty`` as an attribute lookup miss,
    # still AttributeError pre-fix.
    assert (
        _render_repo_prompts(
            {"system_preamble": "ignored"}, kind="pr_diff", example_format=None
        )
        == []
    )


def test_build_review_prompt_survives_non_repoprompts_value():
    """End-to-end: ``build_review_prompt`` with a bogus ``repo_prompts``
    value doesn't raise, and the output has no repo-prompts section.

    This is the direct-caller path analogue of
    ``test_review_text_strips_reserved_khonliang_metadata_keys`` in
    ``test_agent_skills.py``. The agent strip is the primary defense
    on the bus path; this is the belt-and-braces guarantee for
    callers that bypass the agent entirely.
    """
    request = ReviewRequest(kind="pr_diff", content="diff body")
    got = build_review_prompt(request, repo_prompts="evil-string")  # type: ignore[arg-type]

    assert "## Repository System Preamble" not in got
    assert "## Severity Rubric" not in got
    assert "## Examples" not in got
    assert got.rstrip().endswith("diff body")


# -- doc-hunk routing (fr_reviewer_1262ce18) --------------------------


_MD_DIFF = "+++ b/README.md\n@@ -1 +1 @@\n+# Title\n+Some clarifying prose here.\n"
_CODE_DIFF = "+++ b/a.py\n@@ -1 +2 @@\n+def f():\n+    return compute()\n"
_PREPROC_DIFF = (
    "+++ b/a.c\n@@ -1 +5 @@\n"
    "+#include <stdio.h>\n+#define MAX 10\n+#ifndef FOO\n+int main(void){return 0;}\n"
)
# C/C++ also permits whitespace between "#" and the directive keyword.
_PREPROC_SPACED_DIFF = (
    "+++ b/a.c\n@@ -1 +5 @@\n"
    "+# include <stdio.h>\n+# define MAX 10\n+# ifndef FOO\n+int main(void){return 0;}\n"
)
_COMMENT_DIFF = "+++ b/a.py\n@@ -1 +3 @@\n+# explain the why\n+# more rationale\n+# and more\n"
# Comment styles without a space (or with a tab) after the "#" — still doc.
_NOSPACE_COMMENT_DIFF = (
    "+++ b/a.py\n@@ -1 +3 @@\n+#explain the why\n+#\tmore rationale\n+#and more\n"
)
_MIXED_DIFF = "+++ b/a.py\n@@ -1 +2 @@\n+# a comment\n+def f(): return real_work()\n"


def test_classify_doc_file_diff_is_doc():
    assert classify_diff_content(_MD_DIFF) == "doc"


def test_classify_code_diff_is_code():
    assert classify_diff_content(_CODE_DIFF) == "code"


def test_classify_comment_heavy_code_file_is_doc():
    assert classify_diff_content(_COMMENT_DIFF) == "doc"


def test_classify_nospace_and_tab_comments_are_doc():
    # "#comment" (no space) and "#\tcomment" are comments, not code —
    # only shebangs and preprocessor directives escape the "#" doc rule.
    assert classify_diff_content(_NOSPACE_COMMENT_DIFF) == "doc"


def test_classify_shebang_line_is_not_doc():
    # A shebang is code, not a comment; a lone shebang diff is not doc-heavy.
    shebang = "+++ b/run.sh\n@@ -1 +2 @@\n+#!/usr/bin/env bash\n+echo hi\n"
    assert classify_diff_content(shebang) == "code"


def test_classify_mixed_diff_is_mixed():
    assert classify_diff_content(_MIXED_DIFF) == "mixed"


def test_classify_c_preprocessor_directives_are_code():
    # "#include"/"#define"/"#ifndef" are C preprocessor directives, not
    # comments — they must not be mistaken for documentation.
    assert classify_diff_content(_PREPROC_DIFF) == "code"


def test_classify_spaced_c_preprocessor_directives_are_code():
    # C/C++ allows whitespace after "#": "# include" is as valid as
    # "#include" and must still classify as code, not doc.
    assert classify_diff_content(_PREPROC_SPACED_DIFF) == "code"


def test_classify_non_diff_and_empty_are_code():
    assert classify_diff_content("just some text, no diff markers") == "code"
    assert classify_diff_content("") == "code"


def test_doc_heavy_diff_gets_critique_instruction():
    prompt = build_review_prompt(ReviewRequest(kind="pr_diff", content=_MD_DIFF))
    assert "CRITIQUE, do not summarize" in prompt
    assert "paraphrases the change is not a finding" in prompt


def test_code_diff_omits_critique_instruction():
    prompt = build_review_prompt(ReviewRequest(kind="pr_diff", content=_CODE_DIFF))
    assert "CRITIQUE, do not summarize" not in prompt


# -- universal review-discipline calibration (fr_reviewer_ff923ebf) ---


def test_review_discipline_instruction_present_for_code_diff():
    """The discipline calibration must fire on a plain CODE diff — both
    dogfood false-positive cases (echo, inverted-claim) were code diffs where
    the doc/artifact instructions don't fire."""
    prompt = build_review_prompt(ReviewRequest(kind="pr_diff", content=_CODE_DIFF))
    assert "REVIEW DISCIPLINE" in prompt
    # Anti-echo, anti-invert, and severity-discipline clauses all present.
    assert "restate, paraphrase, or summarize" in prompt
    assert "opposite of what the diff does" in prompt
    # Severity discipline forbids style→concern but does NOT cap what counts as
    # blocking (so it doesn't downgrade perf regressions or rubric concerns).
    assert "genuinely blocking defect" in prompt
    assert "Do NOT raise subjective naming, style" in prompt


def test_applied_guard_clause_present_for_code_diff():
    """Clause 4 (fr_reviewer_8d261d32): the 'evaluate AS APPLIED' calibration
    curbs the imagined-pre-state FP (an added guard flagged as the missing
    requirement / the bug — dog_0a178955). It rides on the same rubric-less
    hot-tier discipline path as clauses 1-3, so it fires on a code diff."""
    prompt = build_review_prompt(ReviewRequest(kind="pr_diff", content=_CODE_DIFF))
    assert "Evaluate the diff AS APPLIED" in prompt
    assert "IS the fix" in prompt
    assert "not an imagined version without the change" in prompt


def test_applied_guard_clause_absent_for_artifact_kinds():
    """Clause 4 is scoped exactly like the rest of _REVIEW_DISCIPLINE_INSTRUCTION
    — excluded from artifact reviews (fr/spec/milestone), which carry their own
    framing + rubric."""
    for artifact_kind in ("fr", "spec", "milestone"):
        prompt = build_review_prompt(
            ReviewRequest(kind=artifact_kind, content="# doc\n")
        )
        assert "Evaluate the diff AS APPLIED" not in prompt, artifact_kind


def test_review_discipline_absent_for_artifact_kinds():
    """Scoping regression guard (codex round 3): the code-change-framed
    discipline must NOT apply to artifact reviews (fr/spec/milestone) — those
    carry their own _ARTIFACT_REVIEW_INSTRUCTION + rubric, which own their
    severity. Applying the code-diff wording biases the model against the
    rubrics' legitimate planning-integrity concerns."""
    for artifact_kind in ("fr", "spec", "milestone"):
        prompt = build_review_prompt(
            ReviewRequest(kind=artifact_kind, content="# doc\n")
        )
        assert "REVIEW DISCIPLINE" not in prompt, artifact_kind
        # The artifact framing instruction IS still present.
        assert "complete planning artifact" in prompt, artifact_kind


def test_review_discipline_present_for_non_artifact_non_diff_kind():
    """The discipline covers the rubric-less hot-tier kinds, not just pr_diff —
    e.g. a `doc` / `pr_description` review (no rubric of its own)."""
    prompt = build_review_prompt(ReviewRequest(kind="doc", content="some text"))
    assert "REVIEW DISCIPLINE" in prompt


def test_severity_discipline_does_not_cap_blocking_categories():
    """Regression guard (codex P1, round 2): on the code-diff path the severity
    clause must NOT enumerate/cap what counts as blocking — a code diff's
    perf/resource regression is blocking too, so the clause is framed negatively
    (forbid style→concern only) keeping 'genuinely blocking defect' open.
    (Round 1, artifact rubric concerns, is now handled by scoping the
    instruction out of artifacts entirely — see
    test_review_discipline_absent_for_artifact_kinds.)"""
    code_prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content=_CODE_DIFF)
    )
    assert "genuinely blocking defect" in code_prompt
    # Must NOT cap concern to correctness/security (the round-2 regression).
    assert "for correctness or security" not in code_prompt


def test_severity_discipline_yields_to_repo_severity_rubric():
    """Regression guard (codex round 4): the built-in discipline must defer to a
    repo-provided severity_rubric (the operator's calibration feature) rather
    than silently contradicting it. The instruction states the precedence, and
    the repo rubric renders below it in the same prompt."""
    rp = RepoPrompts(severity_rubric="treat naming nits as concern here")
    prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content=_CODE_DIFF), repo_prompts=rp
    )
    assert "follow ITS calibration wherever it differs" in prompt
    assert "## Severity Rubric" in prompt
    # Precedence reads correctly: discipline first, repo rubric after.
    assert prompt.index("REVIEW DISCIPLINE") < prompt.index("## Severity Rubric")


def test_review_discipline_precedes_kind_routing():
    """Discipline calibration is in the base prompt, before the kind-specific
    doc routing — so it frames every finding on the non-artifact path."""
    prompt = build_review_prompt(ReviewRequest(kind="pr_diff", content=_MD_DIFF))
    assert prompt.index("REVIEW DISCIPLINE") < prompt.index("CRITIQUE, do not summarize")


# -- built-in anti-examples (fr_reviewer_ff923ebf b) ------------------


def test_builtin_anti_examples_loads_from_package():
    """The packaged anti_examples.md is loadable (wheel package-data covers
    reviewer/data/prompts/*.md)."""
    from reviewer.providers._prompt import _builtin_anti_examples

    text = _builtin_anti_examples()
    assert text is not None
    assert "echo-as-finding" in text
    assert "inverted claim" in text.lower()


def test_anti_examples_present_for_code_diff_after_discipline():
    """Concrete anti-examples render on the rubric-less hot-tier path, after the
    prose discipline they reinforce."""
    prompt = build_review_prompt(ReviewRequest(kind="pr_diff", content=_CODE_DIFF))
    assert "Anti-Examples" in prompt
    # The two dogfood patterns are present.
    assert "echo-as-finding" in prompt
    assert "inverted claim" in prompt.lower()
    # Ordering: prose discipline first, then the concrete anti-examples.
    assert prompt.index("REVIEW DISCIPLINE") < prompt.index("Anti-Examples")


def test_anti_examples_present_for_non_artifact_non_diff_kind():
    prompt = build_review_prompt(ReviewRequest(kind="doc", content="some text"))
    assert "Anti-Examples" in prompt


def test_anti_examples_absent_for_artifact_kinds():
    """Anti-examples are scoped like the discipline — excluded from artifact
    reviews, which carry their own rubric and whole-document framing."""
    for artifact_kind in ("fr", "spec", "milestone"):
        prompt = build_review_prompt(
            ReviewRequest(kind=artifact_kind, content="# doc\n")
        )
        assert "Anti-Examples" not in prompt, artifact_kind


def test_mixed_diff_omits_critique_instruction():
    # Only clearly doc-heavy diffs are routed; mixed is conservative.
    prompt = build_review_prompt(ReviewRequest(kind="pr_diff", content=_MIXED_DIFF))
    assert "CRITIQUE, do not summarize" not in prompt


def test_pr_diff_logs_diff_classification(caplog):
    with caplog.at_level(logging.DEBUG, logger="reviewer.providers._prompt"):
        build_review_prompt(ReviewRequest(kind="pr_diff", content=_MIXED_DIFF))

    assert "diff classification (kind=pr_diff): mixed" in caplog.text


def test_classify_added_line_starting_with_plus_not_a_header():
    # An ADDED line whose content begins with "+++" renders as "++++..." and
    # must NOT be mistaken for a "+++ " file header (Copilot PR #47).
    diff = "+++ b/a.py\n@@ -1 +2 @@\n++++ not a header, real code\n+x = 1\n"
    assert classify_diff_content(diff) == "code"


def test_doc_routing_gated_on_pr_diff_kind():
    # A doc-classified payload under a non-diff kind is NOT routed (those go
    # through the artifact-review pipeline, not doc-hunk routing).
    prompt = build_review_prompt(ReviewRequest(kind="spec", content=_MD_DIFF))
    assert "CRITIQUE, do not summarize" not in prompt


# -- artifact review (fr_reviewer_19c871ab) ---------------------------


def test_artifact_kinds_get_full_doc_instruction_and_builtin_rubric():
    for kind in ("fr", "spec", "milestone"):
        prompt = build_review_prompt(ReviewRequest(kind=kind, content="# Doc\n\nbody"))
        # full-document framing, not diff framing
        assert "complete planning artifact" in prompt
        assert "anchor to a named section" in prompt
        # packaged built-in rubric injected under a per-kind header
        assert f"## {kind.capitalize()} Rubric" in prompt


def test_artifact_repo_rubric_override_wins_over_builtin():
    rp = RepoPrompts(kind_rubrics={"spec": "REPO-SPECIFIC SPEC RUBRIC TEXT"})
    prompt = build_review_prompt(
        ReviewRequest(kind="spec", content="# Spec\n\nbody"), repo_prompts=rp
    )
    assert "REPO-SPECIFIC SPEC RUBRIC TEXT" in prompt
    assert "## Spec Rubric" in prompt


def test_pr_diff_has_no_artifact_instruction():
    prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content="+++ b/a.py\n@@ -1 +1 @@\n+x = 1\n")
    )
    assert "complete planning artifact" not in prompt


# -- region-sweep mode (fr_khonliang-reviewer_8fb20f1f) ---------------


def test_region_sweep_instruction_present_when_on():
    """region_sweep=True injects the anti-cascade sweep instruction on a
    non-artifact (code-diff) review."""
    prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content=_CODE_DIFF), region_sweep=True
    )
    assert "REGION-SWEEP MODE" in prompt
    assert "EVERY" in prompt
    assert "single pass" in prompt


def test_region_sweep_instruction_absent_when_off():
    """Default (region_sweep=False) leaves the sweep instruction out — and the
    prompt bytes are byte-identical to omitting the arg entirely."""
    default_prompt = build_review_prompt(ReviewRequest(kind="pr_diff", content=_CODE_DIFF))
    off_prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content=_CODE_DIFF), region_sweep=False
    )
    assert "REGION-SWEEP MODE" not in default_prompt
    assert "REGION-SWEEP MODE" not in off_prompt
    # region_sweep=False must reproduce the exact pre-FR bytes.
    assert default_prompt == off_prompt


def test_region_sweep_absent_for_non_pr_diff_kinds_even_when_on():
    """Scoping guard (codex round 2): region_sweep is gated to ``pr_diff`` —
    the instruction is diff-shaped (paths/lines/hunks). It must NOT appear on
    artifact kinds (no call-site-region shape) OR on the other non-diff
    hot-tier kinds like doc / pr_description (plain prose, no hunks/paths/lines
    — injecting it risks hallucinated locations), even when the mode is on."""
    for kind in ("fr", "spec", "milestone", "doc", "pr_description"):
        prompt = build_review_prompt(
            ReviewRequest(kind=kind, content="# doc\n"),
            region_sweep=True,
        )
        assert "REGION-SWEEP MODE" not in prompt, kind


# -- binary-questions mode (fr_khonliang-reviewer_a585ea3d) ----------


def test_binary_questions_section_present_when_on():
    """binary_questions=True injects the BinEval section and every one of the
    fixed dimensions on a pr_diff review."""
    prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content=_CODE_DIFF), binary_questions=True
    )
    assert "BINARY-QUESTIONS MODE" in prompt
    assert '"verdicts"' in prompt or "verdicts" in prompt
    # All six fixed dimensions must be listed.
    for dimension, question in BINARY_QUESTION_DIMENSIONS:
        assert dimension in prompt, dimension
        assert question in prompt, dimension
    # The confirmed dimension set is exactly these six.
    assert [d for d, _ in BINARY_QUESTION_DIMENSIONS] == [
        "correctness",
        "security",
        "error_handling",
        "tests",
        "performance",
        "clarity",
    ]


def test_binary_questions_section_absent_when_off_byte_identical():
    """Default (binary_questions=False) leaves the section out — and the prompt
    bytes are byte-identical to omitting the arg entirely."""
    default_prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content=_CODE_DIFF)
    )
    off_prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content=_CODE_DIFF), binary_questions=False
    )
    assert "BINARY-QUESTIONS MODE" not in default_prompt
    assert "BINARY-QUESTIONS MODE" not in off_prompt
    assert default_prompt == off_prompt


def test_binary_questions_absent_for_non_pr_diff_kinds_even_when_on():
    """Scoped to pr_diff (the questions evaluate a diff). Must NOT appear on
    artifact kinds or the other non-diff hot-tier kinds, even when on."""
    for kind in ("fr", "spec", "milestone", "doc", "pr_description"):
        prompt = build_review_prompt(
            ReviewRequest(kind=kind, content="# doc\n"),
            binary_questions=True,
        )
        assert "BINARY-QUESTIONS MODE" not in prompt, kind


def test_binary_questions_schema_appears_inline_when_included():
    """When include_schema is set (ollama/copilot path), the emitted schema
    carries the verdicts array in binary-questions mode."""
    prompt = build_review_prompt(
        ReviewRequest(kind="pr_diff", content=_CODE_DIFF),
        include_schema=True,
        binary_questions=True,
    )
    assert '"verdicts"' in prompt
    # Holistic include_schema path has no verdicts key.
    holistic = build_review_prompt(
        ReviewRequest(kind="pr_diff", content=_CODE_DIFF),
        include_schema=True,
        binary_questions=False,
    )
    assert '"verdicts"' not in holistic


def test_review_response_schema_holistic_is_the_constant():
    """review_response_schema(False) IS the unchanged holistic constant."""
    assert review_response_schema(False) == REVIEW_RESPONSE_SCHEMA
    # Same object identity — nothing copies/mutates the holistic schema.
    assert review_response_schema(False) is REVIEW_RESPONSE_SCHEMA


def test_review_response_schema_binary_has_verdicts_without_mutating_constant():
    """review_response_schema(True) adds a verdicts array and does NOT mutate
    the shared holistic constant."""
    binary = review_response_schema(True)
    assert "verdicts" in binary["properties"]
    item = binary["properties"]["verdicts"]["items"]
    assert item["required"] == ["dimension", "question", "answer", "explanation"]
    assert item["properties"]["answer"]["type"] == "boolean"
    # verdicts is REQUIRED in the binary variant so an enforced backend can't
    # silently degrade to the holistic shape (codex PR B review P2).
    assert binary["required"] == ["summary", "verdicts"]
    # Constrained to the fixed dimension set with one item per dimension
    # (codex PR B R4 P2): empty / underfilled / invented-dimension arrays are
    # rejected by schema-enforced backends instead of skewing score_verdicts.
    from reviewer.providers._prompt import BINARY_QUESTION_DIMENSIONS

    expected_dims = [d for d, _ in BINARY_QUESTION_DIMENSIONS]
    assert item["properties"]["dimension"]["enum"] == expected_dims
    assert binary["properties"]["verdicts"]["minItems"] == len(expected_dims)
    assert binary["properties"]["verdicts"]["maxItems"] == len(expected_dims)
    # The holistic constant must be untouched by building the variant —
    # including the item fragment the variant deep-copies before mutating.
    assert "verdicts" not in REVIEW_RESPONSE_SCHEMA["properties"]
    assert REVIEW_RESPONSE_SCHEMA["required"] == ["summary"]
    from reviewer.providers._prompt import _VERDICT_ITEM_SCHEMA

    assert "enum" not in _VERDICT_ITEM_SCHEMA["properties"]["dimension"]


def test_binary_questions_schema_not_emitted_inline_for_non_pr_diff_kind():
    """Schema gate mirrors the instruction gate: binary_questions=True on a
    non-pr_diff kind must NOT emit a verdicts schema (the kind never sees the
    binary-questions instructions). codex PR B review P2."""
    for kind in ("fr", "spec", "milestone", "doc", "pr_description"):
        prompt = build_review_prompt(
            ReviewRequest(kind=kind, content="# doc\n"),
            include_schema=True,
            binary_questions=True,
        )
        assert '"verdicts"' not in prompt, kind
