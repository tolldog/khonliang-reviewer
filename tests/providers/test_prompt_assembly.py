"""Tests for the repo-prompts merge layer in ``reviewer.providers._prompt``.

These cover the pure function shape — given a :class:`RepoPrompts`
snapshot and an ``example_format``, does the assembled prompt contain
the right pieces in the right order with the right framing? Integration
with the agent's load-and-thread path is covered in
``test_agent_skills.py``; this file is strictly about the assembly
primitive.
"""

from __future__ import annotations

from khonliang_reviewer import ReviewRequest

from reviewer.config.prompts import RepoPrompts
from reviewer.providers._prompt import build_review_prompt


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
