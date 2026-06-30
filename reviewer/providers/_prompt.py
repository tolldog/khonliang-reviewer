"""Shared review-prompt assembly + JSON-schema contract.

Every concrete :class:`khonliang_reviewer.ReviewProvider` in this repo
builds the same prompt shape from a :class:`ReviewRequest` and asks the
backend to return the same JSON object shape. Keeping the prompt and the
schema here means adding a new backend adapter does not re-invent either
— it just picks the transport.

The JSON-schema matches the subset of :class:`ReviewResult` +
:class:`ReviewFinding` fields a model can usefully produce. Other fields
(``backend``, ``model``, ``usage``, ``disposition``, ``error`` etc.) are
set by the provider, not the model.

Repo-side prompt merging
------------------------
Repos opt in to few-shot examples + severity-rubric calibration by
shipping a ``.reviewer/prompts/`` directory. When the caller threads a
:class:`reviewer.config.prompts.RepoPrompts` snapshot plus the active
model's ``example_format`` into :func:`build_review_prompt`, the merge
runs in this order:

    built-in reviewer system prompt
      → repo ``system_preamble.md``
      → ``severity_rubric.md``
      → ``examples/<kind>/*`` (wrapped per vendor's ``example_format``)
      → task content (schema, instructions, context, content)

The wrapping step is the one place this module consults the model
config: ``example_format: xml|json|markdown`` decides how each example
block gets framed. Same raw content, different framing per vendor —
Anthropic reads XML better, OpenAI-schema models read JSON better,
everything else falls back to markdown fences.
"""

from __future__ import annotations

import functools
import importlib.resources
import json
import logging
import re
from typing import TYPE_CHECKING, Any

from khonliang_reviewer import ReviewRequest

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Imported for type hints only — keeping the runtime import out of
    # the hot path avoids a circular-import risk if config.prompts ever
    # needs to import anything provider-side.
    from reviewer.config.prompts import RepoPrompts


REVIEW_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["summary"],
    "properties": {
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["severity", "title", "body"],
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["nit", "comment", "concern"],
                    },
                    "category": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "path": {"type": ["string", "null"]},
                    "line": {"type": ["integer", "null"]},
                    "section": {"type": ["string", "null"]},
                    "suggestion": {"type": ["string", "null"]},
                },
            },
        },
    },
}


#: Supported values for ``example_format`` in model config. Anything
#: else (including ``None``) falls back to ``"markdown"`` — the safest
#: framing that any model understands without vendor-specific
#: tokenization sensitivity.
_SUPPORTED_EXAMPLE_FORMATS: frozenset[str] = frozenset({"xml", "json", "markdown"})
_DEFAULT_EXAMPLE_FORMAT = "markdown"


# ---------------------------------------------------------------------------
# Doc-hunk routing (fr_reviewer_1262ce18)
# ---------------------------------------------------------------------------
#: File extensions whose changed lines are treated as documentation/prose.
_DOC_FILE_EXTS: tuple[str, ...] = (".md", ".markdown", ".rst", ".txt", ".adoc")
#: Stripped-line prefixes that mark a comment/doc line inside a code file.
#: Deliberately excludes ``*`` and ``;`` — both start real code lines (C
#: pointer deref / block-comment-continuation; Lisp/asm vs C statements) so
#: they'd misclassify code as doc. ``#`` is handled separately in
#: :func:`_is_doc_line` (a C/C++ preprocessor directive vs a comment).
_COMMENT_PREFIXES: tuple[str, ...] = ("//", "/*", "--", "<!--", '"""', "'''")

#: C/C++ preprocessor directive keywords. A ``#`` line whose first token is
#: one of these is code, not a comment — even with whitespace after the
#: ``#`` (``# include`` is as valid as ``#include``). Trade-off: a prose
#: comment that opens with one of these words (``# define the term``) is
#: counted as code. That's symmetric and rare, and :func:`classify_diff_content`
#: routes on a ratio, so a few miscounted lines don't flip a hunk's verdict.
_PREPROCESSOR_DIRECTIVES: frozenset[str] = frozenset(
    {
        "include", "define", "undef", "if", "ifdef", "ifndef", "else",
        "elif", "endif", "pragma", "error", "warning", "line", "import",
    }
)
#: Matches the first lowercase token after a ``#`` and optional whitespace.
_HASH_DIRECTIVE_RE = re.compile(r"#\s*([a-z_]+)")


def _is_doc_line(line: str) -> bool:
    """True iff a stripped, added line reads as documentation / prose.

    Blank lines, ``#``-comments (Python / shell / yaml, with or without a
    space after the ``#``) or markdown ATX headings, and the
    :data:`_COMMENT_PREFIXES` markers count as doc. The two ``#`` forms that
    are *code*: a shebang (``#!...``) and a C/C++ preprocessor directive whose
    first token is in :data:`_PREPROCESSOR_DIRECTIVES` — covering both
    ``#include`` and the whitespace-after-hash form ``# include``.
    """
    if not line:
        return True
    if line.startswith("#"):
        if line.startswith("#!"):  # shebang — code, not a comment
            return False
        m = _HASH_DIRECTIVE_RE.match(line)
        if m and m.group(1) in _PREPROCESSOR_DIRECTIVES:
            return False
        return True  # any other "#..." is a comment / ATX heading
    return line.startswith(_COMMENT_PREFIXES)

#: Appended to the prompt for doc-heavy reviews. Small local hot-tier models
#: cannot distinguish "summarize" from "critique" on prose, so they echo the
#: changed text back as a "finding" (the dogfood prose-echo pattern). This
#: instruction redirects them; the deeper fix (curated doc-review few-shots)
#: lands with the examples library (fr_reviewer_afd4bab1).
_DOC_REVIEW_INSTRUCTION = (
    "NOTE: this change is predominantly documentation / comments / prose. "
    "CRITIQUE, do not summarize. Only flag a finding if the prose is "
    "factually WRONG, misleading, contradicts the code, or omits a stated "
    "invariant. NEVER restate, rephrase, or summarize what the text says — a "
    "finding that merely paraphrases the change is not a finding. If the "
    "prose is accurate, return zero findings."
)


# ---------------------------------------------------------------------------
# Artifact review (fr_reviewer_19c871ab)
# ---------------------------------------------------------------------------
#: Planning-artifact review kinds. These carry a whole document (an FR, a
#: spec, a milestone) — not a diff — so they skip the diff-hunk classifier and
#: get a full-document framing instruction plus a per-kind rubric.
_ARTIFACT_KINDS: frozenset[str] = frozenset({"fr", "spec", "milestone"})

#: Universal review-discipline instruction injected into every review,
#: regardless of kind. Targets the recurring hot-tier false-positive
#: patterns (fr_reviewer_ff923ebf): small local models echo the code's own
#: prose back as "findings" (dog_9902f2f9), invert the change's intent
#: (dog_3891542a flagged "Code Repetition" as a *concern* on a change that
#: CONSOLIDATES duplicate literals), and over-use `concern` severity — which,
#: via the literal sign_off_trailer mapping, flips an otherwise-clean review
#: to `concerns-raised`. The `_DOC_REVIEW_INSTRUCTION` only fires for
#: doc-heavy diffs and the artifact instruction only for whole-document
#: artifacts; both dogfood cases were *code* diffs where neither fired. This
#: calibration is injected for the rubric-less hot-tier kinds (pr_diff, doc,
#: pr_description) and EXCLUDED for the artifact kinds (fr/spec/milestone),
#: which carry their own framing + rubric — see the gate in
#: :func:`build_review_prompt`.
#:
#: The severity-discipline clause is framed *negatively* on purpose: it forbids
#: raising subjective style/naming/formatting/calibration remarks to `concern`
#: (the dog_3891542a pattern) but does NOT enumerate what counts as blocking.
#: Enumerating a positive list is what two review rounds caught as wrong — any
#: omitted-but-legitimate blocking category (the artifact rubrics'
#: planning-integrity concerns like duplicate_of_existing_fr; a severe
#: performance/resource regression on a plain code diff) would get silently
#: downgraded, flipping the verdict away from `concerns-raised`. "Genuinely
#: blocking defect" keeps all of those eligible while still curbing the
#: style-as-concern false positive.
_REVIEW_DISCIPLINE_INSTRUCTION = (
    "REVIEW DISCIPLINE — applies to every finding:\n"
    "1. A finding must assert something the change gets WRONG — a bug, a "
    "risk, an omission. Do NOT restate, paraphrase, or summarize the code's "
    "own comments, docstrings, identifiers, or the diff's stated intent as a "
    "finding; describing what the change does is not a critique of it.\n"
    "2. Before flagging, confirm the change actually exhibits the problem you "
    "describe. Never assert the opposite of what the diff does — e.g. do not "
    "flag 'duplication' on a change that REMOVES or consolidates duplication.\n"
    "3. Severity discipline: reserve 'concern' for a genuinely blocking defect "
    "— a real bug, regression, security hole, or any other problem that should "
    "stop the change. Do NOT raise subjective naming, style, formatting, or "
    "calibration preferences to 'concern'; those are 'comment' or 'nit'. If "
    "you are unsure a finding is both real and blocking, lower its severity or "
    "omit it."
)


#: Prepended for artifact reviews. The reviewer is reading a whole planning
#: document, not a code change, so findings are holistic and anchor to a named
#: SECTION (e.g. "§Acceptance Criteria"), not a line number.
_ARTIFACT_REVIEW_INSTRUCTION = (
    "NOTE: this is a complete planning artifact (not a diff). Review the "
    "document AS A WHOLE for internal consistency against the rubric below. "
    "Each finding must anchor to a named section via the 'section' field "
    "(e.g. \"§Acceptance Criteria\"); leave 'line' unset — there are no diff "
    "lines. Do not summarize the artifact; critique it. If it is sound, "
    "return zero findings."
)


@functools.lru_cache(maxsize=None)
def _builtin_kind_rubric(kind: str) -> str | None:
    """Return the packaged ``{kind}_rubric.md`` text, or ``None`` if absent.

    Built-in rubrics ship under ``reviewer/data/prompts/`` and are the
    fallback when a repo doesn't override the rubric via
    ``.reviewer/prompts/{kind}_rubric.md``. Cached — the files don't change
    within a process. Same resource-loading pattern as
    ``reviewer/tools/benchmark_sweep.py``.
    """
    try:
        res = importlib.resources.files("reviewer.data.prompts") / f"{kind}_rubric.md"
        text = res.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return None
    return text if text.strip() else None


def classify_diff_content(content: str) -> str:
    """Classify a diff payload as ``"doc"``, ``"code"``, or ``"mixed"`` by
    its ADDED lines (fr_reviewer_1262ce18).

    Doc-heavy = the added content is predominantly prose: lines in a
    markdown / rst / txt file, comment lines, or blank lines. A doc-heavy
    diff is routed to a critique-not-summarize prompt so the reviewer stops
    echoing the prose back as findings.

    Heuristic only (no LLM): walks the unified diff, tracking the current
    file from ``+++`` headers, and buckets each added (``+``) content line.
    Non-diff content (no added lines) classifies as ``"code"`` — there's
    nothing to route, and full-document prose review is the artifact-review
    pipeline's concern (fr_reviewer_19c871ab), not this one.

    Thresholds: ``doc_ratio >= 0.7`` → ``"doc"``; ``<= 0.3`` → ``"code"``;
    in between → ``"mixed"`` (only ``"doc"`` is routed, conservatively).
    """
    doc = 0
    code = 0
    current_is_doc_file = False
    for raw in content.splitlines():
        # The unified-diff "+++ " file header has a trailing space ("+++ b/x",
        # "+++ /dev/null"); match it precisely so an ADDED line whose content
        # itself starts with "+++" (rendered "++++..." with the + prefix) is
        # NOT mistaken for a header. Every other non-added line (context,
        # removed, "---"/"@@"/"diff"/"index" headers) doesn't start with "+",
        # so the next check skips it.
        if raw.startswith("+++ "):
            current_is_doc_file = raw[4:].strip().lower().endswith(_DOC_FILE_EXTS)
            continue
        if not raw.startswith("+"):
            continue
        line = raw[1:].strip()
        if current_is_doc_file or _is_doc_line(line):
            doc += 1
        else:
            code += 1
    total = doc + code
    if total == 0:
        return "code"
    ratio = doc / total
    if ratio >= 0.7:
        return "doc"
    if ratio <= 0.3:
        return "code"
    return "mixed"


def build_review_prompt(
    request: ReviewRequest,
    *,
    include_schema: bool = False,
    repo_prompts: "RepoPrompts | None" = None,
    example_format: str | None = None,
) -> str:
    """Assemble the review prompt text from a :class:`ReviewRequest`.

    Providers that can enforce a JSON schema out-of-band (e.g. Claude CLI
    via ``--json-schema``) pass ``include_schema=False``. Providers that
    must carry the schema inline — OpenAI-compatible endpoints running
    ``response_format={"type": "json_object"}`` — pass ``include_schema=True``
    so the model knows exactly which fields to emit.

    Repo-side merge (FR fr_reviewer_92453047)
    -----------------------------------------
    When ``repo_prompts`` is non-empty, its three fields (system
    preamble, severity rubric, per-kind examples) are merged **before**
    the task content. The merge is pure concatenation-with-separators;
    this function is authoritative for the ordering.

    When ``repo_prompts`` is ``None`` or empty, the function behaves
    identically to its pre-FR shape — callers that don't know or don't
    care about repo prompts (tests, legacy code paths) don't have to
    change, and the prompt bytes are byte-for-byte identical.

    ``example_format`` resolves from the active model config's
    ``example_format`` field (see :mod:`reviewer.config.repo`). When
    ``None`` or an unrecognized value, examples default to markdown
    fences — the tokenization-neutral framing that works everywhere.
    """
    lines: list[str] = []

    # Built-in system prompt comes first. Repo ``system_preamble`` is
    # *merged after* so operators can add to but not replace the
    # reviewer's built-in identity — the trust-boundary design plus
    # the platform charter model the reviewer as a first-class agent
    # with its own voice, not a pure pass-through of whatever the
    # repo says.
    lines += [
        f"You are a code reviewer for the khonliang ecosystem. Read the {request.kind!r}",
        "content below and return ONLY a JSON object matching the schema you were",
        "given. No prose outside the JSON.",
        "",
    ]

    # Review-discipline calibration (fr_reviewer_ff923ebf): curbs the hot-tier
    # false-positive patterns (echo-as-finding, inverted-claim, style-as-
    # concern). Scoped to kinds that have NO rubric/framing of their own —
    # pr_diff, doc, pr_description — i.e. the rubric-less hot-tier path where
    # the dogfood false positives were observed. The artifact kinds
    # (fr/spec/milestone) are EXCLUDED on purpose: they already carry the
    # _ARTIFACT_REVIEW_INSTRUCTION ("critique, don't summarize") plus a per-kind
    # rubric, and that rubric — not this code-change-framed calibration — owns
    # their severity (its planning-integrity categories like
    # duplicate_of_existing_fr are legitimately blocking concerns this code-diff
    # wording would otherwise bias the model against). NB: `fr` runs on the hot
    # tier too, but it is rubric-governed, so the rubric is its calibration.
    if request.kind not in _ARTIFACT_KINDS:
        lines += [_REVIEW_DISCIPLINE_INSTRUCTION, ""]

    # Doc-hunk routing (fr_reviewer_1262ce18): a predominantly-prose change
    # gets a critique-not-summarize instruction so the model doesn't echo the
    # changed text back as a finding. Only fires for clearly doc-heavy diffs;
    # code / mixed diffs are unchanged (byte-identical to the pre-FR prompt).
    # Gated on the diff kind so non-diff payloads (spec/doc/fr full documents)
    # skip the line scan entirely — those route through the artifact-review
    # pipeline (fr_reviewer_19c871ab), not here.
    if request.kind == "pr_diff":
        classification = classify_diff_content(request.content)
        logger.debug(
            "diff classification (kind=%s): %s", request.kind, classification
        )
        if classification == "doc":
            lines += [_DOC_REVIEW_INSTRUCTION, ""]
    elif request.kind in _ARTIFACT_KINDS:
        # Full-document framing + per-kind rubric. A repo override
        # (.reviewer/prompts/<kind>_rubric.md, surfaced via RepoPrompts)
        # wins over the packaged built-in fallback. Built-in fires even when
        # repo_prompts is None (the raw-content ingestion path).
        from reviewer.config.prompts import RepoPrompts  # cold-path local import

        lines += [_ARTIFACT_REVIEW_INSTRUCTION, ""]
        rubric: str | None = None
        if isinstance(repo_prompts, RepoPrompts):
            rubric = repo_prompts.kind_rubrics.get(request.kind)
        rubric = rubric or _builtin_kind_rubric(request.kind)
        if rubric:
            lines += [f"## {request.kind.capitalize()} Rubric", "", rubric.rstrip(), ""]

    # Repo-side additions land between the built-in system and the
    # task-shape details (schema / instructions / context). An empty
    # ``repo_prompts`` (or ``None``) skips the whole block — the
    # degenerate path reproduces the pre-FR prompt bytes exactly.
    lines += _render_repo_prompts(
        repo_prompts, kind=request.kind, example_format=example_format
    )

    if include_schema:
        lines += [
            "## Response Schema",
            "",
            "```json",
            json.dumps(REVIEW_RESPONSE_SCHEMA, indent=2, sort_keys=True),
            "```",
            "",
        ]
    if request.instructions:
        lines += ["## Review Instructions", "", request.instructions, ""]
    if request.context:
        lines += [
            "## Context",
            "",
            json.dumps(request.context, indent=2, sort_keys=True),
            "",
        ]
    lines += ["## Content", "", request.content]
    return "\n".join(lines)


def _render_repo_prompts(
    repo_prompts: "RepoPrompts | None",
    *,
    kind: str,
    example_format: str | None,
) -> list[str]:
    """Render the merged repo-prompts section as prompt lines.

    Returns ``[]`` when nothing is configured so the caller can splice
    the result into the line list unconditionally without an empty-
    block check. The returned lines always end with a blank line when
    non-empty so concatenation stays visually clean.
    """
    # Fail-open fallback for direct-provider-call paths (tests, library
    # consumers, future call paths). The primary defense against a
    # non-``RepoPrompts`` value landing in
    # ``metadata["_khonliang_repo_prompts"]`` is the agent-layer reserved-
    # metadata strip in ``handle_review_text``. That strip only runs on
    # the bus-skill path; if a caller constructs a ``ReviewRequest``
    # directly and hands a bogus value to the provider,
    # ``build_review_prompt`` still has to do the right thing — returning
    # empty is safer than crashing an entire review. Kept above the
    # ``is_empty`` check because ``.is_empty`` is a ``RepoPrompts``-only
    # attribute; accessing it on e.g. a string would ``AttributeError``.
    from reviewer.config.prompts import RepoPrompts  # local import to keep module cold-path light

    if repo_prompts is None:
        return []
    if not isinstance(repo_prompts, RepoPrompts):
        return []
    if repo_prompts.is_empty:
        return []

    fmt = _resolve_example_format(example_format)
    out: list[str] = []

    if repo_prompts.system_preamble:
        # A section header disambiguates the preamble from the built-in
        # reviewer system prompt. Without it the two run together and
        # operators reading a debug dump can't tell where the custom
        # content starts. Cheap cost, high debuggability.
        out += [
            "## Repository System Preamble",
            "",
            repo_prompts.system_preamble.rstrip(),
            "",
        ]

    if repo_prompts.severity_rubric:
        out += [
            "## Severity Rubric",
            "",
            repo_prompts.severity_rubric.rstrip(),
            "",
        ]

    examples = repo_prompts.examples_for_kind(kind)
    if examples:
        out += ["## Examples", ""]
        for severity, text in examples:
            out += _wrap_example(severity, text.rstrip(), fmt=fmt)
            out.append("")

    return out


def _resolve_example_format(example_format: str | None) -> str:
    """Return a supported example format, falling back to markdown.

    Unknown / ``None`` → ``"markdown"``. No warning log — this function
    fires on every review; noise here would flood the review log with
    no actionable signal. Typos in model config are a config-lint
    concern for a future FR, not a runtime concern.
    """
    if example_format in _SUPPORTED_EXAMPLE_FORMATS:
        return example_format  # type: ignore[return-value]
    return _DEFAULT_EXAMPLE_FORMAT


def _wrap_example(severity: str, text: str, *, fmt: str) -> list[str]:
    """Frame an example's raw text per vendor-preferred format.

    The three framings all carry the same information (severity label
    + example body); only the surface markup differs. The model config's
    ``example_format`` picks one:

    - ``"xml"`` — Anthropic family. XML tagging performs better in
      Claude models than unstructured fences.
    - ``"json"`` — OpenAI-schema family. JSON literals land closer to
      how these models expect structured demonstrations.
    - ``"markdown"`` — default. Fenced code blocks are tokenization-
      neutral and work on every provider; the safest fallback.

    The severity label is always carried — wrapping-neutral — so the
    model can line examples up with the severity-rubric section even
    if the rubric shipped separately.
    """
    if fmt == "xml":
        # Minimal tag set. No attributes on the inner content — keeps
        # the model from trying to mirror unused schema bits back in
        # its own response.
        return [
            f'<example severity="{severity}">',
            text,
            "</example>",
        ]
    if fmt == "json":
        # ``json.dumps`` handles multi-line content, embedded quotes,
        # and unicode correctly. A hand-written ``{"example": "..."}``
        # would break on the first embedded quote in real example
        # content.
        payload = json.dumps(
            {"severity": severity, "example": text},
            indent=2,
            sort_keys=True,
        )
        return ["```json", payload, "```"]
    # Default: markdown. The ``### Severity`` subhead matches the
    # section-heading style of the rest of the prompt, and the fenced
    # block keeps the example from being misread as prose instructions.
    return [
        f"### {severity}",
        "",
        "```",
        text,
        "```",
    ]


__all__ = ["REVIEW_RESPONSE_SCHEMA", "build_review_prompt", "classify_diff_content"]
