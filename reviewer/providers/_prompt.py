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

import json
from typing import TYPE_CHECKING, Any

from khonliang_reviewer import ReviewRequest

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
    if repo_prompts is None or repo_prompts.is_empty:
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


__all__ = ["REVIEW_RESPONSE_SCHEMA", "build_review_prompt"]
