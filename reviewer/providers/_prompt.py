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
"""

from __future__ import annotations

import json
from typing import Any

from khonliang_reviewer import ReviewRequest


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


def build_review_prompt(request: ReviewRequest, *, include_schema: bool = False) -> str:
    """Assemble the review prompt text from a :class:`ReviewRequest`.

    Providers that can enforce a JSON schema out-of-band (e.g. Claude CLI
    via ``--json-schema``) pass ``include_schema=False``. Providers that
    must carry the schema inline — OpenAI-compatible endpoints running
    ``response_format={"type": "json_object"}`` — pass ``include_schema=True``
    so the model knows exactly which fields to emit.
    """
    lines = [
        f"You are a code reviewer for the khonliang ecosystem. Read the {request.kind!r}",
        "content below and return ONLY a JSON object matching the schema you were",
        "given. No prose outside the JSON.",
        "",
    ]
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


__all__ = ["REVIEW_RESPONSE_SCHEMA", "build_review_prompt"]
