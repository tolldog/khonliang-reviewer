"""Reviewer as a bus agent.

Skills live here. Providers + selection policy live in sibling modules:

- ``reviewer.providers`` — concrete :class:`ReviewProvider`
  implementations (Ollama, Claude-via-CLI).
- ``reviewer.selector`` — resolves ``(backend, model)`` to a provider
  instance. Currently a stubbed selector (caller override + config
  default); rule-table-driven selection wires in once WU4 (rule table)
  lands on main.

Usage::

    # Install into the bus
    python -m reviewer.agent install --id reviewer-primary --bus http://localhost:8787 --config config.yaml

    # Start (normally done by the bus on boot)
    python -m reviewer.agent --id reviewer-primary --bus http://localhost:8787 --config config.yaml

    # Uninstall
    python -m reviewer.agent uninstall --id reviewer-primary --bus http://localhost:8787
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import uuid
from typing import Any

from khonliang_bus import BaseAgent, Skill, handler
from khonliang_reviewer import (
    SEVERITY_ORDER,
    ReviewFinding,
    ReviewRequest,
    ReviewResult,
    severity_rank,
)

from reviewer.config.prompts import (
    RepoPrompts,
    load_repo_prompts,
)
from reviewer.config.repo import (
    RepoConfig,
    RepoConfigUnreachableError,
    load as load_repo_config,
    provider_to_vendor,
)
from reviewer.github_client import GithubClientError, ReviewerGithubClient
from reviewer.providers import (
    ClaudeCliProvider,
    ClaudeCliProviderConfig,
    CodexCliProvider,
    CodexCliProviderConfig,
    OllamaProvider,
    OllamaProviderConfig,
)
from reviewer.pricing_seed import load_default_pricing
from reviewer.rules import PolicyDecision, PolicyInput, decide
from reviewer.selector import ProviderSelector, SelectorConfig, UnknownBackendError
from reviewer.storage import UsageStore, open_usage_store


logger = logging.getLogger(__name__)


_REQUEST_ID_PREFIX = "rev-"


def _generate_request_id() -> str:
    return f"{_REQUEST_ID_PREFIX}{uuid.uuid4().hex[:16]}"


def _as_dict(val: Any) -> dict[str, Any]:
    """Return ``val`` as a dict, or an empty dict when it isn't one."""
    return val if isinstance(val, dict) else {}


_SEVERITY_LABELS = {
    "nit": "🟢 Nit",
    "comment": "🟡 Comment",
    "concern": "🔴 Concern",
}


#: GitHub-supported values for the ``event`` parameter on a review
#: submission the reviewer agent is allowed to use. ``APPROVE`` is
#: deliberately excluded: FR fr_developer_e72d8835 pins approval
#: authority to humans. If a future FR ever grants machine approval,
#: it should add the value here behind an explicit opt-in flag rather
#: than being broadly accepted on the ``event`` arg.
_VALID_REVIEW_EVENTS = frozenset({"COMMENT", "REQUEST_CHANGES", "PENDING"})


#: Built-in default severity floor. ``"nit"`` means "keep everything" —
#: noise reduction is strictly opt-in so adding severity_floor to the
#: agent can't silently change the review output of repos that haven't
#: configured it. Matches the FR's precedence tail.
_DEFAULT_SEVERITY_FLOOR = "nit"


#: In-process-only passthrough keys on :attr:`ReviewRequest.metadata`.
#:
#: ``_khonliang_repo_prompts`` carries a :class:`RepoPrompts` snapshot
#: into providers so they can merge repo-side calibration material into
#: the LLM prompt. ``_khonliang_example_format`` carries the active
#: model config's ``example_format`` string (``xml`` | ``json`` |
#: ``markdown``) so the wrapping layer knows which vendor framing to
#: use for example blocks.
#:
#: Underscore-prefixed convention signals to anything reading metadata
#: naively (GitHub logs, serializers, bus event consumers) that these
#: keys are implementation detail — skip them if you don't know what
#: they mean. The usage event construction in both providers reads
#: specific named fields (``repo``, ``pr_number``), not the whole
#: metadata dict, so these non-JSON-serializable values never leave
#: the in-process path.
_METADATA_REPO_PROMPTS_KEY = "_khonliang_repo_prompts"
_METADATA_EXAMPLE_FORMAT_KEY = "_khonliang_example_format"

#: Reserved prefix for internal-only passthrough keys on
#: :attr:`ReviewRequest.metadata`. Every key carrying values the agent
#: itself injects (see ``_METADATA_*`` constants above) starts with this
#: prefix. Callers MUST NOT supply keys with this prefix via
#: ``args["metadata"]`` — :func:`_strip_reserved_metadata` scrubs any
#: that slip in before the caller dict is merged into the request. The
#: scrub is the single, authoritative defense: downstream providers and
#: prompt-assembly code can then trust that when a reserved key is
#: present the agent put it there, and can use the expected type
#: without redundant ``isinstance`` checks on a trust boundary the
#: agent already enforces.
_RESERVED_METADATA_PREFIX = "_khonliang_"


def _strip_reserved_metadata(user_metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``user_metadata`` with reserved keys removed.

    Keys starting with :data:`_RESERVED_METADATA_PREFIX` are reserved
    for internal agent-side passthrough (repo-prompts snapshots,
    example-format hints). Accepting them from caller-supplied metadata
    would let an untrusted caller inject values into prompt-assembly
    code paths that expect specific in-process Python types — e.g.
    a :class:`RepoPrompts` instance the agent just built from a git-
    show read. Stripping before merge keeps the boundary simple: by
    the time the request reaches a provider, any reserved key it sees
    was put there by the agent itself.
    """
    if not user_metadata:
        return {}
    return {
        key: value
        for key, value in user_metadata.items()
        if not (isinstance(key, str) and key.startswith(_RESERVED_METADATA_PREFIX))
    }


class SeverityFloorError(ValueError):
    """Raised when a caller-supplied ``severity_floor`` isn't a known value.

    Subclass of :class:`ValueError` so existing ``except ValueError``
    blocks catch it, but distinct enough that callers wanting to branch
    on "validation error vs. other ValueError" can key on the class.
    """


def _validate_severity_floor(value: str, *, source: str) -> str:
    """Validate ``value`` is a known severity string; return it unchanged.

    Empty string is rejected by the caller (it's the "not set" sentinel
    at the skill-arg layer; callers resolve precedence before calling
    here). Unknown values raise :class:`SeverityFloorError` with a
    message naming ``source`` so operators can tell whether a skill
    arg or a config file is to blame.
    """
    if value not in SEVERITY_ORDER:
        raise SeverityFloorError(
            f"{source}: severity_floor={value!r} is not valid; "
            f"expected one of {list(SEVERITY_ORDER)}"
        )
    return value


def _load_repo_prompts_from_context(
    context: dict[str, Any],
) -> RepoPrompts | None:
    """Best-effort load of ``.reviewer/prompts/`` from context hints.

    Symmetric with :func:`_resolve_repo_severity_floor`: activates only
    when the caller threads ``repo_root`` + ``base_sha`` through
    ``context``. Orchestrators without a local clone (``review_pr``
    today, which fetches via the GitHub API) skip this layer and the
    review falls back to the built-in prompt only.

    Returns ``None`` — not an empty :class:`RepoPrompts` — when the
    layer is not applicable. Infrastructure failures (shallow clone,
    git subsystem error) log at warning and collapse to ``None``; the
    reviewer still runs, just without the repo-side prompt additions.
    A hard failure here would defeat the point of the whole graceful-
    absence contract that the rest of the ``.reviewer/`` loader
    carries.
    """
    repo_root = context.get("repo_root")
    base_sha = context.get("base_sha")
    if not isinstance(repo_root, str) or not repo_root:
        return None
    if not isinstance(base_sha, str) or not base_sha:
        return None
    try:
        prompts: RepoPrompts = load_repo_prompts(repo_root, base_sha=base_sha)
    except RepoConfigUnreachableError as exc:
        logger.warning("repo prompts load skipped: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("repo prompts load failed: %s", exc)
        return None
    if prompts.is_empty:
        # Normalise "tree present but empty" to "tree absent" so
        # downstream code has one signal for "nothing to merge" —
        # avoids branching on ``.is_empty`` at every call site.
        return None
    return prompts


def _load_repo_config_from_context(
    context: dict[str, Any],
) -> RepoConfig | None:
    """Best-effort load of ``.reviewer/config.yaml`` from context hints.

    Consolidates the git-plumbing + YAML-parse path that was previously
    duplicated inside :func:`_resolve_repo_severity_floor` and
    :func:`_resolve_example_format_from_config`. Callers that need the
    config for multiple resolutions (e.g. ``handle_review_text`` resolves
    both the severity floor and the model-config example_format) should
    invoke this helper once and thread the result into each resolver —
    otherwise every resolver re-runs ``git show`` and re-parses the YAML.

    Returns ``None`` when the context lacks ``repo_root`` / ``base_sha``
    hints, or when the config is unreachable / fails to load. The
    graceful-absence contract is preserved: a config-loader failure
    never fails the review.
    """
    repo_root = context.get("repo_root")
    base_sha = context.get("base_sha")
    if not isinstance(repo_root, str) or not repo_root:
        return None
    if not isinstance(base_sha, str) or not base_sha:
        return None
    try:
        return load_repo_config(repo_root, base_sha=base_sha)
    except RepoConfigUnreachableError as exc:
        logger.warning("repo config read skipped: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("repo config read failed: %s", exc)
        return None


def _resolve_example_format_from_config(
    cfg: RepoConfig | None,
    *,
    kind: str,
    vendor: str,
    model: str,
) -> str | None:
    """Consult ``.reviewer/models/<vendor>/<model>.yaml`` for ``example_format``.

    Accepts a pre-loaded :class:`RepoConfig` (or ``None`` when the
    context didn't yield one) rather than re-loading from
    ``repo_root``/``base_sha`` — the caller is expected to have obtained
    the config via :func:`_load_repo_config_from_context` and pass it in
    so the underlying ``git show`` + YAML parse happens once per
    ``handle_review_text`` invocation.

    Returns the string value verbatim when present and a string; any
    other value (missing file, wrong type, empty) returns ``None`` so
    the caller falls through to the markdown default inside
    :func:`build_review_prompt`.
    """
    if cfg is None:
        return None
    resolved = cfg.resolve(kind=kind, vendor=vendor, model=model)
    value = resolved.get("example_format")
    if isinstance(value, str) and value:
        return value
    return None


def _resolve_repo_severity_floor(cfg: RepoConfig | None) -> str | None:
    """Best-effort read of ``review.severity_floor`` from ``.reviewer/config.yaml``.

    Accepts a pre-loaded :class:`RepoConfig` (or ``None``) rather than
    re-loading from context — the caller is expected to obtain the config
    once via :func:`_load_repo_config_from_context` and thread it into
    both this helper and
    :func:`_resolve_example_format_from_config`. This prevents the
    duplicate git-plumbing + YAML-parse that would otherwise fire per
    ``handle_review_text`` invocation.

    Returns ``None`` when the layer is not applicable (config absent /
    value unset). The skill-arg layer is still honored by the caller
    because it's resolved first.
    """
    if cfg is None:
        return None
    return cfg.severity_floor


def _filter_findings_by_floor(
    findings: list[dict[str, Any]], floor: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ``findings`` into (kept, dropped) by severity rank.

    A finding is kept when its severity rank is ``>= rank(floor)``.
    Findings whose severity doesn't parse (corrupt provider output,
    unknown severity string) are **kept** — the filter's contract is
    noise reduction, not correctness enforcement; discarding an
    unparseable finding would silently hide real signal. A malformed
    finding is the provider's bug to fix, not the filter's data to
    drop.
    """
    floor_rank = severity_rank(floor)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for finding in findings:
        severity = finding.get("severity") if isinstance(finding, dict) else None
        if not isinstance(severity, str):
            kept.append(finding)
            continue
        try:
            rank = severity_rank(severity)
        except ValueError:
            kept.append(finding)
            continue
        if rank >= floor_rank:
            kept.append(finding)
        else:
            dropped.append(finding)
    return kept, dropped


def _strip_dropped_from_summary(summary: str, dropped: list[dict[str, Any]]) -> str:
    """Remove references to dropped findings from the ``summary`` prose.

    Some providers enumerate findings inside the summary itself
    (markdown bullet list, numbered section). If a dropped finding's
    title appears on a line **in one of three recognized shapes**,
    strip that line so the remaining prose reads cleanly. Lines that
    don't match any of the three shapes pass through untouched — even
    if the title happens to appear mid-sentence as prose.

    The three strip-eligible shapes (per line, after ``line.strip()``):

    1. **Bullet list item** — ``^[-*+]\\s+<title>(?=\\s|[:.,;)!?]|$)``
       (e.g. ``"- race condition: ..."`` or ``"- foo(): ..."``).
    2. **Title-colon prose** — ``^<title>\\s*:`` (e.g.
       ``"Missing docstring: ..."``).
    3. **Standalone title line** — ``^<title>\\s*$`` (just the title,
       nothing else).

    Any other occurrence of the title (mid-sentence, inside a paragraph
    that also says other things) is left alone. This matches the
    docstring's stated intent: only the obviously-enumerated shapes
    are collateral-damage-free to drop; prose paragraphs that mention
    a title in passing may carry information that has nothing to do
    with the dropped finding, so we don't touch them.

    Mid-word collisions (dropped title ``"race"`` vs summary word
    ``"embrace"``) are prevented by **start-of-line anchoring + exact
    escaped title**, not by ``\\b`` word-boundary. All three shapes
    start with either ``^`` (line start, after ``line.strip()``) or a
    bullet-marker prefix; that pins ``<title>`` to the beginning of a
    meaningful position in the line, so ``"embrace"`` never aligns
    with the escaped ``"race"`` pattern. The bullet shape's trailing
    ``(?=\\s|[:.,;)!?]|$)`` lookahead prevents a longer title like
    ``"race-condition-handler"`` from being partially matched by a
    dropped ``"race"`` finding — and critically, the explicit
    character-class trailer handles titles ending in non-word chars
    (e.g. ``"foo()"``) that a bare ``\\b`` would silently skip.

    **Ultra-short titles (``len(title.strip()) < 3``) are not
    strip-eligible.** Single-letter and two-character titles like
    ``"a"``, ``"if"``, ``"or"`` would match every indefinite article
    or conjunction in prose even with start-anchoring — the bullet
    shape ``"- a ..."`` is a legitimate construction. Skipping these
    titles means the caller keeps a slightly-noisier summary (the
    bullet for a dropped ``"a"`` finding survives), which is strictly
    safer than shredding unrelated summary lines. A future FR with a
    structured prompt that keeps summary and findings orthogonal
    would remove the need for this heuristic entirely.
    """
    if not summary or not dropped:
        return summary
    drop_titles = [
        str(f.get("title") or "").strip()
        for f in dropped
        if isinstance(f, dict)
    ]
    # Guard against ultra-short titles that would match common words
    # (``"a"`` / ``"if"`` / ``"or"``) even under start-anchoring — a
    # bullet line like ``"- a common issue"`` would otherwise have
    # the leading ``"a "`` stripped by a dropped ``"a"`` finding.
    # 3 chars is the minimum that empirically avoids the English
    # function-word collisions we've hit; shorter titles pass through
    # untouched rather than risk shredding unrelated summary prose.
    drop_titles = [t for t in drop_titles if len(t) >= 3]
    if not drop_titles:
        return summary
    # Precompile one pattern per title covering all three shapes in a
    # single alternation. ``re.IGNORECASE`` isn't used — provider
    # summaries and titles come from the same generation so casing
    # is consistent; case-insensitive would broaden matches without
    # catching a realistic failure mode.
    #
    # Shape anchors (applied to ``line.strip()``):
    #   1. bullet:         ^[-*+]\s+<title>(?=\s|[:.,;)!?]|$)
    #   2. title-colon:    ^<title>\s*:
    #   3. standalone:     ^<title>\s*$
    #
    # The bullet shape uses a character-class lookahead rather than
    # ``\b`` so titles ending in non-word characters still match. A
    # bare ``\b`` after the escaped title fails silently when the
    # title ends with e.g. ``"foo()"`` — ``\b`` only fires at a
    # word↔non-word transition, and two non-word chars in a row
    # (``) `` + whitespace) don't satisfy it. The explicit class
    # ``[:.,;)!?]`` + whitespace + end-of-line covers the realistic
    # summary-line trailers without false-positives.
    drop_patterns = [
        re.compile(
            rf"(?:^[-*+]\s+{re.escape(title)}(?=\s|[:.,;)!?]|$))"
            rf"|(?:^{re.escape(title)}\s*:)"
            rf"|(?:^{re.escape(title)}\s*$)"
        )
        for title in drop_titles
    ]
    kept_lines: list[str] = []
    for line in summary.splitlines():
        stripped = line.strip()
        if stripped and any(p.search(stripped) for p in drop_patterns):
            continue
        kept_lines.append(line)
    # Collapse 2+ consecutive blank lines (from stripped content) down
    # to a single blank line so the output doesn't end up visually
    # gap-ridden. Cheap — summaries are small.
    collapsed: list[str] = []
    blank_run = 0
    for line in kept_lines:
        if not line.strip():
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        collapsed.append(line)
    return "\n".join(collapsed).rstrip()


def _format_for_github(
    review_result: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Translate a :class:`ReviewResult` dict into GitHub review shape.

    Returns ``(summary_body, inline_comments)``:

    - ``summary_body`` — top-level review body. Always carries the
      review summary; summary-level findings (those without
      ``path``/``line``) are appended as a short bullet list so they
      don't get lost.
    - ``inline_comments`` — list of GitHub inline-comment dicts
      ``{"path", "line", "side": "RIGHT", "body"}``, one per finding
      that carries ``path`` + ``line``. Body includes a severity
      label, the finding title, its body, and an optional
      ````suggestion```` block.
    """
    summary = str(review_result.get("summary") or "")
    findings = review_result.get("findings") or []
    if not isinstance(findings, list):
        findings = []

    inline_comments: list[dict[str, Any]] = []
    summary_extras: list[str] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        title = str(finding.get("title") or "").strip()
        body_text = str(finding.get("body") or "").strip()
        severity = str(finding.get("severity") or "comment")
        label = _SEVERITY_LABELS.get(severity, severity)
        path = finding.get("path")
        line = finding.get("line")
        suggestion = finding.get("suggestion")

        # GitHub review-comments require a string path and a POSITIVE
        # line number; ``bool`` is excluded explicitly because it
        # subclasses ``int`` (``True`` would otherwise read as line 1).
        # Findings that don't satisfy both fall back to summary-level
        # notes to avoid 422 Unprocessable Entity from GitHub.
        anchored = (
            isinstance(path, str)
            and bool(path)
            and isinstance(line, int)
            and not isinstance(line, bool)
            and line > 0
        )
        if anchored:
            parts = [f"**{label} — {title}**" if title else f"**{label}**"]
            if body_text:
                parts.append(body_text)
            if isinstance(suggestion, str) and suggestion:
                parts.append(f"```suggestion\n{suggestion}\n```")
            inline_comments.append(
                {
                    "path": path,
                    "line": int(line),
                    "side": "RIGHT",
                    "body": "\n\n".join(parts),
                }
            )
        else:
            headline = f"- **{label}**"
            if title:
                headline += f" — {title}"
            if body_text:
                headline += f": {body_text}"
            summary_extras.append(headline)

    body = summary.strip()
    if summary_extras:
        # Always mark the extras with a heading so the bullet list has
        # a clear context, even when the model returned an empty
        # top-level summary.
        separator = "\n\n" if body else ""
        body += f"{separator}### Additional notes\n\n" + "\n".join(summary_extras)
    if not body:
        body = "No findings."
    return body, inline_comments


def _estimate_diff_size(content: str, kind: str) -> tuple[int, int]:
    """Rough (line_count, file_count) for rule-table input.

    Only non-zero for ``kind == "pr_diff"``. Cheap to compute and good
    enough for coarse-grained routing; callers with authoritative
    counts can pass them through ``context["diff_line_count"]`` /
    ``context["diff_file_count"]`` to override.
    """
    if kind != "pr_diff":
        return 0, 0
    line_count = content.count("\n")
    file_count = content.count("\ndiff --git")
    if content.startswith("diff --git"):
        file_count += 1
    return line_count, file_count


def _positive_float_or_none(val: Any) -> float | None:
    """Coerce ``val`` to a positive float, or None when it's falsy/invalid.

    The ``usage_summary`` skill accepts ``since``/``until`` as floats
    with a default of 0. Treating 0 as "no filter" lets callers omit
    the field on the bus wire without constructing explicit nulls.
    """
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def _policy_input_for(
    *, kind: str, content: str, context: dict[str, Any]
) -> PolicyInput:
    """Build a :class:`PolicyInput` from the pieces available in the handler.

    Callers can supply authoritative ``diff_line_count`` /
    ``diff_file_count`` / ``profile`` in ``context``; otherwise they're
    estimated from ``content`` (for diffs) or left empty.
    """
    est_lines, est_files = _estimate_diff_size(content, kind)
    return PolicyInput(
        kind=kind,
        diff_line_count=int(context.get("diff_line_count") or est_lines),
        diff_file_count=int(context.get("diff_file_count") or est_files),
        profile=(
            context.get("profile")
            if isinstance(context.get("profile"), dict)
            else None
        ),
    )


class ReviewerAgent(BaseAgent):
    """Bus-native reviewer agent.

    Exposes ``review_text`` + ``review_diff`` skills. Provider selection
    uses the stubbed :class:`ProviderSelector` — caller override wins,
    config default otherwise. Rule-table routing joins once WU4 merges.

    The agent lazily constructs its provider selector on first use from
    its ``config.yaml``. Tests inject a pre-built selector via the
    ``selector=`` kwarg to avoid touching real Ollama / Claude CLI
    subprocesses.
    """

    agent_id = "reviewer-primary"
    agent_type = "reviewer"
    module_name = "reviewer.agent"

    def __init__(
        self,
        *,
        selector: ProviderSelector | None = None,
        usage_store: UsageStore | None = None,
        github_client: ReviewerGithubClient | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._injected_selector = selector
        self._cached_selector: ProviderSelector | None = None
        self._injected_store = usage_store
        self._cached_store: UsageStore | None = None
        self._injected_github = github_client
        self._cached_github: ReviewerGithubClient | None = None

    async def start(self) -> None:
        """Eager-init the usage store so the SQLite file lands on launch.

        Operators rely on seeing ``data/reviewer.db`` appear as soon as
        the agent boots (for tailing, backups, monitoring) rather than
        waiting for the first skill call to create it lazily. The
        selector + github client stay lazy — provider construction can
        be expensive (Ollama HTTP client, Claude CLI probe) and is only
        worth paying for when a review actually runs.
        """
        # _ensure_usage_store is idempotent; tests that inject an
        # in-memory store skip the filesystem touch entirely.
        try:
            self._ensure_usage_store()
        except Exception as exc:
            logger.warning("reviewer usage store init failed at start(): %s", exc)
        await super().start()

    # -- skill surface -------------------------------------------------

    def register_skills(self) -> list[Skill]:
        return [
            Skill(
                "review_text",
                "Run a review over arbitrary content. Returns structured findings + usage record.",
                {
                    "kind": {"type": "string", "required": True},
                    "content": {"type": "string", "required": True},
                    "instructions": {"type": "string", "default": ""},
                    "context": {"type": "object", "default": {}},
                    "backend": {"type": "string", "default": ""},
                    "model": {"type": "string", "default": ""},
                    "request_id": {"type": "string", "default": ""},
                    "metadata": {"type": "object", "default": {}},
                    # severity_floor: drop findings below this severity
                    # from the returned result. "" = no skill-arg
                    # override (fall through to config then default
                    # "nit"). Valid values: "nit"|"comment"|"concern".
                    "severity_floor": {"type": "string", "default": ""},
                },
                since="0.1.0",
            ),
            Skill(
                "review_diff",
                "Shortcut for review_text with kind='pr_diff'.",
                {
                    "diff": {"type": "string", "required": True},
                    "instructions": {"type": "string", "default": ""},
                    "context": {"type": "object", "default": {}},
                    "backend": {"type": "string", "default": ""},
                    "model": {"type": "string", "default": ""},
                    "request_id": {"type": "string", "default": ""},
                    "metadata": {"type": "object", "default": {}},
                    "severity_floor": {"type": "string", "default": ""},
                },
                since="0.1.0",
            ),
            Skill(
                "review_pr",
                "Fetch a GitHub PR diff + metadata, run review_text over it, "
                "and post the result as a GitHub PR review. Returns the "
                "ReviewResult augmented with the posted-review info (or a "
                "dry-run payload when dry_run=true).",
                {
                    "repo": {"type": "string", "required": True},
                    "pr_number": {"type": "integer", "required": True},
                    "instructions": {"type": "string", "default": ""},
                    "backend": {"type": "string", "default": ""},
                    "model": {"type": "string", "default": ""},
                    "dry_run": {"type": "boolean", "default": False},
                    "event": {"type": "string", "default": "COMMENT"},
                    "severity_floor": {"type": "string", "default": ""},
                },
                since="0.1.0",
            ),
            Skill(
                "usage_summary",
                "Aggregated token usage + estimated API cost grouped by "
                "(backend, model). All filters optional; omit to summarize "
                "the full history.",
                {
                    "backend": {"type": "string", "default": ""},
                    "model": {"type": "string", "default": ""},
                    "since": {"type": "number", "default": 0},
                    "until": {"type": "number", "default": 0},
                },
                since="0.1.0",
            ),
        ]

    @handler("review_text")
    async def handle_review_text(self, args: dict[str, Any]) -> dict[str, Any]:
        kind = str(args.get("kind") or "").strip()
        if not kind:
            return {"error": "kind is required"}

        content = args.get("content")
        if not isinstance(content, str) or not content:
            return {"error": "content is required and must be a non-empty string"}

        caller_backend = args.get("backend") or None
        caller_model = args.get("model") or None
        context = _as_dict(args.get("context"))

        # Load ``.reviewer/config.yaml`` **once** per review. Both the
        # severity_floor resolver and the example_format resolver
        # consult the same file via the same git-show plumbing; loading
        # independently would double the ``git show`` + YAML-parse cost
        # on every review with ``repo_root``/``base_sha`` hints.
        # ``None`` when context lacks hints OR the config is unreachable —
        # the resolvers handle that case by falling through to defaults.
        repo_cfg = _load_repo_config_from_context(context)

        # Resolve severity_floor precedence up-front so a validation
        # error surfaces before we spend a provider call. Order matches
        # the FR (skill arg → .reviewer/config.yaml → built-in default).
        # The ``skill_arg`` step validates eagerly because the caller
        # typo'd; the config-layer step validates because operators
        # can typo their YAML too. Default is trusted (module constant).
        try:
            effective_floor = self._resolve_severity_floor(args, repo_cfg)
        except SeverityFloorError as exc:
            return {"error": str(exc)}

        try:
            selector = self._ensure_selector()
            if caller_backend or caller_model:
                provider, chosen_model = selector.select(
                    backend=caller_backend, model=caller_model
                )
                selection_reason = "caller override"
            else:
                decision = decide(
                    _policy_input_for(kind=kind, content=content, context=context)
                )
                provider, chosen_model = selector.select(
                    backend=decision.backend, model=decision.model
                )
                selection_reason = f"rule-table: {decision.reason}"
        except UnknownBackendError as exc:
            return {"error": str(exc)}

        logger.debug(
            "reviewer.select: backend=%s model=%s reason=%s kind=%s",
            provider.name,
            chosen_model,
            selection_reason,
            kind,
        )

        # Strip any ``_khonliang_*`` keys from caller-supplied metadata
        # before merge — those are reserved for in-process passthrough
        # from the agent to the provider (see ``_RESERVED_METADATA_PREFIX``).
        # A caller cannot be allowed to inject a value for e.g.
        # ``_khonliang_repo_prompts``: providers forward it into
        # :func:`build_review_prompt` expecting a :class:`RepoPrompts`
        # instance the agent just loaded from a trusted base SHA.
        metadata = {
            **_strip_reserved_metadata(_as_dict(args.get("metadata"))),
            "model": chosen_model,
        }

        # Repo-side prompt merge (FR fr_reviewer_92453047). Loaded here
        # rather than inside the provider so both Ollama and Claude-CLI
        # providers get the same merge behavior without re-implementing
        # the git-show plumbing. Graceful-absence: when context lacks
        # ``repo_root``/``base_sha``, both helpers return ``None`` and
        # providers fall back to the built-in prompt bytes.
        repo_prompts = _load_repo_prompts_from_context(context)
        # ``provider.name`` is the transport identifier (``ollama`` /
        # ``claude_cli``). ``.reviewer/models/<vendor>/`` is keyed on the
        # upstream model family (``ollama`` / ``anthropic``). Translate
        # so a repo's ``anthropic/_default.yaml: example_format: xml``
        # actually reaches Claude-backed reviews. Without this step the
        # resolver looks under ``claude_cli/`` (which will never exist)
        # and silently falls back to the markdown default.
        example_format = _resolve_example_format_from_config(
            repo_cfg,
            kind=kind,
            vendor=provider_to_vendor(provider.name),
            model=chosen_model,
        )
        if repo_prompts is not None:
            metadata[_METADATA_REPO_PROMPTS_KEY] = repo_prompts
        if example_format is not None:
            metadata[_METADATA_EXAMPLE_FORMAT_KEY] = example_format

        request = ReviewRequest(
            kind=kind,
            content=content,
            instructions=str(args.get("instructions") or ""),
            context=context,
            metadata=metadata,
            request_id=str(args.get("request_id") or _generate_request_id()),
        )

        result = await provider.review(request)
        # Apply the severity_floor post-filter at the edge of the return
        # path, AFTER the provider returns but BEFORE usage recording.
        # The usage event gets the filtered count so downstream
        # analytics can correlate (backend, model, floor) → noise
        # reduction without re-running the review.
        self._apply_severity_floor(result, effective_floor)
        await self._record_usage(result)
        return result.to_dict()

    def _resolve_severity_floor(
        self, args: dict[str, Any], cfg: RepoConfig | None
    ) -> str:
        """Resolve the effective severity_floor per the FR precedence chain.

        High-to-low:

        1. Skill-arg ``severity_floor`` (non-empty string) — **strict**.
           A bad value here is a caller bug; raise
           :class:`SeverityFloorError` so the review fails fast with a
           message the caller can act on.
        2. ``review.severity_floor`` / ``checks.severity_floor`` in
           ``.reviewer/config.yaml`` — **lenient**. Only consulted when
           the caller passes a pre-loaded :class:`RepoConfig` (obtained
           via :func:`_load_repo_config_from_context`). A bad value in
           YAML shouldn't nuke every review for that repo — log a
           warning naming the offending value and fall through to the
           built-in default. Reviewing is more important than
           config-layer correctness.
        3. :data:`_DEFAULT_SEVERITY_FLOOR` (``"nit"`` — no filtering).

        Rationale for asymmetric validation: the skill-arg path is a
        programmatic caller (another agent, a test, an orchestrator) —
        strict failure is the correct feedback channel. The config-layer
        path is a human-edited YAML file on a repo; a typo there
        shouldn't silently wedge CI.

        Note: ``cfg`` is threaded in (rather than re-loaded here) so that
        ``handle_review_text`` loads the config exactly once per review
        and shares it with :func:`_resolve_example_format_from_config`.
        """
        arg_value = args.get("severity_floor")
        if isinstance(arg_value, str) and arg_value:
            return _validate_severity_floor(arg_value, source="skill arg")

        config_value = _resolve_repo_severity_floor(cfg)
        if isinstance(config_value, str) and config_value:
            try:
                return _validate_severity_floor(
                    config_value, source=".reviewer/config.yaml"
                )
            except SeverityFloorError as exc:
                # RepoConfig.severity_floor reads review.severity_floor
                # first, falling back to checks.severity_floor — the
                # warning names the resolved key generically so operators
                # aren't misled about which key actually carried the bad
                # value.
                logger.warning(
                    "reviewer: ignoring invalid .reviewer/config.yaml "
                    "severity_floor=%r (checked review.severity_floor "
                    "and checks.severity_floor); falling back to "
                    "default %r (%s)",
                    config_value,
                    _DEFAULT_SEVERITY_FLOOR,
                    exc,
                )
                return _DEFAULT_SEVERITY_FLOOR

        return _DEFAULT_SEVERITY_FLOOR

    def _apply_severity_floor(
        self, result: ReviewResult, floor: str
    ) -> None:
        """Drop sub-floor findings from ``result`` in place + record the count.

        Mutates ``result.findings`` (which is what the caller will see
        via ``result.to_dict()``), strips references to dropped findings
        from ``result.summary``, and bumps ``result.usage.findings_filtered_count``
        so the usage record reflects the filter outcome.

        When ``floor == _DEFAULT_SEVERITY_FLOOR`` and the default is the
        lowest rank (``"nit"``), the filter is a no-op — the rank
        comparison keeps every finding unchanged. We still run through
        so the ``findings_filtered_count`` field is written to the
        SQLite usage row with an explicit zero so new rows written
        after this code lands always carry a value.

        Important caveat for analytics: the SQLite migration adds the
        column with ``NOT NULL DEFAULT 0``, which back-fills existing
        rows from before the severity_floor feature to 0 as well.
        That means ``findings_filtered_count = 0`` is ambiguous —
        it can mean either "filter ran, dropped nothing" OR "row
        predates the severity_floor feature entirely". The two cases
        are NOT distinguishable from the SQLite column alone; a
        downstream analytic that cares about the distinction needs a
        separate signal (e.g. a row-creation timestamp compared
        against the feature's rollout date, or a dedicated
        ``severity_floor_applied: bool`` column in a future migration).

        Note on the bus-event payload: ``UsageEvent.to_dict()`` omits
        ``findings_filtered_count`` when the value is 0 (wire-shape
        preservation — see khonliang-reviewer-lib#4). Bus-subscriber
        consumers that want to know the filter ran at all must infer
        from the absence / presence of the field plus review-handler
        timing, not from a value comparison.
        """
        original = [
            f.to_dict() if isinstance(f, ReviewFinding) else f
            for f in (result.findings or [])
        ]
        kept_dicts, dropped_dicts = _filter_findings_by_floor(original, floor)

        # Rebuild the findings list as ReviewFinding objects so the
        # dataclass shape is preserved — to_dict() handles both, but
        # downstream code that peeks at result.findings sees the
        # typed form.
        result.findings = [
            ReviewFinding.from_dict(f) if isinstance(f, dict) else f
            for f in kept_dicts
        ]
        if dropped_dicts:
            result.summary = _strip_dropped_from_summary(
                result.summary, dropped_dicts
            )
        if result.usage is not None:
            result.usage.findings_filtered_count = len(dropped_dicts)

    @handler("review_diff")
    async def handle_review_diff(self, args: dict[str, Any]) -> dict[str, Any]:
        diff = args.get("diff")
        if not isinstance(diff, str) or not diff:
            return {"error": "diff is required and must be a non-empty string"}
        forwarded = {
            k: v for k, v in args.items() if k not in {"diff", "kind", "content"}
        }
        forwarded["kind"] = "pr_diff"
        forwarded["content"] = diff
        return await self.handle_review_text(forwarded)

    @handler("review_pr")
    async def handle_review_pr(self, args: dict[str, Any]) -> dict[str, Any]:
        """End-to-end: fetch PR, review via review_text, post back to GitHub.

        ``event`` defaults to ``"COMMENT"`` — the only event the reviewer
        agent is allowed to use autonomously. ``APPROVE`` is rejected at
        the validation step (approval authority stays human per FR).
        ``REQUEST_CHANGES`` / ``PENDING`` are still accepted for operator
        tooling + human-in-the-loop scenarios.
        """
        repo = str(args.get("repo") or "").strip()
        if not repo:
            return {"error": "repo is required (owner/name form)"}
        pr_raw = args.get("pr_number")
        # Reject bool explicitly — bool subclasses int, so `int(True)` silently
        # becomes PR #1 and `int(False)` becomes 0. Both are wrong and need to
        # surface as errors rather than targeting the wrong PR.
        if isinstance(pr_raw, bool):
            return {"error": "pr_number must be an integer, not a boolean"}
        try:
            pr_number = int(pr_raw)
        except (TypeError, ValueError):
            return {"error": "pr_number is required and must be an integer"}
        if pr_number <= 0:
            return {"error": "pr_number must be positive"}

        dry_run_raw = args.get("dry_run", False)
        # Strict bool: `bool(val)` would accept any truthy value including the
        # string "false", which would unexpectedly skip posting. Force a real
        # boolean so operator typos / YAML-to-JSON mishaps fail loudly.
        if not isinstance(dry_run_raw, bool):
            return {
                "error": (
                    f"dry_run must be a boolean, got {type(dry_run_raw).__name__}"
                )
            }
        dry_run = dry_run_raw
        event_raw = str(args.get("event") or "COMMENT").strip().upper()
        if event_raw not in _VALID_REVIEW_EVENTS:
            return {
                "error": (
                    "event must be one of "
                    f"{sorted(_VALID_REVIEW_EVENTS)}; got {event_raw!r}"
                )
            }
        event = event_raw

        github = self._ensure_github_client()
        try:
            # Fetches are independent; run them concurrently to halve
            # end-to-end latency on high-latency GitHub API links.
            metadata, diff = await asyncio.gather(
                github.get_pr_metadata(repo, pr_number),
                github.get_pr_diff(repo, pr_number),
            )
        except GithubClientError as exc:
            return {"error": f"github fetch failed: {exc}"}

        # Feed the diff into review_text via the shared path so selector,
        # rule table, severity_floor, and usage recording all run exactly
        # once per review, regardless of which entry skill was called.
        review_args: dict[str, Any] = {
            "kind": "pr_diff",
            "content": diff,
            "instructions": str(args.get("instructions") or ""),
            "context": {
                "pr": metadata.to_dict(),
            },
            "backend": args.get("backend") or "",
            "model": args.get("model") or "",
            "metadata": {"repo": repo, "pr_number": pr_number},
            # review_pr fetches via API (no local clone), so the
            # ``.reviewer/config.yaml`` layer can't activate here —
            # callers who want a config-layer floor should call
            # ``review_text`` directly with repo_root/base_sha in
            # context. Pass through the skill-arg floor only.
            "severity_floor": args.get("severity_floor") or "",
        }
        review_result = await self.handle_review_text(review_args)
        # ReviewResult.to_dict() always carries an ``error`` key (empty
        # string when the review succeeded). Early-return only on a
        # truthy error message.
        if review_result.get("error"):
            return review_result

        posted_body, posted_comments = _format_for_github(review_result)

        if dry_run:
            return {
                **review_result,
                "pr": metadata.to_dict(),
                "github": {
                    "dry_run": True,
                    "body": posted_body,
                    "comments": posted_comments,
                    "event": event,
                },
            }

        try:
            submitted = await github.submit_review(
                repo,
                pr_number,
                body=posted_body,
                comments=posted_comments,
                event=event,
                commit_sha=metadata.head_sha or None,
            )
        except GithubClientError as exc:
            return {
                **review_result,
                "pr": metadata.to_dict(),
                "error": f"github post failed: {exc}",
            }

        return {
            **review_result,
            "pr": metadata.to_dict(),
            "github": {
                "dry_run": False,
                "review": submitted.to_dict(),
                "inline_comments_posted": len(posted_comments),
                "event": event,
            },
        }

    @handler("usage_summary")
    async def handle_usage_summary(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return token + cost aggregates grouped by (backend, model).

        Storage failures (DB path unreadable, permission issue, bad
        YAML on seed) surface as a structured ``{"error": "..."}``
        response rather than raising — keeps the skill surface from
        crashing out when the store is broken.
        """
        try:
            store = self._ensure_usage_store()
            summaries = store.summarize(
                backend=(args.get("backend") or None),
                model=(args.get("model") or None),
                since=_positive_float_or_none(args.get("since")),
                until=_positive_float_or_none(args.get("until")),
            )
        except Exception as exc:
            logger.warning("usage_summary failed: %s", exc)
            return {
                "error": f"usage_summary failed: {exc}",
                "entries": [],
                "total_rows": 0,
                "total_cost_usd": 0.0,
            }
        return {
            "entries": [s.to_dict() for s in summaries],
            "total_rows": sum(s.rows for s in summaries),
            "total_cost_usd": sum(s.total_cost_usd for s in summaries),
        }

    # -- internals -----------------------------------------------------

    async def _record_usage(self, result: ReviewResult) -> None:
        """Persist the usage event + emit ``reviewer.usage`` on the bus.

        Called from the review handlers after provider.review() returns.
        Zero-cost events (Ollama-backed) get their ``estimated_api_cost_usd``
        back-filled from the ``model_pricing`` table before storage.

        Every failure path — store open/create, cost back-fill, write,
        publish — is wrapped so accounting problems never propagate
        into the caller's review. That's the whole point of this
        method being separate from the review flow itself.
        """
        if result.usage is None:
            return
        try:
            store = self._ensure_usage_store()
        except Exception as exc:
            logger.warning("reviewer usage store unavailable: %s", exc)
            return

        try:
            filled = store.back_fill_cost(result.usage)
        except Exception as exc:
            logger.warning("reviewer usage back_fill_cost failed: %s", exc)
            filled = result.usage
        # Mutate the result so the caller sees the back-filled cost too.
        result.usage = filled

        try:
            store.write_usage(filled)
        except Exception as exc:
            logger.warning("reviewer_usage write failed: %s", exc)

        try:
            await self.publish("reviewer.usage", filled.to_dict())
        except RuntimeError as exc:
            # Agent not connected (tests, dry-run) — expected in those paths.
            logger.debug("skipping reviewer.usage publish: %s", exc)
        except Exception as exc:
            logger.warning("reviewer.usage publish failed: %s", exc)

    def _ensure_selector(self) -> ProviderSelector:
        if self._injected_selector is not None:
            return self._injected_selector
        if self._cached_selector is None:
            self._cached_selector = self._build_default_selector()
        return self._cached_selector

    def _ensure_github_client(self) -> ReviewerGithubClient:
        if self._injected_github is not None:
            return self._injected_github
        if self._cached_github is None:
            self._cached_github = ReviewerGithubClient()
        return self._cached_github

    def _ensure_usage_store(self) -> UsageStore:
        if self._injected_store is not None:
            return self._injected_store
        if self._cached_store is None:
            config = self._load_config()
            db_path = str(config.get("db_path") or "data/reviewer.db")
            store = open_usage_store(db_path)
            try:
                seeded = store.seed_pricing_if_empty(load_default_pricing())
                if seeded:
                    logger.info("seeded %d default pricing rows", seeded)
            except Exception as exc:
                # Seeding is best-effort — missing YAML or parse error
                # shouldn't block agent startup.
                logger.warning("default pricing seed failed: %s", exc)
            self._cached_store = store
        return self._cached_store

    def _build_default_selector(self) -> ProviderSelector:
        config = self._load_config()
        providers_cfg = _as_dict(config.get("providers"))
        claude_cfg = _as_dict(providers_cfg.get("claude_cli"))
        codex_cfg = _as_dict(providers_cfg.get("codex_cli"))
        ollama_cfg = _as_dict(providers_cfg.get("ollama"))
        providers = {
            "claude_cli": ClaudeCliProvider(
                ClaudeCliProviderConfig(
                    binary=str(claude_cfg.get("binary") or "claude"),
                )
            ),
            "codex_cli": CodexCliProvider(
                CodexCliProviderConfig(
                    binary=str(codex_cfg.get("binary") or "codex"),
                    default_model=str(codex_cfg.get("default_model") or ""),
                )
            ),
            "ollama": OllamaProvider(
                OllamaProviderConfig(
                    base_url=str(
                        ollama_cfg.get("base_url") or "http://localhost:11434/v1"
                    ),
                    # Source Ollama's provider-default model from
                    # ``providers.ollama.default_model`` (per-provider
                    # config), falling back to the built-in qwen
                    # baseline. Decoupled from the global
                    # ``config.default_model`` so an operator who sets
                    # ``default_provider: claude_cli`` and
                    # ``default_model: claude-opus-4-7`` doesn't
                    # accidentally inject a Claude model id into Ollama
                    # when a caller picks ``backend: ollama`` without
                    # specifying a model. The selector deliberately
                    # returns ``""`` for non-default-backend selections
                    # (see ``ProviderSelector.select``); each provider
                    # then applies its own config-level default.
                    default_model=str(
                        ollama_cfg.get("default_model") or "qwen2.5-coder:14b"
                    ),
                )
            ),
        }
        return ProviderSelector(
            providers,
            SelectorConfig(
                default_backend=str(config.get("default_provider") or "ollama"),
                default_model=str(config.get("default_model") or "qwen2.5-coder:14b"),
            ),
        )

    def _load_config(self) -> dict[str, Any]:
        """Load ``self.config_path`` as YAML, tolerating missing file / pyyaml / parse or IO errors.

        Every failure path falls back to an empty dict so the agent can
        still start with defaults — useful for tests, first-run setups,
        and operator misconfiguration that would otherwise crash the
        first skill call.
        """
        path = getattr(self, "config_path", "") or ""
        if not path:
            return {}
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("pyyaml not installed; using default reviewer config")
            return {}
        try:
            with open(path) as f:
                loaded = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("reviewer config %s not found; using defaults", path)
            return {}
        except yaml.YAMLError as exc:
            logger.warning(
                "reviewer config %s failed to parse (%s); using defaults", path, exc
            )
            return {}
        except OSError as exc:
            # Covers PermissionError, IsADirectoryError, and other IO failures.
            logger.warning(
                "reviewer config %s not readable (%s); using defaults", path, exc
            )
            return {}
        if isinstance(loaded, dict):
            return loaded
        if loaded is not None:
            logger.warning(
                "reviewer config %s must have a mapping at the YAML root; "
                "got %s; using defaults",
                path,
                type(loaded).__name__,
            )
        return {}


def create_reviewer_agent(
    *, agent_id: str, bus_url: str, config_path: str
) -> ReviewerAgent:
    agent = ReviewerAgent(
        agent_id=agent_id,
        bus_url=bus_url,
        config_path=config_path,
    )
    return agent


def main() -> None:
    from khonliang_bus import add_version_flag

    parser = argparse.ArgumentParser(
        prog="reviewer.agent",
        description="khonliang-reviewer bus agent",
    )
    add_version_flag(parser)
    parser.add_argument(
        "command",
        nargs="?",
        choices=["install", "uninstall"],
        help="install or uninstall from the bus",
    )
    parser.add_argument("--id", default="reviewer-primary")
    parser.add_argument("--bus", default="http://localhost:8787")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if args.command in ("install", "uninstall"):
        ReviewerAgent.from_cli(
            [
                args.command,
                "--id", args.id,
                "--bus", args.bus,
                "--config", args.config,
            ]
        )
        return

    agent = create_reviewer_agent(
        agent_id=args.id,
        bus_url=args.bus,
        config_path=args.config,
    )
    asyncio.run(agent.start())


if __name__ == "__main__":
    main()
