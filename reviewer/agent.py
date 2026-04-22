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

from reviewer.config.repo import (
    RepoConfig,
    RepoConfigUnreachableError,
    load as load_repo_config,
)
from reviewer.github_client import GithubClientError, ReviewerGithubClient
from reviewer.providers import (
    ClaudeCliProvider,
    ClaudeCliProviderConfig,
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


def _resolve_repo_severity_floor(context: dict[str, Any]) -> str | None:
    """Best-effort read of ``review.severity_floor`` from ``.reviewer/config.yaml``.

    Activates only when the caller threads ``repo_root`` + ``base_sha``
    through ``context`` — ``review_text`` as a bus skill has no local
    filesystem assumption, so the caller (orchestrator with a local
    clone) opts in by providing those hints. Orchestrators without a
    local clone (``review_pr`` today) will skip this layer and fall
    straight to the built-in default unless the caller passed
    ``severity_floor`` on the skill args.

    Returns ``None`` when the layer is not applicable (hints missing,
    config file absent, value unset). Infrastructure failures
    (shallow clone, git error) are logged at warning and collapse to
    ``None`` — the reviewer still runs, just without the config-layer
    floor. The skill-arg layer is still honored because it was resolved
    first.
    """
    repo_root = context.get("repo_root")
    base_sha = context.get("base_sha")
    if not isinstance(repo_root, str) or not repo_root:
        return None
    if not isinstance(base_sha, str) or not base_sha:
        return None
    try:
        cfg: RepoConfig = load_repo_config(repo_root, base_sha=base_sha)
    except RepoConfigUnreachableError as exc:
        logger.warning("severity_floor config read skipped: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("severity_floor config read failed: %s", exc)
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
    title appears as a line in the summary, strip that line so the
    remaining prose reads cleanly. Lines that don't match any dropped
    finding pass through untouched.

    Matches each title as a whole-word token (``\b<title>\b``) per line
    rather than as a bare substring. Word-boundary anchoring keeps
    dropped title ``"race"`` from collaterally nuking a line that just
    happens to contain ``"embrace"``. Multi-word titles work because
    ``\b`` triggers at each space boundary.

    **Ultra-short titles (``len(title.strip()) < 3``) are not
    strip-eligible.** Single-letter and two-character titles like
    ``"a"``, ``"if"``, ``"or"`` would match every indefinite article
    or conjunction in prose even with ``\b`` anchoring — the
    word-boundary guard isn't enough on its own. Skipping these titles
    means the caller keeps a slightly-noisier summary (the bullet for
    the dropped "a" finding survives), which is strictly safer than
    shredding unrelated summary lines. A future FR with a structured
    prompt that keeps summary and findings orthogonal would remove the
    need for this heuristic entirely.

    Doesn't touch paragraph-style mentions that aren't on their own
    line — anchoring by line prevents collateral damage (dropping a
    paragraph that happens to contain a finding title in passing).
    """
    if not summary or not dropped:
        return summary
    drop_titles = [
        str(f.get("title") or "").strip()
        for f in dropped
        if isinstance(f, dict)
    ]
    # Guard against ultra-short titles that would match common words
    # (``"a"`` / ``"if"`` / ``"or"``) even under ``\b`` anchoring.
    # 3 chars is the minimum that empirically avoids the English
    # function-word collisions we've hit; shorter titles pass through
    # untouched rather than risk shredding unrelated summary prose.
    drop_titles = [t for t in drop_titles if len(t) >= 3]
    if not drop_titles:
        return summary
    # Precompile once — summaries have O(N_lines) scans and we'd
    # otherwise recompile per line.
    drop_patterns = [
        re.compile(rf"\b{re.escape(title)}\b") for title in drop_titles
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

        # Resolve severity_floor precedence up-front so a validation
        # error surfaces before we spend a provider call. Order matches
        # the FR (skill arg → .reviewer/config.yaml → built-in default).
        # The ``skill_arg`` step validates eagerly because the caller
        # typo'd; the config-layer step validates because operators
        # can typo their YAML too. Default is trusted (module constant).
        try:
            effective_floor = self._resolve_severity_floor(args, context)
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

        metadata = {**_as_dict(args.get("metadata")), "model": chosen_model}
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
        self, args: dict[str, Any], context: dict[str, Any]
    ) -> str:
        """Resolve the effective severity_floor per the FR precedence chain.

        High-to-low:

        1. Skill-arg ``severity_floor`` (non-empty string) — **strict**.
           A bad value here is a caller bug; raise
           :class:`SeverityFloorError` so the review fails fast with a
           message the caller can act on.
        2. ``review.severity_floor`` / ``checks.severity_floor`` in
           ``.reviewer/config.yaml`` — **lenient**. Only consulted when
           the context supplies ``repo_root`` + ``base_sha`` hints. A bad
           value in YAML shouldn't nuke every review for that repo — log
           a warning naming the offending value and fall through to the
           built-in default. Reviewing is more important than config-layer
           correctness.
        3. :data:`_DEFAULT_SEVERITY_FLOOR` (``"nit"`` — no filtering).

        Rationale for asymmetric validation: the skill-arg path is a
        programmatic caller (another agent, a test, an orchestrator) —
        strict failure is the correct feedback channel. The config-layer
        path is a human-edited YAML file on a repo; a typo there
        shouldn't silently wedge CI.
        """
        arg_value = args.get("severity_floor")
        if isinstance(arg_value, str) and arg_value:
            return _validate_severity_floor(arg_value, source="skill arg")

        config_value = _resolve_repo_severity_floor(context)
        if isinstance(config_value, str) and config_value:
            try:
                return _validate_severity_floor(
                    config_value, source=".reviewer/config.yaml"
                )
            except SeverityFloorError as exc:
                logger.warning(
                    "reviewer: ignoring invalid .reviewer/config.yaml "
                    "review.severity_floor=%r; falling back to default %r (%s)",
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
        so the ``findings_filtered_count`` field lands on the usage
        event even when no filtering occurred (value 0) — downstream
        analytics can't distinguish "filter ran, dropped nothing" from
        "filter didn't run" without an explicit zero.
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
        ollama_cfg = _as_dict(providers_cfg.get("ollama"))
        providers = {
            "claude_cli": ClaudeCliProvider(
                ClaudeCliProviderConfig(
                    binary=str(claude_cfg.get("binary") or "claude"),
                )
            ),
            "ollama": OllamaProvider(
                OllamaProviderConfig(
                    base_url=str(
                        ollama_cfg.get("base_url") or "http://localhost:11434/v1"
                    ),
                    default_model=str(config.get("default_model") or "qwen3.5"),
                )
            ),
        }
        return ProviderSelector(
            providers,
            SelectorConfig(
                default_backend=str(config.get("default_provider") or "ollama"),
                default_model=str(config.get("default_model") or "qwen3.5"),
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
