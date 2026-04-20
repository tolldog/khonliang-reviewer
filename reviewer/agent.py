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
import sys
import uuid
from typing import Any

from khonliang_bus import BaseAgent, Skill, handler
from khonliang_reviewer import ReviewRequest, ReviewResult, UsageEvent

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
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._injected_selector = selector
        self._cached_selector: ProviderSelector | None = None
        self._injected_store = usage_store
        self._cached_store: UsageStore | None = None

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
        await self._record_usage(result)
        return result.to_dict()

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

    @handler("usage_summary")
    async def handle_usage_summary(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return token + cost aggregates grouped by (backend, model)."""
        store = self._ensure_usage_store()
        summaries = store.summarize(
            backend=(args.get("backend") or None),
            model=(args.get("model") or None),
            since=_positive_float_or_none(args.get("since")),
            until=_positive_float_or_none(args.get("until")),
        )
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

        Publish failures (agent not connected to the bus, transport
        errors) log and are swallowed — they must not cause the caller's
        review to appear failed.
        """
        if result.usage is None:
            return
        store = self._ensure_usage_store()
        filled = store.back_fill_cost(result.usage)
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
    parser = argparse.ArgumentParser(description="khonliang-reviewer bus agent")
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
