"""``.reviewer/prompts/`` directory loader (FR fr_reviewer_92453047).

Sibling to :mod:`reviewer.config.repo`. Loads repo-side calibration
material — severity rubric, per-kind/per-severity few-shot examples, and
a vendor-agnostic system preamble — from the **base-branch HEAD** of
the repository being reviewed. Same trust boundary as the rest of the
``.reviewer/`` surface: every read goes through ``git show
<base_sha>:.reviewer/prompts/<path>``; a PR cannot mutate its own
review's prompt by editing ``.reviewer/prompts/`` on the PR branch.

Directory layout::

    .reviewer/
      prompts/
        system_preamble.md          # vendor-agnostic; merged first
        severity_rubric.md          # severity calibration, cross-kind
        examples/
          <kind>/                   # e.g. "pr_diff", "spec"
            <severity>.md           # "nit.md", "comment.md", "concern.md"

Graceful absence
----------------
Missing ``.reviewer/prompts/`` is a silent no-op (same rule as the rest
of the ``.reviewer/`` loader, proposal §Resolved #13). Missing individual
files inside an otherwise-present ``prompts/`` directory also silently
skip — a repo that ships only a ``severity_rubric.md`` but no examples
gets exactly that, no warnings.

Parse errors
------------
Malformed inputs (a ``git show`` failure on a file that ``ls-tree``
reports, non-string content at the read boundary) silently drop just
the one file — :func:`reviewer.config.repo._git_show_text` returns
``None`` on a non-zero ``git show`` exit, and :func:`_read_optional`
passes that through. The rest of the load proceeds. Matching the
graceful-absence policy of :mod:`reviewer.config.repo`: operator typos
shouldn't block an entire review run, and a warning every review for
a missing file would just flood the log.

Assembly order
--------------
The loader does **not** assemble the final prompt here — it returns the
raw pieces in a :class:`RepoPrompts` dataclass. The prompt-assembly
layer (see :func:`reviewer.providers._prompt.build_review_prompt` and
its helper :func:`reviewer.providers._prompt._render_repo_prompts`)
does the actual ordering:

    built-in reviewer system prompt
      → repo ``system_preamble.md``
      → ``severity_rubric.md``
      → ``examples/<kind>/*`` (in severity order)
      → review task content

Keeping the loader storage-only and the assembly in the prompt module
means the wrapping logic (markdown / xml / json per model config) has
one home and the file-reading logic has another.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the trust-boundary primitives shipped with the .reviewer/
# config loader. Importing via the package namespace keeps the two
# modules logically siblings — if the git-show primitives ever get
# factored into their own module, both importers move together.
from reviewer.config.repo import (
    RepoConfigUnreachableError,
    _assert_base_sha_reachable,
    _git_show_text,
    _ls_tree,
)

__all__ = [
    "RepoConfigUnreachableError",
    "RepoPrompts",
    "load_repo_prompts",
]

logger = logging.getLogger(__name__)

#: Severity levels in the order they should be concatenated when
#: assembling the final prompt. Matches ``khonliang_reviewer.SEVERITY_ORDER``
#: — hard-coded here rather than imported to avoid a cross-package
#: coupling for a three-item tuple that is not going to change shape.
#: A mismatch against the library's order would be caught by the
#: examples-assembly tests.
_SEVERITY_ORDER: tuple[str, ...] = ("nit", "comment", "concern")


@dataclass(frozen=True)
class RepoPrompts:
    """Immutable snapshot of ``.reviewer/prompts/`` at a specific base SHA.

    All three fields default to "absent" (``None`` for the scalars, empty
    dict for examples). Consumers differentiate "no repo prompts at all"
    from "repo prompts present but this specific file is missing" only
    when they care to — both degrade gracefully.

    Trust boundary
    --------------
    Instances of this class travel from the agent into providers via
    :attr:`ReviewRequest.metadata` under the reserved
    ``_khonliang_repo_prompts`` key. The agent strips any
    ``_khonliang_*`` keys from caller-supplied metadata before merge
    (see :func:`reviewer.agent._strip_reserved_metadata`), so a
    provider that finds this key on a request can trust it's a
    ``RepoPrompts`` instance the agent built from a ``git show`` read
    against the configured base SHA — not caller-controlled bytes.
    """

    #: Contents of ``.reviewer/prompts/system_preamble.md``, or ``None``
    #: when the file is absent. Merged after the built-in reviewer
    #: system prompt and before the severity rubric.
    system_preamble: str | None = None

    #: Contents of ``.reviewer/prompts/severity_rubric.md``, or ``None``
    #: when absent. A single vendor-agnostic document; the FR commits
    #: to a cross-kind rubric rather than per-kind splits.
    severity_rubric: str | None = None

    #: Per-``(kind, severity)`` example text. Keys are tuples
    #: ``("pr_diff", "concern")``; values are the raw file contents.
    #: Consumers filter by ``kind`` at prompt-assembly time so a
    #: ``pr_diff`` review never sees ``spec`` examples (and vice versa).
    examples: dict[tuple[str, str], str] = field(default_factory=dict)

    #: Base SHA this snapshot was loaded from. Kept for diagnostics +
    #: symmetry with :class:`reviewer.config.repo.RepoConfig`.
    base_sha: str = ""

    @property
    def is_empty(self) -> bool:
        """True iff no prompt content was present on the base branch.

        Convenience for callers that want to skip the repo-prompts
        merge path entirely when nothing was loaded — marginally
        cheaper than walking an empty dataclass every time.
        """
        return (
            self.system_preamble is None
            and self.severity_rubric is None
            and not self.examples
        )

    def examples_for_kind(self, kind: str) -> list[tuple[str, str]]:
        """Return ``(severity, text)`` pairs for ``kind`` in severity order.

        Only returns severities the library knows about (`nit`, `comment`,
        `concern`). A stray ``examples/pr_diff/high.md`` — accidentally
        shipped, or from a future severity split — is dropped here: the
        rest of the pipeline expects the canonical severity set, and
        silently passing through unknowns would make the prompt wedge
        in ways that are hard to debug downstream.
        """
        out: list[tuple[str, str]] = []
        for severity in _SEVERITY_ORDER:
            text = self.examples.get((kind, severity))
            if text:
                out.append((severity, text))
        return out


def load_repo_prompts(
    repo_root: Path | str,
    *,
    base_sha: str,
    git_binary: str = "git",
) -> RepoPrompts:
    """Load ``.reviewer/prompts/`` from ``base_sha`` in ``repo_root``.

    Parameters mirror :func:`reviewer.config.repo.load` so callers can
    thread the same ``(repo_root, base_sha)`` pair through both loaders
    without a second round of argument plumbing.

    Raises
    ------
    RepoConfigUnreachableError:
        If ``base_sha`` cannot be read locally (shallow clone). Callers
        decide whether to fall back to the built-in prompt only or
        surface the infrastructure failure as a finding — same contract
        as the ``.reviewer/`` config loader.
    """
    root = Path(repo_root)

    # Probe base SHA reachability first so shallow-clone failures
    # surface with the targeted error message instead of a generic
    # "git show returned non-zero" downstream. Identical gating to
    # config.repo.load — keeps the two loaders behavior-compatible
    # for any caller that wraps both in a try/except.
    _assert_base_sha_reachable(root, base_sha, git_binary=git_binary)

    prompts_dir = ".reviewer/prompts"
    tree_entries = _ls_tree(root, base_sha, prompts_dir, git_binary=git_binary)
    if tree_entries is None:
        # Graceful-absence path: no ``.reviewer/prompts/`` at all. Same
        # silence policy as the .reviewer/ config loader — no warning,
        # no raise, no log noise. Callers get an empty RepoPrompts and
        # fall back to the built-in prompt.
        return RepoPrompts(base_sha=base_sha)

    system_preamble = _read_optional(
        root, base_sha, f"{prompts_dir}/system_preamble.md", git_binary=git_binary
    )
    severity_rubric = _read_optional(
        root, base_sha, f"{prompts_dir}/severity_rubric.md", git_binary=git_binary
    )

    examples: dict[tuple[str, str], str] = {}
    # Only descend into examples/ if it's listed in the prompts/ tree.
    # Avoids a speculative git call in the common case of a repo that
    # ships a rubric + preamble but no few-shot files yet.
    if "examples" in tree_entries:
        examples_dir = f"{prompts_dir}/examples"
        kind_entries = _ls_tree(
            root, base_sha, examples_dir, git_binary=git_binary
        ) or []
        for kind_name in kind_entries:
            kind_path = f"{examples_dir}/{kind_name}"
            severity_files = _ls_tree(
                root, base_sha, kind_path, git_binary=git_binary
            )
            if severity_files is None:
                # Entry listed under examples/ is not a tree (e.g. a
                # stray README). Ignored silently — same graceful-
                # absence rule; we don't want to log-spam a review
                # log for what is almost always a typo.
                continue
            for file_name in severity_files:
                if not file_name.endswith(".md"):
                    # Future-proof against operators dropping .txt or
                    # similar; ignored rather than guessed-about. Not
                    # logged — keeps the review log quiet in the
                    # common "I renamed it and forgot" case.
                    continue
                severity = file_name[: -len(".md")]
                if severity not in _SEVERITY_ORDER:
                    # Only accept the library's canonical severities.
                    # A stray ``high.md`` or ``warning.md`` is a
                    # silent no-op — the example dict key would never
                    # match an actual review's severity set, and
                    # surfacing a warning every review for a typo
                    # isn't worth the noise budget.
                    logger.debug(
                        ".reviewer/prompts/examples/%s/%s: unknown severity %r, skipping",
                        kind_name,
                        file_name,
                        severity,
                    )
                    continue
                text = _read_optional(
                    root,
                    base_sha,
                    f"{kind_path}/{file_name}",
                    git_binary=git_binary,
                )
                if text:
                    examples[(kind_name, severity)] = text

    return RepoPrompts(
        system_preamble=system_preamble,
        severity_rubric=severity_rubric,
        examples=examples,
        base_sha=base_sha,
    )


# -- internals ---------------------------------------------------------


def _read_optional(
    repo_root: Path,
    base_sha: str,
    rel_path: str,
    *,
    git_binary: str,
) -> str | None:
    """``git show`` wrapper that collapses whitespace-only files to None.

    An empty / whitespace-only prompt file is semantically "no content"
    — merging it into the final prompt as a blank line would just
    widen the prompt without adding signal. Collapsing it here means
    the assembly layer can key on "field present?" to decide whether
    to emit a section header, without a secondary strip-and-test step.
    """
    raw = _git_show_text(repo_root, base_sha, rel_path, git_binary=git_binary)
    if raw is None:
        return None
    if not raw.strip():
        return None
    return raw


# ``RepoConfigUnreachableError`` is re-exported (see ``__all__``) so
# downstream modules can import it from either ``reviewer.config.repo``
# or ``reviewer.config.prompts`` without knowing which one physically
# owns the primitive. Callers wrapping both loaders in a single
# try/except catch one class from one symbol either way.
