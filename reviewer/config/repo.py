"""``.reviewer/`` directory loader (Milestone B.1).

Loads repo-side reviewer configuration from the **base-branch HEAD** of
the repository being reviewed. Every read goes through
``git show <base_sha>:<path>`` — never through a plain filesystem open
on the working tree. The trust boundary is load-bearing: a PR that
modifies ``.reviewer/instructions.md`` must **not** change the
reviewer's effective instructions for that PR (proposal §Resolved #4).

Directory layout the loader understands::

    .reviewer/
      config.yaml
      instructions.md
      models/
        <vendor>/
          <model>.yaml
          _default.yaml
      checks/     # reserved — paths enumerated but files not opened
      baselines/  # reserved — paths enumerated but files not opened

Graceful absence
----------------
Missing ``.reviewer/`` = built-in defaults returned, no error, no
log warning, no check disabled (proposal §Resolved #13). Callers
cannot distinguish "no ``.reviewer/`` present" from "empty
``.reviewer/config.yaml``" — both collapse to defaults. That's
intentional: we don't want a PR reviewer to branch on whether a repo
happens to have opted in.

Shallow clone
-------------
When ``base_sha`` is not reachable locally (typical CI default
``fetch-depth: 1``), :func:`load` raises :class:`RepoConfigUnreachableError`
with a message naming ``fetch-depth: 0`` as the fix. We deliberately do
**not** fall back to built-in defaults here — the caller (check
framework, review agent) decides whether the infrastructure failure
should surface as a ``concern`` finding or silently fall back. Symmetric
with the ``DiffContext`` failure-mode pattern from proposal §B.2.

Override chain
--------------
:meth:`RepoConfig.resolve` applies, high-to-low precedence:

1. Model-specific: ``models/<vendor>/<model>.yaml``
2. Vendor default: ``models/<vendor>/_default.yaml``
3. Repo-level: ``config.yaml``
4. Built-in defaults: :data:`BUILTIN_DEFAULTS`

Merge semantics are **shallow** — top-level keys overlay wholesale.
Nested-dict deep-merge is explicitly out of scope for this FR; proposal
flagged it as an open question and the MVP documents the shallow choice.
"""

from __future__ import annotations

import copy
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "BUILTIN_DEFAULTS",
    "RepoConfig",
    "RepoConfigUnreachableError",
    "ResolvedConfig",
    "load",
]

logger = logging.getLogger(__name__)

# Model-filename suffix for vendor defaults. Kept as a module constant
# so the "not a specific model" sentinel has one source of truth —
# the override-chain merge logic, the enumerator, and the tests all
# reference the same string.
_VENDOR_DEFAULT_STEM = "_default"


class RepoConfigUnreachableError(RuntimeError):
    """Raised when ``base_sha`` cannot be read locally (shallow clone).

    The message names ``fetch-depth: 0`` as the workflow fix so a
    human reading the traceback doesn't have to chase through layers
    of infrastructure docs. Callers deciding whether to emit a
    ``concern`` finding vs. fall back to defaults can catch this
    exception and key on the error class, not the message string.
    """


# Built-in defaults. Kept intentionally small — a follow-up FR can
# grow them once we know what knobs actually matter in practice.
# Anything a check, provider, or prompt wires in today should either
# be represented here with a sensible default or explicitly marked
# as "no default, caller must supply".
BUILTIN_DEFAULTS: dict[str, Any] = {
    "temperature": 0.2,
    "checks": {},
}


@dataclass(frozen=True)
class ResolvedConfig:
    """Effective config for a single ``(kind, vendor, model)`` lookup.

    ``values`` is the merged-down dict after applying the override chain.
    ``sources`` is an ordered list of the layers that contributed,
    high-to-low precedence — useful for debugging "why is my
    temperature 0.8 not 0.2?". Each source is a short label:
    ``"model:ollama/qwen2.5-coder"``, ``"vendor:ollama"``,
    ``"repo"``, ``"builtin"``.
    """

    kind: str
    vendor: str
    model: str
    values: dict[str, Any]
    sources: tuple[str, ...]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    @property
    def temperature(self) -> float | None:
        """Convenience accessor for the most-used knob.

        Returns ``None`` if the key is absent or not coercible. Callers
        that need strict typing should pull from ``values`` directly
        and validate at their own boundary.
        """
        val = self.values.get("temperature")
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class RepoConfig:
    """Resolved ``.reviewer/`` view of a repository at a specific base SHA.

    Immutable snapshot — caller can hold a reference across requests.
    All paths are strings relative to repo root (never absolute, never
    working-tree paths; callers should not pass them to ``open()``).
    """

    #: Contents of ``.reviewer/instructions.md`` on base branch, or
    #: ``None`` when the file is absent. Downstream consumers decide
    #: how to merge it into the LLM prompt — this FR only exposes it.
    instructions_text: str | None = None

    #: Raw parsed ``.reviewer/config.yaml`` (repo-level layer). Empty
    #: dict when the file is absent, present-but-empty, or parses as
    #: a non-mapping YAML root.
    repo_yaml: dict[str, Any] = field(default_factory=dict)

    #: Parsed model configs, keyed by ``(vendor, model_stem)``. The
    #: vendor-default entry uses stem ``_default``. Never ``None``;
    #: vendors with no configured models simply don't appear.
    model_yamls: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    #: Paths (repo-relative) under ``.reviewer/checks/``. Directory
    #: contents are enumerated but **not opened** — the checks loader
    #: is a separate FR. Empty tuple when the directory is absent.
    custom_checks_paths: tuple[str, ...] = ()

    #: Paths (repo-relative) under ``.reviewer/baselines/``. Same
    #: contract as ``custom_checks_paths``: enumerated, not opened.
    #: Empty tuple when the directory is absent.
    baseline_paths: tuple[str, ...] = ()

    #: Base SHA this snapshot was loaded from. Kept for diagnostics +
    #: so consumers that need to cross-reference another ``git show``
    #: don't have to re-plumb it.
    base_sha: str = ""

    @property
    def severity_floor(self) -> str | None:
        """Return the repo-declared severity floor, or ``None`` when unset.

        Reads ``review.severity_floor`` first, falling back to
        ``checks.severity_floor`` for forward-compat with the FR's
        original naming. The ``review.*`` form is preferred: the floor
        is a review-output knob, not a check-definition knob (checks
        live under their own config namespace). Either key returns the
        same string; the two-way lookup is just so an early adopter who
        wrote ``checks.severity_floor`` doesn't silently get the
        default.

        Returns ``None`` (not the empty string) when the key is absent
        or the value isn't a string — the reviewer's precedence chain
        distinguishes "no repo-level override" from "repo-level override
        is the empty string (fall through to default)", so a single
        falsy sentinel would collapse those cases together.
        """
        for key in ("review", "checks"):
            section = self.repo_yaml.get(key)
            if not isinstance(section, dict):
                continue
            floor = section.get("severity_floor")
            if isinstance(floor, str) and floor:
                return floor
        return None

    def resolve(self, kind: str, vendor: str, model: str) -> ResolvedConfig:
        """Apply the override chain for a concrete ``(kind, vendor, model)``.

        Shallow merge: top-level keys from higher-precedence layers
        overlay lower layers wholesale. ``kind`` is currently carried
        through for provenance but doesn't participate in merging — a
        future FR can add a ``kinds:`` sub-map to the repo YAML if
        per-kind overrides start mattering. Including it in the
        signature now means we don't have to change callers later.

        The ``_default`` sentinel is reserved for the vendor-default
        layer; passing it as ``model`` is nonsensical (it would try
        to resolve the placeholder as a real model). The caller almost
        always got the model string from a real request, so treat
        it as a programming error.
        """
        if model == _VENDOR_DEFAULT_STEM:
            raise ValueError(
                f"model={_VENDOR_DEFAULT_STEM!r} is the vendor-default "
                "sentinel, not a real model identifier"
            )

        # Start at the lowest-precedence layer and overlay upwards so
        # higher layers win. Collecting ``sources`` in build order then
        # reversing at the end gives the caller a high-to-low debug
        # view consistent with the documented override chain.
        #
        # Deep-copy every layer as it's folded in. Merge stays shallow
        # at the top level (documented semantics), but inner container
        # values (e.g. a nested ``checks:`` dict) get their own copies
        # so a caller mutating ``resolved.values["checks"]["x"] = ...``
        # can never leak into the next ``resolve()`` call or the
        # module globals. Cheap: configs are small and resolve is not
        # a hot path.
        merged: dict[str, Any] = {}
        sources: list[str] = []

        merged.update(copy.deepcopy(BUILTIN_DEFAULTS))
        sources.append("builtin")

        if self.repo_yaml:
            merged.update(copy.deepcopy(self.repo_yaml))
            sources.append("repo")

        vendor_default = self.model_yamls.get((vendor, _VENDOR_DEFAULT_STEM))
        if vendor_default:
            merged.update(copy.deepcopy(vendor_default))
            sources.append(f"vendor:{vendor}")

        model_stem = _model_stem(model)
        model_specific = self.model_yamls.get((vendor, model_stem))
        if model_specific:
            merged.update(copy.deepcopy(model_specific))
            sources.append(f"model:{vendor}/{model_stem}")

        # Reverse so the highest-precedence source is first — matches
        # how humans read an override chain ("the model file wins,
        # then vendor default, then repo, then builtin").
        sources.reverse()
        return ResolvedConfig(
            kind=kind,
            vendor=vendor,
            model=model,
            values=merged,
            sources=tuple(sources),
        )


def load(
    repo_root: Path | str,
    *,
    base_sha: str,
    git_binary: str = "git",
) -> RepoConfig:
    """Load ``.reviewer/`` from ``base_sha`` in ``repo_root``.

    Parameters
    ----------
    repo_root:
        Path to the git repository being reviewed. Must contain
        ``.git/``; no sentinel walk is performed here (the caller
        knows which repo they mean to review).
    base_sha:
        Commit SHA (or any revspec ``git show`` accepts) to read
        ``.reviewer/`` from. In practice this is the PR base branch
        HEAD. Passing the PR branch tip here would break the trust
        boundary and is the caller's bug to avoid.
    git_binary:
        Override for the ``git`` binary used. Tests use this to pin
        against a known build; operators shouldn't need to.

    Raises
    ------
    RepoConfigUnreachableError:
        If ``base_sha`` cannot be read locally (shallow clone). The
        message names ``fetch-depth: 0`` as the fix.
    """
    root = Path(repo_root)

    # Probe reachability first. If `git cat-file -e <sha>` fails we
    # want a targeted error, not a generic "git show returned
    # non-zero" that the caller has to reverse-engineer. This also
    # catches the case where repo_root isn't a git repo at all —
    # `cat-file` errors out the same way and the error message makes
    # the distinction clear via stderr plumbing.
    _assert_base_sha_reachable(root, base_sha, git_binary=git_binary)

    # Enumerate the .reviewer/ tree at base_sha. If the directory is
    # absent we short-circuit to defaults — that's the graceful-absence
    # path, and it must stay silent (no warn, no raise).
    tree_entries = _ls_tree(root, base_sha, ".reviewer", git_binary=git_binary)
    if tree_entries is None:
        return RepoConfig(base_sha=base_sha)

    instructions_text = _git_show_text(
        root, base_sha, ".reviewer/instructions.md", git_binary=git_binary
    )

    repo_yaml = _parse_yaml_mapping(
        _git_show_text(
            root, base_sha, ".reviewer/config.yaml", git_binary=git_binary
        ),
        label=".reviewer/config.yaml",
    )

    model_yamls: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in tree_entries:
        # Only descend into models/. Everything else in the tree (e.g.
        # config.yaml, instructions.md) is handled by explicit reads
        # above; reserved dirs (checks/, baselines/) are enumerated
        # separately to keep that intent obvious in the diff.
        if entry != "models":
            continue
        for vendor_entry in _ls_tree(
            root, base_sha, ".reviewer/models", git_binary=git_binary
        ) or []:
            vendor_path = f".reviewer/models/{vendor_entry}"
            # Vendor must be a directory. A stray file directly under
            # models/ is ignored (silently — same graceful-absence
            # rule). Callers catching "my config isn't applying"
            # should look at their layout; we don't want log noise
            # every review for what's almost always a typo.
            files = _ls_tree(root, base_sha, vendor_path, git_binary=git_binary)
            if files is None:
                continue
            for file_entry in files:
                if not file_entry.endswith(".yaml"):
                    continue
                stem = file_entry[: -len(".yaml")]
                rel_path = f"{vendor_path}/{file_entry}"
                parsed = _parse_yaml_mapping(
                    _git_show_text(root, base_sha, rel_path, git_binary=git_binary),
                    label=rel_path,
                )
                model_yamls[(vendor_entry, stem)] = parsed

    custom_checks_paths = _enumerate_reserved_dir(
        root, base_sha, ".reviewer/checks", git_binary=git_binary
    )
    baseline_paths = _enumerate_reserved_dir(
        root, base_sha, ".reviewer/baselines", git_binary=git_binary
    )

    return RepoConfig(
        instructions_text=instructions_text,
        repo_yaml=repo_yaml,
        model_yamls=model_yamls,
        custom_checks_paths=custom_checks_paths,
        baseline_paths=baseline_paths,
        base_sha=base_sha,
    )


# -- internals ----------------------------------------------------------


def _model_stem(model: str) -> str:
    """Strip a trailing ``:tag`` from a model identifier.

    Ollama-style models are commonly written as ``qwen2.5-coder:14b`` in
    caller code, but the on-disk config file is named
    ``qwen2.5-coder.yaml`` (the vendor doesn't get a separate file per
    tag in the MVP). Strip the tag when resolving; a follow-up FR can
    introduce per-tag files if the need actually materializes.
    """
    return model.split(":", 1)[0]


def _run_git(
    repo_root: Path,
    *args: str,
    git_binary: str = "git",
) -> subprocess.CompletedProcess[str]:
    """Run git with ``text=True`` + captured streams.

    Caller handles returncode. No timeout: these are local read-only
    operations against an on-disk repo; a hang here is a bigger
    problem than a missed review, and surfacing it via a test hang
    (rather than a silent timeout fallback) is the correct failure
    mode for CI.
    """
    return subprocess.run(
        [git_binary, "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_base_sha_reachable(
    repo_root: Path, base_sha: str, *, git_binary: str
) -> None:
    """Raise :class:`RepoConfigUnreachableError` if ``base_sha`` isn't local.

    Uses ``git cat-file -e <sha>`` — the canonical "does this object
    exist" probe. On a shallow clone the SHA is simply absent from the
    object store and cat-file exits non-zero; on a full clone it
    succeeds silently.
    """
    result = _run_git(
        repo_root, "cat-file", "-e", f"{base_sha}^{{commit}}", git_binary=git_binary
    )
    if result.returncode != 0:
        raise RepoConfigUnreachableError(
            f"base SHA {base_sha!r} is not reachable in {repo_root}; "
            "this usually means a shallow clone. Set `fetch-depth: 0` "
            "on the checkout step (or explicitly fetch the base SHA "
            f"before invoking the reviewer). git stderr: {result.stderr.strip()!r}"
        )


def _git_show_text(
    repo_root: Path,
    base_sha: str,
    rel_path: str,
    *,
    git_binary: str,
) -> str | None:
    """Return ``git show <base_sha>:<rel_path>`` as text, or None if absent.

    Absence is the non-exceptional case (most repos won't have every
    optional file); errors don't need to distinguish "doesn't exist"
    from "is a tree not a blob" — both map to None and the caller
    handles the absence gracefully. A genuine git-subsystem failure
    still surfaces via the earlier ``_assert_base_sha_reachable`` probe.
    """
    result = _run_git(
        repo_root, "show", f"{base_sha}:{rel_path}", git_binary=git_binary
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _ls_tree(
    repo_root: Path,
    base_sha: str,
    rel_path: str,
    *,
    git_binary: str,
) -> list[str] | None:
    """List immediate children of ``rel_path`` at ``base_sha``.

    Returns the list of entry basenames, or ``None`` when the path
    doesn't exist as a tree at that SHA. Trailing slash is required
    on the ls-tree arg so git treats it as "contents of", not the
    tree object itself; we add it here so callers don't have to
    remember.
    """
    result = _run_git(
        repo_root,
        "ls-tree",
        "--name-only",
        base_sha,
        f"{rel_path}/",
        git_binary=git_binary,
    )
    if result.returncode != 0:
        return None
    entries: list[str] = []
    for line in result.stdout.splitlines():
        # `git ls-tree` prints full paths relative to repo root. We
        # want just the basename to match caller intuition ("what's
        # in this directory?"). Stripping the prefix is safe: git
        # guarantees the prefix matches the arg we passed.
        prefix = f"{rel_path}/"
        if line.startswith(prefix):
            tail = line[len(prefix):]
            # Only include immediate children — skip anything with a
            # further slash (git ls-tree without `-r` already does
            # this, but guard defensively in case future git changes
            # default behavior).
            if "/" not in tail and tail:
                entries.append(tail)
    return entries


def _enumerate_reserved_dir(
    repo_root: Path,
    base_sha: str,
    rel_path: str,
    *,
    git_binary: str,
) -> tuple[str, ...]:
    """List every file (recursively) under a reserved directory.

    Reserved directories (``checks/``, ``baselines/``) are enumerated
    but not opened — this FR just exposes the paths so downstream
    loaders (separate FRs) know what to read. Recursive because those
    loaders may want subdirectory grouping once they ship, and
    exposing only top-level would force them to re-enumerate.
    """
    result = _run_git(
        repo_root,
        "ls-tree",
        "-r",
        "--name-only",
        base_sha,
        f"{rel_path}/",
        git_binary=git_binary,
    )
    if result.returncode != 0:
        return ()
    # Return repo-relative paths verbatim. Callers that want just the
    # basename can split; callers that want to feed it to another git
    # show already have the right shape.
    return tuple(line for line in result.stdout.splitlines() if line)


def _parse_yaml_mapping(raw: str | None, *, label: str) -> dict[str, Any]:
    """Parse ``raw`` as a YAML mapping, returning ``{}`` on absence/error.

    Four collapse-to-empty cases, all silent:

    - ``raw is None`` — file absent at base SHA.
    - Empty / whitespace-only / ``null`` YAML — present but opted-out.
    - Parse error — operator typo; logged at debug so it's diagnosable
      without spamming a review log with every malformed config.
    - Non-mapping root (list, scalar) — wrong shape; logged at debug.

    The silence policy is load-bearing (graceful-absence rule):
    callers must not be able to distinguish these cases. Downstream
    lint / validation is a follow-up FR.
    """
    if raw is None:
        return {}
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.debug("%s failed to parse as YAML: %s", label, exc)
        return {}
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        logger.debug(
            "%s root must be a mapping; got %s", label, type(loaded).__name__
        )
        return {}
    return loaded


