"""Unit tests for :mod:`reviewer.staging` — staging_handle resolver.

Pairs with ``fr_reviewer_800e851d``. Covers handle-shape parsing,
``fs:`` bundle resolution from a tmp staging root, and the
forward-stable rejection paths (``store:`` reserved; unknown backends
and bundle kinds).

Integration with ``review_text`` / ``review_diff`` is exercised in
``test_agent_skills.py``; this file stays focused on the resolver
contract so failures point at the right layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reviewer import staging


def _write_bundle(
    root: Path,
    bundle_id: str,
    *,
    kind: str = "diff",
    diff_bytes: bytes = b"diff --git a/foo b/foo\n",
    omit_manifest: bool = False,
    bad_manifest: str | None = None,
    omit_diff: bool = False,
) -> Path:
    bundle_dir = root / bundle_id
    bundle_dir.mkdir(parents=True)
    if not omit_manifest:
        if bad_manifest is not None:
            (bundle_dir / "manifest.json").write_text(bad_manifest)
        else:
            (bundle_dir / "manifest.json").write_text(
                json.dumps({"kind": kind, "created_at": 1.0, "files": ["diff.patch"]})
            )
    if not omit_diff:
        (bundle_dir / "diff.patch").write_bytes(diff_bytes)
    return bundle_dir


# ---------------------------------------------------------------------------
# parse_handle
# ---------------------------------------------------------------------------


def test_parse_handle_fs_happy() -> None:
    assert staging.parse_handle("fs:8f3a2e1c-4d2a") == ("fs", "8f3a2e1c-4d2a")


def test_parse_handle_store_happy() -> None:
    assert staging.parse_handle("store:art_8f3a2e1c") == ("store", "art_8f3a2e1c")


def test_parse_handle_unknown_backend_returns_parsed_pair() -> None:
    """parse_handle validates shape only; backend support is resolve_*'s job."""
    assert staging.parse_handle("future:xyz") == ("future", "xyz")


@pytest.mark.parametrize(
    "handle",
    ["", "no-colon", ":missing-backend", "fs:", "fs::two-colons-ok"],
    ids=["empty", "no_colon", "empty_backend", "empty_id", "double_colon"],
)
def test_parse_handle_rejects_malformed(handle: str) -> None:
    if handle == "fs::two-colons-ok":
        # partition splits on the first ':' — backend='fs', ident=':two-colons-ok',
        # which is non-empty so parse_handle accepts. Documented as such; this
        # case asserts that behavior so future refactors keep the contract.
        assert staging.parse_handle(handle) == ("fs", ":two-colons-ok")
        return
    with pytest.raises(staging.StagingHandleError):
        staging.parse_handle(handle)


def test_parse_handle_rejects_non_string() -> None:
    with pytest.raises(staging.StagingHandleError):
        staging.parse_handle(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# resolve_fs
# ---------------------------------------------------------------------------


def test_resolve_fs_happy_path(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path, "abc", diff_bytes=b"diff --git a/foo b/foo\n")
    result = staging.resolve_fs("abc", root=tmp_path)
    assert result.kind == "diff"
    assert result.diff_bytes == b"diff --git a/foo b/foo\n"
    assert isinstance(bundle, Path)  # sanity: helper actually wrote the dir


def test_resolve_fs_missing_bundle(tmp_path: Path) -> None:
    with pytest.raises(staging.StagingHandleError, match="manifest not found"):
        staging.resolve_fs("nonexistent", root=tmp_path)


def test_resolve_fs_corrupt_manifest(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "bad", bad_manifest="not json {{")
    with pytest.raises(staging.StagingHandleError, match="unreadable"):
        staging.resolve_fs("bad", root=tmp_path)


def test_resolve_fs_manifest_must_be_object(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "bad", bad_manifest=json.dumps(["a", "list"]))
    with pytest.raises(staging.StagingHandleError, match="JSON object"):
        staging.resolve_fs("bad", root=tmp_path)


def test_resolve_fs_manifest_missing_kind(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "nokind", bad_manifest=json.dumps({"other": "x"}))
    with pytest.raises(staging.StagingHandleError, match="'kind'"):
        staging.resolve_fs("nokind", root=tmp_path)


def test_resolve_fs_unknown_kind_not_implemented(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "future", kind="changed-files")
    with pytest.raises(staging.StagingNotImplemented, match="changed-files"):
        staging.resolve_fs("future", root=tmp_path)


def test_resolve_fs_missing_diff_patch(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "nopatch", omit_diff=True)
    with pytest.raises(staging.StagingHandleError, match="diff.patch"):
        staging.resolve_fs("nopatch", root=tmp_path)


def test_resolve_fs_root_defaults_to_env(tmp_path: Path, monkeypatch) -> None:
    """When root=None, resolve_fs reads KHONLIANG_STAGING_ROOT."""
    _write_bundle(tmp_path, "env-default", diff_bytes=b"X")
    monkeypatch.setenv(staging.ENV_STAGING_ROOT, str(tmp_path))
    result = staging.resolve_fs("env-default")
    assert result.diff_bytes == b"X"


# ---------------------------------------------------------------------------
# resolve (dispatch)
# ---------------------------------------------------------------------------


def test_resolve_dispatches_fs(tmp_path: Path, monkeypatch) -> None:
    _write_bundle(tmp_path, "dispatch", diff_bytes=b"diff body")
    monkeypatch.setenv(staging.ENV_STAGING_ROOT, str(tmp_path))
    result = staging.resolve("fs:dispatch")
    assert result.kind == "diff"
    assert result.diff_bytes == b"diff body"


def test_resolve_store_backend_not_implemented() -> None:
    with pytest.raises(staging.StagingNotImplemented, match="store"):
        staging.resolve("store:art_123")


def test_resolve_unknown_backend_rejected() -> None:
    with pytest.raises(staging.StagingHandleError, match="unknown"):
        staging.resolve("ipfs:Qm123")


def test_resolve_malformed_handle_propagates_parse_error() -> None:
    with pytest.raises(staging.StagingHandleError):
        staging.resolve("no-colon")


# ---------------------------------------------------------------------------
# staging_root helper
# ---------------------------------------------------------------------------


def test_staging_root_default(monkeypatch) -> None:
    monkeypatch.delenv(staging.ENV_STAGING_ROOT, raising=False)
    assert staging.staging_root() == staging.DEFAULT_STAGING_ROOT


def test_staging_root_env_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(staging.ENV_STAGING_ROOT, str(tmp_path))
    assert staging.staging_root() == tmp_path


# ---------------------------------------------------------------------------
# Path-traversal hardening (pass-1 finding on reviewer/staging.py:105).
#
# The reviewer is a bus service that can be called by arbitrary clients;
# an attacker-controlled ``staging_handle`` must not be able to read
# files outside the staging root via ``../`` segments, absolute paths,
# or symlink swaps.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    [
        "../etc",
        "../../etc/passwd",
        "subdir/escape",
        "/absolute/path",
        ".",
        "..",
        ".tmp-half-written",
        ".hidden",
        "with\x00nul",
        "",
    ],
)
def test_resolve_fs_rejects_unsafe_bundle_id(tmp_path: Path, bad_id: str) -> None:
    """``bundle_id`` must be a single safe path segment. Anything that
    could traverse out of the staging root (``..``), reach in via an
    absolute path, point at a writer-side temp dir (``.tmp-...``), or
    smuggle a NUL byte must be rejected before any filesystem access.
    """
    with pytest.raises(staging.StagingHandleError):
        staging.resolve_fs(bad_id, root=tmp_path)


def test_resolve_fs_rejects_symlink_escape(tmp_path: Path) -> None:
    """Defense in depth: even if a bundle dir name passes the segment
    check, the resolved path must live under the resolved staging
    root. Symlink ``<root>/escape`` -> ``<tmp>/outside`` should be
    rejected, NOT followed.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "manifest.json").write_text(
        json.dumps({"kind": "diff", "created_at": 1.0})
    )
    (outside / "diff.patch").write_bytes(b"leaked")

    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    (staging_dir / "escape").symlink_to(outside)

    with pytest.raises(staging.StagingHandleError, match="escapes staging root"):
        staging.resolve_fs("escape", root=staging_dir)


def test_resolve_dispatch_rejects_traversal_handle(tmp_path: Path, monkeypatch) -> None:
    """Top-level ``resolve("fs:../etc")`` must reject before any fs read."""
    monkeypatch.setenv(staging.ENV_STAGING_ROOT, str(tmp_path))
    with pytest.raises(staging.StagingHandleError):
        staging.resolve("fs:../etc")


def test_is_within_helper(tmp_path: Path) -> None:
    """Sanity-check the internal containment predicate."""
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "child"
    inside.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    assert staging._is_within(inside.resolve(), root.resolve()) is True
    assert staging._is_within(root.resolve(), root.resolve()) is True
    assert staging._is_within(outside.resolve(), root.resolve()) is False
