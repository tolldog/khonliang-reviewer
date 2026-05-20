"""Resolve ``staging_handle`` tokens produced by ``kh-stage`` to bundle bytes.

Pattern (closes ``fr_reviewer_800e851d``; pairs with
``fr_khonliang-bus-lib_520ce3bf`` on the writer side): the caller hands the
reviewer an opaque ``<backend>:<id>`` token instead of the raw diff. The
reviewer reads the bundle on its side, so the caller never pays diff-size
token cost on its MCP context.

Today's only wired backend is ``fs`` (group-readable bundles under
``/var/lib/khonliang/staging/<uuid>/``). ``store:`` is reserved for the
future store-agent staging surface — handles parse, but resolution raises
:class:`StagingNotImplemented` so the surface is forward-stable.

The staging root is configurable via the ``KHONLIANG_STAGING_ROOT`` env
var (default ``/var/lib/khonliang/staging``). Tests and dev environments
that can't write to ``/var/lib`` override via the env var; production
overrides via the systemd unit's ``Environment=`` line if a non-default
location is needed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple

DEFAULT_STAGING_ROOT = Path("/var/lib/khonliang/staging")
ENV_STAGING_ROOT = "KHONLIANG_STAGING_ROOT"


class StagingHandleError(ValueError):
    """Raised when a staging_handle is malformed or the bundle can't be read."""


class StagingNotImplemented(NotImplementedError):
    """Raised when the handle's backend or bundle kind isn't wired yet.

    Distinct from :class:`StagingHandleError` so callers can render
    "we know what you meant; come back when X lands" differently from
    "the handle itself is bad."
    """


class ResolvedBundle(NamedTuple):
    """The output of ``resolve()`` — what the review pipeline needs.

    ``kind`` carries the manifest's bundle kind verbatim (``"diff"`` today;
    ``"changed-files"`` / ``"module"`` / ``"repo"`` / ``"with-context"``
    once kh-stage implements them) so the caller can route based on
    bundle shape without re-reading the manifest.
    """

    kind: str
    diff_bytes: bytes


def staging_root() -> Path:
    """Return the active staging root.

    Read from the env var on every call (no caching) so test fixtures
    that ``monkeypatch.setenv(...)`` per-test are honored.
    """
    return Path(os.environ.get(ENV_STAGING_ROOT, str(DEFAULT_STAGING_ROOT)))


def _is_within(candidate: Path, root: Path) -> bool:
    """True iff ``candidate`` is ``root`` or a descendant of it.

    Uses ``Path.relative_to`` (3.9-compatible) rather than ``is_relative_to``
    (3.9+) for portability, then verifies the result doesn't escape via
    ``..`` segments. Both args must already be ``.resolve()``-d by the
    caller for symlink resistance.
    """
    try:
        rel = candidate.relative_to(root)
    except ValueError:
        return False
    # ``relative_to`` raises on non-prefix paths; defense in depth
    # against any future Path bug that lets ``..`` slip through.
    return ".." not in rel.parts


def parse_handle(handle: str) -> tuple[str, str]:
    """Split ``<backend>:<id>`` into ``(backend, id)``.

    Validates only the *shape*; whether the backend is supported or the
    id resolvable is resolve_*'s job.
    """
    if not isinstance(handle, str) or not handle:
        raise StagingHandleError("staging_handle must be a non-empty string")
    if ":" not in handle:
        raise StagingHandleError(
            f"staging_handle missing '<backend>:' prefix: {handle!r}"
        )
    backend, _, ident = handle.partition(":")
    if not backend or not ident:
        raise StagingHandleError(
            f"staging_handle has empty backend or id: {handle!r}"
        )
    return backend, ident


def _validate_bundle_id_segment(bundle_id: str) -> None:
    """Reject bundle_ids that could escape the staging root.

    The reviewer is a service that can be called by arbitrary bus
    clients; an attacker who can submit ``staging_handle`` values
    could try ``fs:../../etc`` or ``fs:/abs/path`` to make
    ``resolve_fs`` read ``/etc/manifest.json`` and ``/etc/diff.patch``.
    We constrain ``bundle_id`` to a single safe path segment:

    - non-empty
    - no path separators (``/`` or ``\\``)
    - not equal to ``.`` or ``..``
    - not starting with ``.`` (excludes both ``..`` and the
      ``.tmp-<uuid>`` half-written bundles that kh-stage uses)
    - not absolute (no leading ``/`` on POSIX; ``Path.is_absolute()``
      catches Windows shapes too)

    The check is shape-only — bundle_id doesn't have to be a UUID.
    Defense-in-depth path-containment is applied by
    :func:`resolve_fs` after constructing the final ``Path``.
    """
    if not bundle_id:
        raise StagingHandleError("staging_handle: bundle_id is empty")
    if "/" in bundle_id or "\\" in bundle_id:
        raise StagingHandleError(
            f"staging_handle: bundle_id must be a single path segment "
            f"(got {bundle_id!r} containing path separator)"
        )
    if bundle_id in (".", ".."):
        raise StagingHandleError(
            f"staging_handle: bundle_id may not be {bundle_id!r}"
        )
    if bundle_id.startswith("."):
        # Reject both ".." (caught above), ".tmp-<uuid>" half-written
        # bundles, and any other dot-prefixed name that suggests a
        # leak from the writer side.
        raise StagingHandleError(
            f"staging_handle: bundle_id may not start with '.' "
            f"(got {bundle_id!r})"
        )
    # NUL byte is invalid in POSIX paths; reject pre-emptively so a
    # clever payload doesn't end up with a truncated open() call.
    if "\x00" in bundle_id:
        raise StagingHandleError(
            "staging_handle: bundle_id contains NUL byte"
        )


def resolve_fs(bundle_id: str, *, root: Path | None = None) -> ResolvedBundle:
    """Read an ``fs:`` bundle from the staging root.

    Bundle layout (set by ``kh-stage``):
        <root>/<bundle_id>/manifest.json    {"kind": "diff", ...}
        <root>/<bundle_id>/diff.patch       raw unified-diff bytes

    Today only ``kind="diff"`` is wired. Other kinds parse correctly but
    raise :class:`StagingNotImplemented` so reviewer adoption can lag
    kh-stage by one ship cycle for each new depth mode.

    Security: ``bundle_id`` is constrained to a single safe path segment
    (see :func:`_validate_bundle_id_segment`) and the resolved bundle
    dir is checked to live under the staging root before any read,
    so an attacker-controlled handle can't escape via ``../`` or a
    symlink swap.
    """
    if root is None:
        root = staging_root()
    _validate_bundle_id_segment(bundle_id)
    bundle_dir = root / bundle_id
    # Defense in depth: even with the segment check above, a clever
    # symlink under the staging root could redirect the bundle_dir
    # resolution outside the root. ``Path.resolve()`` collapses any
    # symlinks; if the result still lives under the resolved root, the
    # access is safe. ``strict=False`` so a non-existent bundle still
    # reaches the "manifest not found" branch (clearer error) rather
    # than raising FileNotFoundError here.
    try:
        resolved_bundle = bundle_dir.resolve(strict=False)
        resolved_root = root.resolve(strict=False)
    except OSError as exc:
        raise StagingHandleError(
            f"fs bundle path could not be resolved: {exc}"
        ) from exc
    if not _is_within(resolved_bundle, resolved_root):
        raise StagingHandleError(
            f"fs bundle path escapes staging root: bundle_id={bundle_id!r}"
        )
    manifest_path = bundle_dir / "manifest.json"
    manifest_bytes = _read_regular_file_nofollow(manifest_path, "manifest")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StagingHandleError(
            f"fs bundle manifest unreadable at {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise StagingHandleError(
            f"fs bundle manifest at {manifest_path} must be a JSON object"
        )
    kind = manifest.get("kind")
    if not isinstance(kind, str) or not kind:
        raise StagingHandleError(
            f"fs bundle manifest at {manifest_path} missing required 'kind' string"
        )
    if kind != "diff":
        raise StagingNotImplemented(
            f"fs bundle kind={kind!r} not yet wired in reviewer; "
            "only 'diff' is implemented today (richer modes land per "
            "fr_khonliang-bus-lib_520ce3bf follow-ups)"
        )
    diff_path = bundle_dir / "diff.patch"
    diff_bytes = _read_regular_file_nofollow(diff_path, "diff.patch")
    return ResolvedBundle(kind=kind, diff_bytes=diff_bytes)


def _read_regular_file_nofollow(path: Path, label: str) -> bytes:
    """Read a regular file without following symlinks or blocking.

    Defense beyond the bundle-dir containment check: even if the
    bundle dir itself resolves cleanly under the staging root, a
    *file inside* the dir could be a symlink pointing at
    ``/etc/passwd`` or a FIFO that blocks the reviewer indefinitely.

    Order matters:

    1. ``lstat`` first (does not follow symlinks): reject anything
       that isn't a regular file — symlink, FIFO, device, socket,
       directory. This catches FIFOs *before* ``os.open``, which
       would otherwise block forever waiting for a writer.
    2. ``os.open(..., O_NOFOLLOW)``: TOCTOU defense — between the
       lstat and the open, a symlink could be swapped in; the
       kernel refuses to follow it (``ELOOP``).
    3. ``fstat`` once more on the open fd: belt-and-suspenders
       against a race that swapped the inode after the lstat passed
       but before the open succeeded.
    """
    import stat as _stat

    try:
        lst = os.lstat(str(path))
    except FileNotFoundError as exc:
        raise StagingHandleError(
            f"fs bundle {label} not found at {path}"
        ) from exc
    except OSError as exc:
        raise StagingHandleError(
            f"fs bundle {label} unstatable at {path}: {exc}"
        ) from exc

    if _stat.S_ISLNK(lst.st_mode):
        raise StagingHandleError(
            f"fs bundle {label} at {path} is a symlink"
        )
    if not _stat.S_ISREG(lst.st_mode):
        raise StagingHandleError(
            f"fs bundle {label} at {path} is not a regular file "
            f"(mode={oct(lst.st_mode)})"
        )

    flags = os.O_RDONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)  # POSIX; absent on Windows
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError as exc:
        # ELOOP / EPERM / EACCES / TOCTOU race with a symlink swap —
        # surface as a clean staging error rather than letting the OS
        # error bubble as an internal-server-error shape.
        raise StagingHandleError(
            f"fs bundle {label} could not be opened at {path}: {exc}"
        ) from exc
    try:
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode):
            raise StagingHandleError(
                f"fs bundle {label} at {path} is not a regular file "
                f"(mode={oct(st.st_mode)})"
            )
        with os.fdopen(fd, "rb", closefd=False) as f:
            return f.read()
    finally:
        os.close(fd)


def resolve(handle: str, *, root: Path | None = None) -> ResolvedBundle:
    """Resolve any backend's handle to a :class:`ResolvedBundle`.

    Dispatch table:
        ``fs:<uuid>``    -> :func:`resolve_fs`
        ``store:<id>``   -> :class:`StagingNotImplemented` (reserved; lands
                            with the store-agent staging surface)
        anything else    -> :class:`StagingHandleError`
    """
    backend, ident = parse_handle(handle)
    if backend == "fs":
        return resolve_fs(ident, root=root)
    if backend == "store":
        raise StagingNotImplemented(
            "staging_handle backend='store' lands when the store agent "
            "grows a byte-shaped staging surface; see fr_reviewer_800e851d"
        )
    raise StagingHandleError(
        f"unknown staging_handle backend {backend!r} "
        "(supported: fs; reserved: store)"
    )
