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

# Hard ceiling on any single bundle file (manifest.json / diff.patch).
# ``staging_handle`` deliberately bypasses the bus request-size limits —
# the whole point is to keep a large diff off the caller's MCP context —
# so the size guard against a malicious or runaway bundle has to live
# here, on the read side. 64 MiB is far above any real review diff.
MAX_BUNDLE_FILE_BYTES = 64 * 1024 * 1024

# ``os.open`` / ``os.lstat`` accept a ``dir_fd`` only on platforms that
# advertise it (POSIX yes, Windows no). When present we anchor every
# bundle read to an O_NOFOLLOW directory fd (openat semantics); when
# absent we fall back to resolved-path reads. Computed once at import.
_SUPPORTS_DIR_FD = os.open in getattr(os, "supports_dir_fd", set())


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
    if Path(bundle_id).is_absolute():
        # POSIX absolute paths are already caught by the separator check
        # above; this honors the documented contract and also rejects the
        # Windows drive / UNC shapes (``C:\...``, ``\\host\share``).
        raise StagingHandleError(
            f"staging_handle: bundle_id may not be absolute (got {bundle_id!r})"
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

    Security (defense in depth; ``bundle_id`` is untrusted):

    1. ``bundle_id`` is constrained to a single safe path segment
       (see :func:`_validate_bundle_id_segment`).
    2. The resolved bundle dir is verified to live under the resolved
       staging root before any open — a clear early reject for ``../``
       or symlink-to-outside bundle dirs.
    3. Reads are anchored to an ``O_NOFOLLOW`` directory fd opened *at*
       the staging root (openat semantics, where the platform supports
       ``dir_fd``). This closes the TOCTOU gap between the containment
       check and the read: a writable staging root can't swap
       ``<root>/<bundle_id>`` for a symlink after the check to redirect
       reads outside the root, and per-file ``O_NOFOLLOW`` rejects a
       symlinked ``manifest.json`` / ``diff.patch``.
    4. Each file is size-capped (:data:`MAX_BUNDLE_FILE_BYTES`) so a
       handle — which bypasses the bus request-size limit — can't OOM
       the reviewer.
    """
    if root is None:
        root = staging_root()
    _validate_bundle_id_segment(bundle_id)
    bundle_dir = root / bundle_id
    # Step 2: collapse any symlinks and confirm containment under the
    # root before any open. ``strict=False`` so a non-existent bundle
    # still reaches the not-found branch below with a clear error.
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

    # Step 3: open the bundle dir as an O_NOFOLLOW fd anchored at the
    # staging root, then read every file relative to that fd so an
    # intermediate symlink swap can't redirect access after the check.
    bundle_fd = _open_bundle_dir_fd(root, bundle_id, resolved_bundle)
    try:
        manifest_bytes = _read_bundle_file(
            bundle_fd, resolved_bundle, "manifest.json", "manifest"
        )
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StagingHandleError(
                f"fs bundle manifest unreadable in {bundle_dir}: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise StagingHandleError(
                f"fs bundle manifest in {bundle_dir} must be a JSON object"
            )
        kind = manifest.get("kind")
        if not isinstance(kind, str) or not kind:
            raise StagingHandleError(
                f"fs bundle manifest in {bundle_dir} missing required 'kind' string"
            )
        if kind != "diff":
            raise StagingNotImplemented(
                f"fs bundle kind={kind!r} not yet wired in reviewer; "
                "only 'diff' is implemented today (richer modes land per "
                "fr_khonliang-bus-lib_520ce3bf follow-ups)"
            )
        diff_bytes = _read_bundle_file(
            bundle_fd, resolved_bundle, "diff.patch", "diff.patch"
        )
    finally:
        if bundle_fd is not None:
            os.close(bundle_fd)
    return ResolvedBundle(kind=kind, diff_bytes=diff_bytes)


def _open_bundle_dir_fd(
    root: Path, bundle_id: str, resolved_bundle: Path
) -> int | None:
    """Open the bundle directory as an ``O_NOFOLLOW`` fd anchored at ``root``.

    Returns an open directory fd the caller must ``os.close``, or ``None``
    on platforms without ``dir_fd`` support (callers then read via the
    resolved path). The staging root itself is trusted configuration, so
    following symlinks to *reach* it is fine; only the untrusted
    ``bundle_id`` segment is opened ``O_NOFOLLOW | O_DIRECTORY`` relative
    to the root fd. A symlink swapped in for ``<root>/<bundle_id>`` after
    the containment check is therefore refused by the kernel (``ELOOP``)
    rather than followed.
    """
    if not _SUPPORTS_DIR_FD:
        return None
    dir_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        root_fd = os.open(str(root), dir_flags)
    except OSError as exc:
        raise StagingHandleError(
            f"fs staging root could not be opened at {root}: {exc}"
        ) from exc
    try:
        try:
            return os.open(
                bundle_id,
                dir_flags | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_fd,
            )
        except FileNotFoundError as exc:
            raise StagingHandleError(
                f"fs bundle not found at {resolved_bundle}"
            ) from exc
        except OSError as exc:
            # ELOOP (symlinked bundle dir) / ENOTDIR (bundle_id is a
            # file) / EACCES — surface as a clean staging error.
            raise StagingHandleError(
                f"fs bundle dir could not be opened at {resolved_bundle}: {exc}"
            ) from exc
    finally:
        os.close(root_fd)


def _read_bundle_file(
    bundle_fd: int | None,
    resolved_bundle: Path,
    name: str,
    label: str,
    *,
    max_bytes: int | None = None,
) -> bytes:
    """Read one regular bundle file: no symlink follow, no block, size-capped.

    ``bundle_fd`` is the ``O_NOFOLLOW`` directory fd from
    :func:`_open_bundle_dir_fd`; when it is ``None`` (no ``dir_fd``
    support) the file is read via its resolved path instead. Either way:

    1. ``lstat`` first (no follow, never blocks): reject symlink / FIFO /
       device / socket / dir with a clear error before ``os.open``.
    2. ``os.open`` with ``O_NOFOLLOW | O_NONBLOCK``: ``O_NOFOLLOW`` is the
       TOCTOU defense against a symlink swapped in after the lstat;
       ``O_NONBLOCK`` ensures a FIFO swapped in after the lstat returns
       immediately instead of blocking the reviewer, then fails the
       post-open ``fstat`` regular-file check.
    3. ``fstat`` on the open fd: authoritative regular-file check plus the
       size guard.
    4. bounded read (``max_bytes + 1``) so a file that grows between the
       fstat and the read is rejected, not silently truncated.
    """
    import stat as _stat

    # Read the cap at call time (not as a default arg) so it stays a
    # single live knob — tests monkeypatch the module constant.
    if max_bytes is None:
        max_bytes = MAX_BUNDLE_FILE_BYTES
    display = resolved_bundle / name
    if bundle_fd is not None:
        target: str = name
        fd_kwargs: dict[str, int] = {"dir_fd": bundle_fd}
    else:
        target = str(display)
        fd_kwargs = {}

    try:
        lst = os.lstat(target, **fd_kwargs)
    except FileNotFoundError as exc:
        raise StagingHandleError(
            f"fs bundle {label} not found at {display}"
        ) from exc
    except OSError as exc:
        raise StagingHandleError(
            f"fs bundle {label} unstatable at {display}: {exc}"
        ) from exc

    if _stat.S_ISLNK(lst.st_mode):
        raise StagingHandleError(
            f"fs bundle {label} at {display} is a symlink"
        )
    if not _stat.S_ISREG(lst.st_mode):
        raise StagingHandleError(
            f"fs bundle {label} at {display} is not a regular file "
            f"(mode={oct(lst.st_mode)})"
        )

    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)  # POSIX; absent on Windows
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(target, flags, **fd_kwargs)
    except OSError as exc:
        # ELOOP / EPERM / EACCES / TOCTOU race with a symlink swap —
        # surface as a clean staging error rather than letting the OS
        # error bubble as an internal-server-error shape.
        raise StagingHandleError(
            f"fs bundle {label} could not be opened at {display}: {exc}"
        ) from exc
    try:
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode):
            raise StagingHandleError(
                f"fs bundle {label} at {display} is not a regular file "
                f"(mode={oct(st.st_mode)})"
            )
        if st.st_size > max_bytes:
            raise StagingHandleError(
                f"fs bundle {label} at {display} too large: "
                f"{st.st_size} bytes exceeds {max_bytes}-byte cap"
            )
        with os.fdopen(fd, "rb", closefd=False) as f:
            # Read one past the cap so a file that grew between the fstat
            # and this read (TOCTOU) is rejected rather than truncated.
            data = f.read(max_bytes + 1)
    finally:
        os.close(fd)
    if len(data) > max_bytes:
        raise StagingHandleError(
            f"fs bundle {label} at {display} exceeds {max_bytes}-byte cap"
        )
    return data


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
