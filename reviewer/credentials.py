"""Credential discovery for the reviewer agent.

Each credential type has one function that returns the token (or
``None``) after walking a priority chain of canonical sources. Callers
never look inside keyrings, ~/.netrc files, or vendor config files
themselves — they ask this module and trust the result.

The design principle is **delegate, don't re-implement**: each
external tool (gh, claude, etc.) already handles its own
platform-specific credential storage. We invoke the tool's CLI to
extract the current token rather than trying to crack the storage
ourselves. That keeps this module portable across generic unix
environments (Linux + macOS + BSD) without per-OS branches.

When the second consumer appears, this module promotes to a separate
``khonliang-credentials-lib`` package (see
``fr_reviewer_XXXXXXXX`` once filed). Until then, a single-module
surface inside the reviewer keeps blast radius small.

Hygiene:

- Nothing in this module logs or persists a returned token.
- All subprocess calls run with captured stdout + stderr so a leaked
  token never echoes to an inherited TTY.
- Callers should treat return values as ephemeral; never cache them.
"""

from __future__ import annotations

import logging
import os
import subprocess

__all__ = ["get_github_token"]


logger = logging.getLogger(__name__)


#: Env vars checked before shelling out to ``gh``. Order matches the gh
#: CLI's own precedence, so our chain does not disagree with gh about
#: which token "wins" when both are set.
_GITHUB_ENV_VARS = ("GITHUB_TOKEN", "GH_TOKEN")


def get_github_token() -> str | None:
    """Return a GitHub OAuth token, or ``None`` if none can be discovered.

    Discovery order (each step is skipped silently if the source is
    empty / unreadable):

    1. ``GITHUB_TOKEN`` environment variable.
    2. ``GH_TOKEN`` environment variable (``gh``'s own override).
    3. ``gh auth token --hostname github.com`` subprocess — pulls the
       token from whatever keyring/keychain/credential store gh is
       configured to use on this host. Portable across unix variants
       because gh handles the platform-specific details.

    The resulting string is never cached or logged; every call reruns
    the discovery so a rotated token is picked up without restart.
    """
    for name in _GITHUB_ENV_VARS:
        value = os.environ.get(name)
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            return stripped
        # Whitespace-only env vars are treated as "not set" so shell
        # quoting mishaps (``export GITHUB_TOKEN=""``) don't silently
        # block the gh-CLI fallback.
    return _gh_auth_token()


def _gh_auth_token() -> str | None:
    """Ask ``gh`` for the github.com token via its own CLI.

    Returns ``None`` (with a debug-level log) whenever gh is not
    installed, the user is logged out, or the subprocess fails for any
    reason. Silence is intentional — callers decide whether to treat
    "no token" as an error.
    """
    try:
        proc = subprocess.run(
            ["gh", "auth", "token", "--hostname", "github.com"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except FileNotFoundError:
        logger.debug("gh CLI not installed; skipping gh auth token lookup")
        return None
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("gh auth token lookup failed: %s", exc)
        return None

    if proc.returncode != 0:
        # Logged out / no credentials for github.com — not an error for us,
        # the caller may be fine unauthenticated for public reads.
        logger.debug(
            "gh auth token returned %s (stderr omitted to avoid leaking state)",
            proc.returncode,
        )
        return None

    token = proc.stdout.strip()
    return token or None
