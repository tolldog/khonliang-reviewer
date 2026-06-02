#!/usr/bin/env bash
#
# Deploy a khonliang agent from its GitHub source clone into its service venv,
# then restart it on the bus.
#
# Deploy model (per operator): pull the source clone from GitHub, reinstall the
# package into the (copied, non-editable) service venv, and ONLY if pip
# requirements changed also sync dependencies -- all as the venv-owning user.
# Restart is bus-managed (no systemd).
#
# The service venv has no setuptools, so `pip install <sourcetree>` builds via
# pip's build isolation, which fetches setuptools from PyPI. The box must be
# online; the script fails loudly otherwise rather than carrying an offline path.
#
# Parametrized (vars below, override via environment) so it copies cleanly to the
# sibling agents (developer / researcher / ...).
#
# Usage:
#   scripts/deploy.sh [--ref <branch>] [--dry-run] [--no-restart] [--verify-review]
#
#   --ref <branch>    git ref to deploy (default: main)
#   --dry-run         print the mutating commands without running them
#   --no-restart      pull + reinstall but leave the running process untouched
#   --verify-review   after restart, run a large-diff review through the bus and
#                     assert it is no longer truncated at 4096 input tokens
#
set -euo pipefail

# ---- config (override via environment) ------------------------------------
AGENT_ID="${AGENT_ID:-reviewer-primary}"
SRC="${SRC:-/opt/khonliang/src/khonliang-reviewer}"
VENV="${VENV:-/opt/khonliang/agents/reviewer/.venv}"
BUS="${BUS:-http://localhost:8788}"
REF="${REF:-main}"
DEPLOY_USER="${DEPLOY_USER:-khonliang}"
# An import that succeeds only when the freshly-installed code is present. Acts
# as a post-install smoke so a half-built install can't silently ship.
VERIFY_IMPORT="${VERIFY_IMPORT:-import reviewer.providers.ollama as o; assert hasattr(o, '_native_base_url')}"

PY="$VENV/bin/python"

# ---- args ------------------------------------------------------------------
DRY_RUN=0
RESTART=1
VERIFY_REVIEW=0
while [ $# -gt 0 ]; do
  case "$1" in
    --ref) REF="${2:?--ref needs a value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-restart) RESTART=0; shift ;;
    --verify-review) VERIFY_REVIEW=1; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

log() { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[deploy:error]\033[0m %s\n' "$*" >&2; exit 1; }

# Pull over HTTPS rather than the clone's (SSH) origin. The deploy user owns the
# venv but has no GitHub SSH key; these repos are public, so an HTTPS read needs
# no credentials and no per-user key setup. Override via $FETCH_URL for a private
# repo (e.g. a token URL). Derives the HTTPS form from the origin remote.
origin_to_https() {
  local url
  url="$1"
  case "$url" in
    git@github.com:*)        echo "https://github.com/${url#git@github.com:}" ;;
    ssh://git@github.com/*)  echo "https://github.com/${url#ssh://git@github.com/}" ;;
    *)                       echo "$url" ;;
  esac
}

# Run a command as the venv-owning user (so new files stay owned by it). Exec
# directly if we already are that user, else via passwordless sudo.
run_as_owner() {
  if [ "$(id -un)" = "$DEPLOY_USER" ]; then
    "$@"
  else
    sudo -n -u "$DEPLOY_USER" "$@"
  fi
}

# Echo a mutating command, then run it unless --dry-run.
do_cmd() {
  log "+ $*"
  [ "$DRY_RUN" = "1" ] && return 0
  "$@"
}

# ---- preflight -------------------------------------------------------------
[ -d "$SRC/.git" ] || die "source clone is not a git repo: $SRC"
[ -x "$PY" ] || die "venv python not found/executable: $PY"
if [ "$(id -un)" != "$DEPLOY_USER" ]; then
  sudo -n true 2>/dev/null || die "need passwordless sudo to act as '$DEPLOY_USER' (or run this script as $DEPLOY_USER)"
fi
# Reachability probe: no -f, so a 404 on the root still counts as "connected".
curl -sS -o /dev/null --max-time 5 "$BUS/" 2>/dev/null || die "bus not reachable at $BUS"

log "agent=$AGENT_ID ref=$REF src=$SRC user=$DEPLOY_USER dry_run=$DRY_RUN"

# ---- pull (HTTPS; khonliang is a keyless service account) ------------------
FETCH_URL="${FETCH_URL:-$(origin_to_https "$(run_as_owner git -C "$SRC" remote get-url origin)")}"
log "fetch url: $FETCH_URL"
OLD_SHA="$(run_as_owner git -C "$SRC" rev-parse HEAD)"
do_cmd run_as_owner git -C "$SRC" checkout --quiet "$REF"
do_cmd run_as_owner git -C "$SRC" pull --ff-only --quiet "$FETCH_URL" "$REF"
NEW_SHA="$(run_as_owner git -C "$SRC" rev-parse HEAD)"
if [ "$OLD_SHA" = "$NEW_SHA" ]; then
  log "source already at $NEW_SHA (forcing reinstall + restart anyway)"
else
  log "source $OLD_SHA -> $NEW_SHA"
fi

# ---- reinstall package code (always; cheap, idempotent) --------------------
do_cmd run_as_owner "$PY" -m pip install --force-reinstall --no-deps --no-cache-dir "$SRC"

# ---- sync dependencies only if pyproject.toml changed ----------------------
if [ "$OLD_SHA" != "$NEW_SHA" ] \
   && run_as_owner git -C "$SRC" diff --name-only "$OLD_SHA" "$NEW_SHA" | grep -qx "pyproject.toml"; then
  log "pyproject.toml changed -> syncing dependencies"
  # No --force-reinstall: installs new/changed deps, leaves satisfied ones
  # (so the git-sourced lib is not needlessly re-cloned).
  do_cmd run_as_owner "$PY" -m pip install --no-cache-dir "$SRC"
else
  log "no pyproject.toml change -> skipping dependency sync"
fi

# ---- post-install smoke (neutral cwd, so it can't import from a stray dir) --
if [ "$DRY_RUN" != "1" ]; then
  ( cd /tmp && run_as_owner "$PY" -c "$VERIFY_IMPORT" ) \
    && log "install smoke OK" \
    || die "install smoke failed -- new code not importable: $VERIFY_IMPORT"
fi

# ---- restart (bus-managed) -------------------------------------------------
if [ "$RESTART" = "1" ]; then
  OLD_PID="$(pgrep -f "reviewer.agent --id $AGENT_ID" | head -1 || true)"
  do_cmd curl -fsS -X POST "$BUS/v1/install/$AGENT_ID/restart"
  [ "$DRY_RUN" = "1" ] && { log "dry-run: restart skipped"; exit 0; }
  echo
  NEW_PID=""
  for _ in $(seq 1 30); do
    NEW_PID="$(pgrep -f "reviewer.agent --id $AGENT_ID" | head -1 || true)"
    [ -n "$NEW_PID" ] && [ "$NEW_PID" != "${OLD_PID:-}" ] && break
    sleep 1
  done
  if [ -n "$NEW_PID" ] && [ "$NEW_PID" != "${OLD_PID:-}" ]; then
    log "restarted: pid ${OLD_PID:-none} -> $NEW_PID"
  else
    die "could not confirm a new $AGENT_ID process after restart (old pid=${OLD_PID:-none})"
  fi
else
  log "--no-restart: running process left untouched (will pick up code on next restart)"
fi

# ---- optional end-to-end verification --------------------------------------
if [ "$VERIFY_REVIEW" = "1" ] && [ "$DRY_RUN" != "1" ]; then
  log "verify-review: routing a large diff through the bus (expect input_tokens > 4096) ..."
  # Retries with backoff: a just-restarted agent needs a few seconds to
  # re-register and warm the model, so the first attempt can return an empty
  # (input_tokens=0) result. Poll until it serves a real review.
  "$PY" - "$BUS" <<'PYEOF'
import sys, time, httpx
bus = sys.argv[1]
big = "diff --git a/f b/f\n" + "".join(
    f"+line {i}: reviewable content for the model here\n" for i in range(1100)
)
payload = {
    "operation": "review_diff", "agent_type": "reviewer",
    "args": {"kind": "pr_diff", "diff": big, "model": "deepseek-coder-v2:16b"},
    "timeout": 300.0, "response_mode": "raw",
}
deadline = time.monotonic() + 180
attempt = 0
while time.monotonic() < deadline:
    attempt += 1
    try:
        r = httpx.post(f"{bus}/v1/request", json=payload,
                       timeout=httpx.Timeout(connect=10, write=30, pool=30, read=320))
        usage = r.json().get("result", {}).get("usage", {}) or {}
        it = int(usage.get("input_tokens", 0) or 0)
        if it > 0:
            print(f"[deploy] verify-review: input_tokens={it} "
                  f"est~{len(big.encode())//3} model={usage.get('model')}")
            sys.exit(0 if it > 4096 else 1)
        print(f"[deploy] verify-review attempt {attempt}: agent not ready yet "
              "(empty result); retrying ...")
    except Exception as exc:  # best-effort readiness poll
        print(f"[deploy] verify-review attempt {attempt}: {exc}; retrying ...")
    time.sleep(6)
print("[deploy] verify-review: timed out waiting for a real review", file=sys.stderr)
sys.exit(1)
PYEOF
  log "verify-review PASSED (no longer truncated at 4096)"
fi

log "deploy complete: $AGENT_ID at $NEW_SHA"
