# khonliang-reviewer Agent Notes

Bus-native reviewer agent. This repository owns provider implementations,
bus skill wiring, GitHub review posting, SQLite usage storage, and
rule-table policy for picking `(backend, model)` from `(kind, profile, size)`.

When working here:

- Keep generic, content-agnostic review primitives in `khonliang-reviewer-lib`.
  This repo imports them; it does not duplicate them.
- Keep active FR lifecycle, milestones, specs, git/GitHub workflow, and repo
  hygiene in `khonliang-developer`. Developer consumes reviewer output; it
  does not own how reviews are performed.
- Keep ingestion, distillation, and evidence workflows in
  `khonliang-researcher`. Reviewer reads cached repo profiles from the
  researcher's knowledge store.
- Keep transport and skill registration contracts in `khonliang-bus-lib`.
- `config.yaml`, SQLite `reviewer.db`, logs, and machine-specific paths stay
  local (git-ignored).

## Backends

The reviewer ships with two first-class backends:

1. **Ollama** via `httpx` against the **native** `/api/chat` endpoint
   (`http://localhost:11434/api/chat`). Configs still set `base_url` to
   the OpenAI-compat `/v1` base for back-compat; the provider strips the
   `/v1` suffix to reach the native surface. The native endpoint is
   required because the `/v1` OpenAI-compat shim *silently drops*
   `options.num_ctx`, truncating every large review at the 4096-token
   default (`bug_reviewer_832a909b`); the native endpoint honors it.
2. **Claude-via-CLI** via subprocess around `claude -p --output-format=json`,
   consuming the Claude Pro/Max subscription quota via
   `CLAUDE_CODE_OAUTH_TOKEN` (provisioned per-machine via `claude setup-token`).

The Claude subprocess is a deliberate exception to the usual
SDK-over-subprocess preference: the Anthropic SDK does not accept
subscription OAuth tokens, and using them from third-party SDKs violates
the 2026 Consumer TOS. `claude -p` is the only sanctioned path for
subscription-backed Claude usage.

## Credentials

`reviewer/credentials.py` is the single entry point for every secret the
agent needs. **Nothing in the repo carries a real token** — discovery
delegates to external stores: env vars, `gh`'s keyring (via `gh auth
token`), eventually Claude's keyring and others. No caching, no file
writes inside the repo, no logging of token bodies. Tokens are
re-discovered on each call so rotation Just Works.

For generic unix portability the module delegates to each tool's own
CLI rather than cracking platform-specific keyrings directly — the
CLIs already know how to find their own credentials on Linux /
macOS / BSD. If a credential type grows a second consumer beyond
reviewer, promote the module to `khonliang-credentials-lib` (scoped in
a dedicated FR).

## Bus-Boundary Validation

`khonliang-reviewer-lib` contracts use `Literal` types for fields like
`severity` and `disposition`. Those are type-checker contracts — not
runtime validation. Any time data crosses the bus (skill args, cached
profile payloads, GitHub-sourced diffs) or an external API (HTTP responses,
subprocess JSON envelopes), validate shape and enum membership explicitly
before constructing the library dataclasses. Trust the library shape
within this repo; validate at every untrusted boundary.

## Validation

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall reviewer
```

For provider changes, include focused coverage around transport mocking,
usage-record population, and error dispositions.

## Deployment

The live agent runs from `/opt/khonliang/agents/reviewer/.venv` as a
**non-editable copied install** (NOT editable against this dev checkout)
and is supervised by the bus (no systemd). The source-of-truth clone is
`/opt/khonliang/src/khonliang-reviewer`. So a merge to `main` does **not**
go live until the source is pulled, the package reinstalled into the venv,
and the agent restarted.

`scripts/deploy.sh` does exactly that (parametrized for the sibling agents):
pull the source clone over **HTTPS** (the `khonliang` service account is
keyless; the repo is public), reinstall the package, sync dependencies only
when `pyproject.toml` changed, then restart via the bus. It runs every
mutating step as the venv-owning `khonliang` user (via `sudo -u`).

```sh
scripts/deploy.sh                 # pull main, reinstall, restart reviewer-primary
scripts/deploy.sh --dry-run       # print the commands without running them
scripts/deploy.sh --verify-review # after restart, prove a large diff is no longer
                                  # truncated at 4096 input tokens (bug_reviewer_832a909b)
```

Requires passwordless `sudo` (or run as `khonliang`) and PyPI reachable (the
venv has no setuptools, so `pip install` builds via isolation).
