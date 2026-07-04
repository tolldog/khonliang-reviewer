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

The reviewer ships with five first-class backends (`reviewer/providers/`):
Ollama, TabbyAPI (the resident hot-tier engine, `reviewer.defaults.DEFAULT_REVIEWER_BACKEND`),
Claude-via-CLI, Codex-via-CLI, and GitHub-Copilot-via-CLI. See `README.md`
for provisioning/setup per backend. Two notes worth keeping here rather
than in the README:

- **Ollama** talks to the **native** `/api/chat` endpoint
  (`http://localhost:11434/api/chat`), not the OpenAI-compat `/v1` shim —
  the shim *silently drops* `options.num_ctx`, truncating every large
  review at the 4096-token default (`bug_reviewer_832a909b`); the native
  endpoint honors it. Configs still set `base_url` to the `/v1` base for
  back-compat; the provider strips the `/v1` suffix itself.
- **Claude-via-CLI** is a deliberate exception to the usual
  SDK-over-subprocess preference: the Anthropic SDK does not accept
  subscription OAuth tokens, and using them from third-party SDKs
  violates the 2026 Consumer TOS. `claude -p` is the only sanctioned path
  for subscription-backed Claude usage (`CLAUDE_CODE_OAUTH_TOKEN`,
  provisioned per-machine via `claude setup-token`).

**Forward note:** per-call `(backend, model)` selection is expected to be
superseded by a dispatcher-owned skill-request model (the caller asks for
a capability, dispatcher picks the engine) once that work lands — see the
`dispatcher-will-own-tabbyapi` project memory. Don't invest further in
backend-selection plumbing/docs beyond factual accuracy until that
direction is confirmed.

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
