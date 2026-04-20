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

1. **Ollama** via `openai` SDK against `http://localhost:11434/v1`.
2. **Claude-via-CLI** via subprocess around `claude -p --output-format=json`,
   consuming the Claude Pro/Max subscription quota via
   `CLAUDE_CODE_OAUTH_TOKEN` (provisioned per-machine via `claude setup-token`).

The Claude subprocess is a deliberate exception to the usual
SDK-over-subprocess preference: the Anthropic SDK does not accept
subscription OAuth tokens, and using them from third-party SDKs violates
the 2026 Consumer TOS. `claude -p` is the only sanctioned path for
subscription-backed Claude usage.

## Validation

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall reviewer
```

For provider changes, include focused coverage around transport mocking,
usage-record population, and error dispositions.
