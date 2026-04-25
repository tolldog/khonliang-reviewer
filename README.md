# khonliang-reviewer

Bus-native reviewer agent for the khonliang ecosystem.

The reviewer role is broader than code review: it evaluates PR diffs today and
extends to specs, FRs, docs, and other content via the library's kind-
extensible contract. Concrete review providers (Ollama, Claude-via-CLI,
Codex-via-CLI, GitHub-Copilot-via-CLI, future GPT/Gemini direct) live in
this repo; shared primitives live in `khonliang-reviewer-lib`.

## Current Role

Reviewer is the evaluation layer:

- accept a `ReviewRequest` from any caller (developer, bus, direct CLI)
- select a provider + model from the rule table, or honor the caller's choice
- run the review, parse structured findings, emit usage telemetry
- (for PR reviews) post results via the GitHub API
- track token usage and cost-estimates so subscription value is measurable

Reviewer does not own FR lifecycle, milestones, specs, git/GitHub workflow,
or repo hygiene. Those live in `khonliang-developer`.

## Runtime Shape

The active path is:

1. `khonliang-bus` runs the shared bus and MCP adapter.
2. Reviewer starts as an agent and registers skills with the bus.
3. Developer (or any caller) invokes `reviewer.review_pr` / `review_diff` /
   `usage_summary` through the bus.
4. Reviewer returns compact responses by default; large outputs are stored as
   bus artifacts and returned as refs.

## Setup

```sh
python -m venv .venv
.venv/bin/python -m pip install -e .
```

During local development of both repos in lockstep, install
`khonliang-reviewer-lib` as editable from the sibling checkout rather than
pulling from GitHub:

```sh
.venv/bin/python -m pip install -e ../khonliang-reviewer-lib
```

Copy and edit the config:

```sh
cp config.example.yaml config.yaml
```

`config.yaml` is local-only.

### Claude backend provisioning (optional)

The Claude-via-CLI provider needs a subscription OAuth token. This is not an
API key — it draws from your Claude Pro/Max quota. Generate once per
machine:

```sh
claude setup-token
# copy the printed token and add it to the agent's environment:
export CLAUDE_CODE_OAUTH_TOKEN=...
```

Never commit the token. Treat it like `GITHUB_WEBHOOK_SECRET`.

### Codex backend provisioning (optional)

The Codex-via-CLI provider drives the OpenAI ChatGPT subscription quota
through the `codex` binary's stored OAuth token (`~/.codex/auth.json`).
Provision once per machine:

```sh
codex login        # interactive ChatGPT sign-in
codex login status # confirm "Logged in using ChatGPT"
```

For the API-key fallback (pay-per-token, not subscription), set
`OPENAI_API_KEY` instead — `codex` reads it automatically when
`auth.json` is absent.

The `codex exec review` subcommand is *not* used by this provider: it is
repo-bound (sources the diff from `--uncommitted` / `--base` / `--commit`,
no stdin path) and lacks `--output-schema`, so it cannot honor the
`ReviewFinding` contract. The provider uses parent `codex exec` with
`--output-schema` and a stdin-piped prompt, which works on arbitrary diff
bytes and produces schema-validated JSON.

### GitHub Copilot backend provisioning (optional)

The GitHub-Copilot-via-CLI provider rides the operator's GitHub Copilot
Pro/Pro+/Business subscription via the `copilot` CLI binary (separate
from `gh`). Provision once per machine:

```sh
copilot login          # OAuth device flow; stores credentials
                       # in ~/.copilot/ or the OS keyring
```

For headless / CI use, set one of `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`,
or `GITHUB_TOKEN` to a fine-grained PAT with the "Copilot Requests"
permission, an OAuth token from the GitHub Copilot CLI app, or an
OAuth token from the GitHub CLI (`gh`). Classic `ghp_` PATs are NOT
supported by Copilot.

Unlike claude_cli (`--json-schema`) and codex_cli (`--output-schema`),
the GitHub Copilot CLI has no schema-enforcement flag. The provider
embeds the response schema in the prompt body (same approach as the
Ollama path) and parses the JSONL event stream — taking the
`assistant.message` event with `phase: "final_answer"` and decoding
its `content` field as JSON. Tool surface is locked down via
`--available-tools=` (empty list) so the model can reason but not
edit files or invoke shell commands.

`copilot exec` and other interactive subcommands are NOT used. The
provider only uses `copilot -p <prompt> --output-format json
--allow-all-tools --available-tools= --no-color` — the headless gate
plus an empty tool surface gives review-only behavior with no risk of
the model attempting an action.

> **Privacy / argv exposure.** The GitHub Copilot CLI has no stdin or
> `--prompt-file` variant for non-interactive mode; the entire prompt
> (including the diff body) is passed as a single argv element to
> `copilot -p`. While the subprocess is running, that argument is
> visible to other local users via `ps`, `/proc/<pid>/cmdline`, or
> equivalent process-listing surfaces. **Don't route sensitive diff
> content through gh_copilot on a multi-user host.** The stdin-piping
> backends (claude_cli, codex_cli) keep the diff out of argv and do
> not have this exposure; route confidential reviews there. Argv
> overflow on very large diffs is also a real failure mode (hits OS
> `ARG_MAX`, ~128KB on Linux); the provider catches the resulting
> `OSError` and returns a structured errored result with a hint, but
> the reviewer should fall back to a stdin-capable backend for those
> cases anyway.

## Running

Start the reviewer agent against the bus:

```sh
.venv/bin/python -m reviewer.agent --id reviewer-primary --bus http://localhost:8787 --config /absolute/path/to/config.yaml
```

The agent registers `review_text` and `review_diff` bus skills. Provider
selection resolves in this order:

1. Caller-supplied `backend` + `model` (on either skill) — always win.
2. Rule table (`reviewer.rules.decide`) when the caller supplies neither
   — picks `(backend, model)` from `(kind, diff_size)` + cached
   `profile` (when available). Falls back to config defaults if no
   rule matches.
3. Config-level `default_provider` / `default_model` — the ultimate
   fallback.

The `model` argument is honored on all four backends:

- **Ollama** uses `model` directly in the `chat.completions.create`
  call (any Ollama-served model id — `qwen2.5-coder:14b`, `kimi-k2.5:cloud`, etc.).
- **Claude-via-CLI** threads `model` through as `claude -p --model
  <spec>` (accepts aliases like `opus`/`sonnet` or full ids like
  `claude-opus-4-7`).
- **Codex-via-CLI** threads `model` through as `codex exec -m <spec>`.
  On the bus-skill path the selector usually supplies a model — either
  the caller's `model` argument or, when the caller picks `codex_cli`
  without a `model`, the provider's own
  `CodexCliProviderConfig.default_model` (configurable in
  `config.yaml` under `providers.codex_cli.default_model`). Only when
  both are empty does the provider omit `-m`, and in that case codex
  falls back to its **built-in default model** — not the user's
  `~/.codex/config.toml`, because the subprocess argv also passes
  `--ignore-user-config` for deterministic behavior across operators.
  If you want a deterministic per-provider default, set
  `providers.codex_cli.default_model` explicitly. Note also that the
  selector deliberately does **not** apply `config.default_model` to a
  non-default backend, because the global default is paired with the
  default backend (e.g. `qwen2.5-coder:14b` is an Ollama model, not a
  Codex one).
- **GitHub-Copilot-via-CLI** threads `model` through as `copilot -p
  -m <spec>` (`gpt-5.4`, `claude-sonnet-4.5`, etc.; see the GitHub
  Copilot model catalog). Same precedence as Codex: caller's `model`
  arg → `providers.gh_copilot.default_model` → omit `-m` and let
  `copilot -p` pick its own ambient default. Optional
  `providers.gh_copilot.reasoning_effort` (`low`/`medium`/`high`/
  `xhigh`) maps to `--effort` on the CLI for callers that want
  depth control without per-request plumbing.

`review_pr` (GitHub fetch + post) and `usage_summary` (SQLite-backed
aggregation) are also registered alongside `review_text` / `review_diff`
as part of the current bus-skill surface.

`list_models` returns the catalog of registered backends and their
declared models, including a cheap availability hint per backend
(binary on PATH, auth file present, env var set — no network probe,
no model invocation):

```python
result = await bus.call("reviewer-primary.list_models", {})
# {"providers": [
#   {"backend": "claude_cli", "default_model": "",
#    "models": ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"],
#    "available": true, "reason": ""},
#   {"backend": "codex_cli", "default_model": "",
#    "models": ["gpt-5"], "available": true, "reason": ""},
#   {"backend": "ollama", "default_model": "qwen2.5-coder:14b",
#    "models": ["qwen2.5-coder:14b", "glm-4.7-flash", ...],
#    "available": true, "reason": ""}
# ]}
```

Pass `{"backend": "<name>"}` to filter to a single backend (or get
an empty list when unknown). The declared-models list is sourced
from `default_pricing.yaml` rows for that backend, with the
provider's own default model surfaced first when set. For real
liveness checks, call the provider's `healthcheck()` method directly
— the `available` flag from `list_models` is intentionally cheap
and may report true for backends that are configured but currently
down (e.g. an Ollama server that has stopped).

## Repository Boundaries

- `khonliang-reviewer`: this repo — provider implementations, bus skills,
  GitHub posting, SQLite usage storage, rule-table policy.
- `khonliang-reviewer-lib`: shared review contracts and helpers (provider
  interface, review / usage / pricing dataclasses, cost math).
- `khonliang-developer`: FR lifecycle, specs, milestones, git/GitHub
  workflow. Developer consumes reviewer output; it does not own review.
- `khonliang-bus` / `khonliang-bus-lib`: service registry, skill contracts,
  transport, agent communication.

When logic is useful outside this application, put it in
`khonliang-reviewer-lib`. When logic manages implementation work (FRs,
specs, milestones, PRs), put it in `khonliang-developer`.

## Validation

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall reviewer
```
