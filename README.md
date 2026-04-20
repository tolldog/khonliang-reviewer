# khonliang-reviewer

Bus-native reviewer agent for the khonliang ecosystem.

The reviewer role is broader than code review: it evaluates PR diffs today and
extends to specs, FRs, docs, and other content via the library's kind-
extensible contract. Concrete review providers (Ollama, Claude-via-CLI,
future GPT/Gemini) live in this repo; shared primitives live in
`khonliang-reviewer-lib`.

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

## Running

Start the reviewer agent against the bus:

```sh
.venv/bin/python -m reviewer.agent --id reviewer-primary --bus http://localhost:8787 --config /absolute/path/to/config.yaml
```

At the current scaffold stage no skills are registered yet. They land in
subsequent work units.

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
