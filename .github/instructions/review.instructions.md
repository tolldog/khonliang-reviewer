---
applyTo: "**"
---

# Review Instructions

Review this repository as the reviewer agent for the khonliang ecosystem.

Prioritize findings in this order:

1. Correctness, data loss risks, security issues, and behavioral regressions
   — especially around provider subprocess/HTTP plumbing and usage-record
   persistence.
2. Compatibility with `khonliang-reviewer-lib` contracts: `ReviewProvider`
   subclasses must honor the interface; serialized shapes must stay
   compatible with the library's dataclasses.
3. Context economy. Large diffs, profile summaries, and review outputs should
   flow through bus artifacts rather than inline context whenever possible.
4. Dependency discipline. Keep runtime deps thin; avoid pulling in a large
   library to cover one narrow behavior.
5. Test coverage for provider behavior (mocked transports), rule-table
   policy, usage-record serialization, and end-to-end review flows.
6. Documentation accuracy for the current reviewer/bus/developer workflow
   and the `CLAUDE_CODE_OAUTH_TOKEN` provisioning path.

Do not leave actionable correctness issues as vague future work. If a change
is needed for correctness or compatibility, call it out directly with the
affected file and line.

When reviewing provider changes, check that:

- Provider adapters do not leak transport details (aiohttp/subprocess) into
  the public surface; callers see `ReviewProvider.review()` only.
- Subscription-backed subprocess paths (Claude CLI) do not log or persist the
  bearer token.
- Usage records include cache-read / cache-creation token splits where the
  backend exposes them.

When reviewing rule-table or policy changes, check that:

- The `(kind, profile, size)` axis is honored — per-kind rules do not bleed
  across kinds.
- Fallback defaults remain correct and are exercised by tests.
- Callers can override provider/model explicitly and that path is tested.
