# MS-D: Provider polish + sign-off ergonomics

**Milestone:** `ms_reviewer_44949efb`
**Target:** `reviewer`
**Status:** proposed
**FRs:**
- `fr_reviewer_1c25c615` — `SelectorConfig` per-backend default models
- `fr_reviewer_2c751c3b` — `num_ctx` / context-window as reviewer provider knob
- `fr_reviewer_d8556085` — Ollama provider: support `format=json` for structured-output enforcement
- `fr_reviewer_b846a19c` — `sign_off_trailer()` skill — formatted git trailer from review result
- `fr_reviewer_8fb104e9` — `review_diff` / `review_text` arg consistency (`diff` vs `content`)

## Problem

Five small, mostly-independent reviewer-surface improvements that don't justify standalone milestones but do justify a single bundle so they ship together rather than sprawl as five tiny PRs over weeks. Each has a clear scope; together they round off the friction surface that has accumulated through dogfooding.

1. **`SelectorConfig` default-model is a single string.** PR#21 added per-backend defaults at the *Registry* layer (each provider registers with its own `default_model`), but `SelectorConfig.default_model: str = "qwen2.5-coder:14b"` is still a single global default. Result: a caller passing `backend="claude_cli"` with no `model` either gets the registry's per-provider default (correct) or — depending on which path the request takes — falls through to the SelectorConfig default and pipes a qwen-flavored model id into `claude -p --model qwen2.5-coder:14b`, which fails. The selector layer needs to either route to the registry's per-provider default or own a per-backend dict.
2. **`num_ctx` is partially exposed.** PR#20 shipped auto-bump + warning when the diff exceeds `num_ctx`. What did *not* ship: the explicit `num_ctx: int | None` kwarg on `review_diff` / `review_text` / `review_pr`, and the `.reviewer/models/<vendor>/<model>.yaml` config-file override. Without them, callers who want to *force* a large context (or pin a small one for benchmarking) have no surface for it; the auto-bump heuristic is the only knob, which is the wrong default for measurement use cases.
3. **Ollama `format=json` is not exposed.** The Ollama HTTP API supports `format: "json"` to constrain output to valid JSON. Smaller models (3B–7B) regularly fail JSON-schema enforcement when run as `ReviewProvider`s; turning on `format=json` at the provider level eliminates parse failure as a class of error. Today the option is unreachable through the reviewer's Ollama provider config.
4. **`Agent-Reviewed-by:` trailers drift.** Subagents have to hand-assemble the trailer line every commit (e.g. `Agent-Reviewed-by: khonliang-reviewer/ollama/qwen2.5-coder:14b approved-with-findings: 2 nits filtered`). The format drifts: punctuation, ordering, whether the verdict includes a reason. A `sign_off_trailer()` skill that takes a `ReviewResult` and returns a properly-formatted trailer line removes the drift surface entirely.
5. **`review_diff` takes `diff`; `review_text` takes `content`.** Subagents trip over this every other commit. Both skills review the same kind of payload — text bytes — and the only difference is the framing ("this is a diff" vs "this is freeform text"). The arg-name asymmetry exists for historical reasons, not design ones. Either alias them, or document the convention prominently enough that subagents stop tripping.

None of these are large; together they remove ~5 categories of friction.

## Design Principle

**Bundle as one milestone, ship as one PR if possible.** Each FR is a 1–2 file change. A single PR with all five reviews together (and 1 round of Copilot iteration) is cheaper than five separate PR cycles. The exception is `format=json` (#3) — that one needs end-to-end testing against a real small model and is the most likely to need its own iteration loop.

**Backwards compatibility, not backwards alias.** The `SelectorConfig.default_model` single string field stays valid: existing configs work unchanged. The new `default_models: dict[str, str]` field overrides it when present; the single-string field becomes the fallback. Same pattern for `review_diff(diff=...)` vs `review_text(content=...)` — keep both arg names, accept either, document the convention. We prefer "additive backwards-compatible" over "rename + alias" because the latter accumulates dead aliases over time.

**Trailer skill is the canonical formatter.** Once `sign_off_trailer()` ships, the user CLAUDE.md updates to instruct subagents to call it instead of hand-assembling. The hand-assembly path stays valid (we don't enforce skill use) but the documented happy path is the skill. Trailer format becomes a single source of truth.

**`format=json` is provider-level, not request-level.** Per-call `format=json` would let callers toggle structured output per request, but that's the wrong place — a model that needs `format=json` to reliably produce JSON needs it for *every* request, not selectively. The knob lives in `OllamaProviderConfig` (alongside existing `default_model`), and the rule-table can route to a config that has it set when the chosen model is in the "needs format-constraint" list.

## Scope

### In scope

**(1) Per-backend default models — `fr_reviewer_1c25c615`**
- Add `SelectorConfig.default_models: dict[str, str]` field.
- `ProviderSelector.select(backend, model)` resolution order:
  1. Caller-supplied non-empty `model`.
  2. `SelectorConfig.default_models.get(chosen_backend)`.
  3. Registry's `ProviderRegistration.default_model` for that backend.
  4. `SelectorConfig.default_model` (legacy single-string global default).
- Suggested defaults (operator-overridable in `config.yaml`):
  - `claude_cli: "sonnet"`
  - `codex_cli: ""` (codex picks its own; empty means "let the binary choose").
  - `gh_copilot: ""` (same).
  - `ollama: "qwen2.5-coder:14b"`.
- `_build_default_selector` in `agent.py` reads `config["default_models"]` (dict) when present; falls back to `config["default_model"]` (string) for backward-compat. Logs a deprecation note (info level) when only the legacy key is present.
- `config.example.yaml` updated to show both the legacy and the new shape.

**(2) `num_ctx` kwarg + per-model config — `fr_reviewer_2c751c3b`**
- Thread `num_ctx: int | None` through `review_text` / `review_diff` / `review_pr` agent skills → `ReviewRequest.metadata["num_ctx"]` → `OllamaProvider`.
- Resolution order in `OllamaProvider`:
  1. Caller-supplied `metadata["num_ctx"]` (if non-None).
  2. `.reviewer/models/ollama/<model>.yaml` `num_ctx:` field (when the `.reviewer/` loader from MS-A1/B is in scope).
  3. Auto-estimate (existing PR#20 behavior).
  4. Model's documented default.
- The auto-bump heuristic stays as default behavior; the new kwarg lets callers opt out for measurement runs.
- New ProviderConfig field `num_ctx: int | None = None` so the CLI / config path mirrors the runtime kwarg.

**(3) Ollama `format=json` — `fr_reviewer_d8556085`**
- New `OllamaProviderConfig.format: "" | "json" = ""`.
- When set to `"json"`, the Ollama HTTP request body includes `"format": "json"`. The OpenAI-compatible SDK (which we're using against `localhost:11434/v1`) supports `extra_body={"format": "json"}` — same path as the existing `extra_body={"options": {"num_ctx": ...}}`.
- Compose with the rule table: a rule row can specify `provider.format: "json"` so small evaluator models route through the format-constrained path automatically.
- Existing JSON-parse defensive coercion (`_coerce_str`, `_coerce_severity`) stays in place — `format=json` reduces but does not eliminate parse failures.

**(4) `sign_off_trailer()` skill — `fr_reviewer_b846a19c`**
- New MCP skill on the reviewer agent: `sign_off_trailer(result_or_review_args, *, role="khonliang-reviewer") -> {trailer_line: str, verdict: str}`.
- Two call shapes:
  - **Pass-through**: `sign_off_trailer({"backend": "ollama", "model": "qwen2.5-coder:14b", "diff_path": "/abs/path/diff.patch", ...})` — runs a review internally then formats the trailer.
  - **Result-only**: `sign_off_trailer({"result": {"backend": "...", "model": "...", "findings": [...], "summary": "..."}})` — formats from a result the caller already has.
- Verdict mapping:
  - 0 concern + 0 comment + 0 nit (or all filtered) → `approved`
  - 0 concern + ≥1 comment/nit → `approved-with-findings`
  - ≥1 concern → `concerns-raised` (caller decides whether to escalate; the trailer is honest about the count).
  - Provider returned `error_category="claude_cli_escalation"` → `escalated-approved`.
- Trailer format (locked):
  ```
  Agent-Reviewed-by: <role>/<backend>/<model> <verdict>[: <short reason ≤ 80 chars>]
  ```
- The `<short reason>` is required when verdict is `approved-with-findings` or `concerns-raised`; the skill builds it from the finding histogram (e.g. `"2 nits + 1 comment filtered"` or `"1 concern: false positive — separate control-flow branches"`).

**(5) `review_diff` / `review_text` arg consistency — `fr_reviewer_8fb104e9`**
- Accept both `content=` and `diff=` on both skills; either resolves to the same internal field.
- Document the convention in `reviewer/agent.py` skill docstrings *and* the developer guide: `review_diff` is "this is unified-diff bytes"; `review_text` is "this is freeform text". The arg-name difference exists only because of legacy.
- No deprecation warning yet (silent acceptance keeps the migration painless). Add the warning in a follow-up only if a future refactor needs the alias gone.

### Out of scope

- **Aggressive per-call rule-table override** of `format=json`. Provider-level only; the rule-table can route to a config that has it set, but there is no per-request toggle.
- **Trailer formats other than `Agent-Reviewed-by:`.** Reviewing-tool-side trailers (e.g. ESLint sign-off) are not covered.
- **Auto-detection of model context window.** `num_ctx` resolution still relies on operator-supplied config + the existing auto-bump heuristic; no `ollama show <model> --info`-style probe yet.
- **Renaming any existing arg.** Both `content=` and `diff=` are accepted; neither is deprecated yet.
- **OpenAI / Anthropic structured-output enforcement equivalents.** This milestone only adds the Ollama-side knob; the codex_cli / claude_cli paths already use `--output-schema` / `--json-schema` respectively.

## Acceptance Criteria

1. **`SelectorConfig.default_models`**: a config with `default_models: {claude_cli: sonnet, ollama: qwen2.5-coder:14b}` and *no* legacy `default_model` field loads cleanly. `select(backend="claude_cli", model=None)` returns `("claude_cli", "sonnet")`. Legacy single-string config still loads.
2. **`num_ctx` kwarg**: `review_diff(..., num_ctx=16384)` results in the Ollama HTTP request body carrying `options.num_ctx=16384`. `num_ctx=None` falls through to the auto-bump heuristic. `.reviewer/models/ollama/qwen2.5-coder_14b.yaml: num_ctx: 32768` overrides the heuristic (when the `.reviewer/` loader is reachable).
3. **`format=json`**: with `OllamaProviderConfig.format="json"`, the Ollama HTTP request body carries `format: "json"`. A 3B model (`llama3.2:3b`) that fails JSON parsing without the constraint succeeds with it. (Validated against the live model; smoke test in tests/.)
4. **`sign_off_trailer()`** returns the documented trailer format for each of the four verdict cases. Trailer parses cleanly via standard git trailer parser (`git interpret-trailers`). When called with a `result` containing 0 findings, returns `verdict: "approved"`. Result with 2 nits returns `verdict: "approved-with-findings", trailer_line: "Agent-Reviewed-by: khonliang-reviewer/<backend>/<model> approved-with-findings: 2 nits filtered"`.
5. **Arg consistency**: `review_diff(diff="...")`, `review_diff(content="...")`, `review_text(diff="...")`, `review_text(content="...")` all succeed and review the same payload. Tests cover all four shapes.
6. Tests: per-FR unit coverage as listed above; one integration test exercises `format=json` end-to-end against a containerized Ollama if the harness has one (otherwise mark skip-without-Ollama).
7. **Backward compat**: every existing test passes unchanged. No deprecation warnings emitted on legacy paths.
8. **Single PR shipping**: when 1–4 are independent file-level (no cross-FR conflicts), bundle into one PR. (5) is one-line + docs; bundles trivially. (3) `format=json` may peel off into its own PR if Ollama integration testing requires extra iteration.

## Open Questions

1. **Trailer reason composition.** `sign_off_trailer` builds `<short reason>` from the finding histogram. For `concerns-raised`, should the reason cite the *first concern's title* or summarize the count? Tentative: count + first concern's category (e.g. `"1 concern: race-condition"`). Easier to skim in `git log --oneline`.
2. **`format=json` vs schema enforcement.** The reviewer's prompt already includes the JSON schema in the system message. With `format=json` set, do we also keep the schema in the prompt (belt + suspenders), or rely on the Ollama-side constraint alone? Tentative: keep both. Some local models honor the prompt schema while ignoring the API-level constraint; redundancy is cheap.
3. **Legacy `default_model` deprecation.** Do we log a deprecation warning when only the legacy single-string is present? Pro: nudges operators to migrate. Con: nags every startup of every dev environment for the next year. Tentative: emit at INFO level once per process startup, not per request.
4. **`num_ctx` threading**: do we put it in `ReviewRequest.metadata` (alongside `model`) or in a separate `ReviewRequest.runtime: dict[str, Any]` namespace? Metadata is currently the only escape hatch and growing it makes the contract murky. Tentative: keep it in metadata for now; if we add 2+ more knobs of the same kind, promote to a `runtime` field as a follow-up.

## Dependencies

- **Soft-blocks on:** the `.reviewer/` loader (Milestone B) for `num_ctx` per-model config-file override. The kwarg + ProviderConfig path can ship without the loader; the `.reviewer/models/ollama/<model>.yaml` override path is gated on the loader landing first. Acceptance #2's last sentence is gated accordingly.
- **Composes with:** MS-B's distill pipeline. `format=json` reduces the output-shape variance the distill pipeline has to defend against; the two are complementary but neither blocks the other.
- **External:** Ollama HTTP server reachable for `format=json` integration test; same as today for any Ollama provider work.

## Implementation Notes (non-binding)

- File touch list (estimate):
  - `reviewer/selector.py` — `default_models` field, resolution order.
  - `reviewer/agent.py` — `_build_default_selector` reads new field; new `sign_off_trailer` skill registration.
  - `reviewer/providers/ollama.py` — `format` config field, `num_ctx` per-call override.
  - `reviewer/skills/sign_off_trailer.py` — new module.
  - `reviewer/agent.py::handle_review_text/diff/pr` — accept both `content=` and `diff=`.
  - `config.example.yaml` — show the new shape.
  - `tests/test_selector.py`, `tests/providers/test_ollama.py`, `tests/skills/test_sign_off_trailer.py`, `tests/test_agent_skills.py` — coverage.
- Suggested PR sequencing if not bundled:
  1. (1) + (5) + (4) — selector + arg consistency + trailer skill (smallest; 3 files).
  2. (2) — `num_ctx` kwarg + config (second; depends on .reviewer/ loader for the per-model override path).
  3. (3) — `format=json` (likely needs separate iteration against a live Ollama model).

## Revision history

- **rev 1** (2026-04-26): initial spec, author: Claude. Per-FR scope distilled from each FR's description; common bundle rationale documented in the design principle.
