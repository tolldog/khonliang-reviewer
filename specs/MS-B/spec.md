# MS-B: Reviewer distill pipeline + repo-aware prompts

**Milestone:** `ms_reviewer_e3f751cd`
**Target:** `reviewer`
**Status:** proposed
**FRs:**
- `fr_reviewer_de1694a8` — Uniform distill-before-handoff pipeline driven by rule-table DistillConfig (anchor)
- `fr_reviewer_cb081fa8` — Best-of-N local consensus + three-tier evaluator gate
- `fr_reviewer_afd4bab1` — Populate `.reviewer/prompts/examples/` with repo-specific invariant patterns

## Problem

Today every reviewer-output quality concern wires itself into `review_text` directly. `severity_floor` (merged via PR#13) is its own post-filter step bolted onto the return path; future transforms — consensus merging, dedup, body-mode shaping, max-findings caps — would each add another bolted-on step. The result is fragile:

1. **No composition order.** Two transforms that should run in a specific order (consensus before dedup, dedup before severity filter, severity before max-findings) have nowhere to live except inline in the agent.
2. **No audience awareness.** A finding emitted to `github_comment` should be terser than the same finding emitted to an `audit_corpus`. The rule table picks `(backend, model)` per `(kind, profile, size)`; it does not pick *output shape*. So callers either get one shape for everything or each caller hand-shapes the result.
3. **No prompt-side calibration data.** The `.reviewer/prompts/` loader (merged via PR#14) reads `severity_rubric.md`, `examples/<kind>/<severity>.md` (one file per `(kind, severity)` cell — full file content becomes the few-shot block for that cell), and `system_preamble.md` if they exist — but no consumer repo has any of those files. The loader has nothing real to validate against, and the calibration story stays theoretical.
4. **Consensus is structurally homeless.** `fr_reviewer_cb081fa8` (three-tier evaluator gate: local Best-of-N → claude_cli escalation) has no place to plug in; it would need its own pipeline, which is the third bolted-on step in a row.

The connective tissue is missing. Until it lands, every quality-improvement FR re-litigates "where in the call path does this run?" instead of "what does this transform do?"

## Design Principle

**One pipeline. Many transforms. Rule-table-driven config.**

The rule table's output shape grows from `PolicyDecision` to `(ProviderDecision, DistillConfig)`. A new `reviewer.distill` module owns an ordered pipeline of pluggable transforms, each with the signature `(ReviewResult, DistillConfig) -> ReviewResult`. Every quality concern — severity_floor, consensus, dedup, body shaping, max-findings — becomes a transform that plugs into that pipeline at a fixed position. New concerns add a new transform; they do not edit `review_text`.

**Every transform passes the "10×-outlier survives unchanged" test.** Per `project_reviewer_distill_principle` memory: aggressive smoothing destroys signal. Distillation reduces *noise* (low-value findings, duplicates, out-of-audience verbosity) without flattening *features* (the one concern hiding among 20 nits). Each transform ships with both a noise-reduction test and a feature-preservation test (typically: "given an outlier concern alongside 20 nits, the concern survives the transform unchanged").

**Audience is a first-class rule-table input.** The rule table already takes `(kind, profile, size)`; this milestone adds `audience: github_comment | developer_handoff | human_review | agent_consumption | audit_corpus`. The default audience is `agent_consumption` (low aggression — most reviews are still bot-to-bot). `audit_corpus` is the special case that always emits raw findings with zero distillation.

**Reversibility.** Dropped findings are not destroyed; they live in `ReviewResult.dropped_findings` so audit corpora and benchmarks can still see what the model produced before the pipeline shaped it. This is what makes aggressive `github_comment`-mode tuning safe: the raw output is still recoverable.

**Examples corpus seeds the prompt loader.** `fr_reviewer_afd4bab1` ships the first reference set of `.reviewer/prompts/examples/` files — drawn from milestone_store invariants since that's a real repo with real findings the reviewer agent has been generating. With concrete files in place, the prompts-loader's assertions become measurable: "given this severity rubric and these 3 examples, does qwen2.5-coder:14b's calibration improve on this held-out diff?"

## Scope

### In scope

- **Rule-table output shape** evolves from `PolicyDecision` to `(ProviderDecision, DistillConfig)`. Existing `rules[].provider` block stays as-is; rules gain a sibling `distill:` block. Rule-table evaluation returns both decisions in one call so callers never split the query.
- **`DistillConfig`** dataclass — initial fields:
  - `severity_floor: "nit" | "comment" | "concern"` — wires to existing severity_floor filter (no behavior change for callers that already pass it inline; adding the rule-table path).
  - `body_mode: "compact" | "brief" | "full"` — shaped after researcher's `detail=` vocabulary.
  - `consensus: bool` — gates `fr_reviewer_cb081fa8`'s evaluator-gate runtime.
  - `dedup: "none" | "exact" | "title_substring" | "semantic"` — `semantic` is reserved (not implemented this milestone).
  - `max_findings: int | None` — post-filter cap.
  - `audience: "github_comment" | "developer_handoff" | "human_review" | "agent_consumption" | "audit_corpus"` — informs `body_mode` defaults but is independently inspectable (so transforms that care about audience without caring about body_mode can fork on it).
- **`reviewer.distill` module** — post-provider pipeline. Fixed order:
  1. **dedup** — collapse exact-text or title-substring duplicates into a single finding, preserving the highest severity among the merged set.
  2. **severity_filter** — drop findings below `DistillConfig.severity_floor` (existing behavior, lifted into the pipeline).
  3. **body_mode** — shape `summary` + finding `body` length per `body_mode`. `compact` strips finding bodies; `brief` keeps the first sentence; `full` no-op.
  4. **max_findings** — sort findings by severity desc + first-position then truncate.
- **Consensus is a selector-layer concern, not a distill transform.** When `DistillConfig.consensus` is true, `fr_reviewer_cb081fa8`'s three-tier evaluator gate runs *before* the distill pipeline: parallel-sample N=3 local-tier runs, converge findings via overlap, escalate to `claude_cli` on threshold disagreement. The single result that survives consensus then flows into the distill pipeline like any provider output. (Open Question #2 asked whether consensus is pipeline or selector; this spec decides selector. The `DistillConfig.consensus` field stays — it's how the rule-table tells the selector to engage consensus mode for a given `(kind, profile, size, audience)`.)
- **`audit_corpus` audience** is a hard short-circuit: the pipeline returns `ReviewResult` with `findings` unchanged and `dropped_findings=[]`, regardless of the configured transforms.
- **`ReviewResult.dropped_findings`** — new field on the result dataclass. Persists across the pipeline. Always populated (empty list when nothing dropped).
- **`.reviewer/prompts/examples/` seed corpus** — populated against `tolldog/khonliang-reviewer-store` (or another representative milestone_store consumer). Layout matches the existing prompts loader (one file per `(kind, severity)` cell; the entire file contents become the few-shot block for that cell):
  - `.reviewer/prompts/examples/pr_diff/nit.md` — 3 examples drawn from real reviewer output, concatenated within one file (e.g. separated by `---` rules so a human reader can still tell them apart).
  - `.reviewer/prompts/examples/pr_diff/comment.md` — 3 examples.
  - `.reviewer/prompts/examples/pr_diff/concern.md` — 3 examples.
  - `.reviewer/prompts/severity_rubric.md` — the 2026-04-22 dogfood-derived rubric (project_evaluator_gate_exp memory).
  - `.reviewer/prompts/system_preamble.md` — `khonliang-reviewer-store` repo invariants the reviewer should know about (e.g. "store agents own artifact lifecycle; non-store callers must not write store DB").

### Out of scope

- **Per-caller runtime overrides of `DistillConfig` fields.** First cut: rule-table produces a single config; callers cannot override individual fields like `severity_floor` / `body_mode` / `consensus` inline. If real demand emerges, add a merge-with-caller-dict pass in a follow-up FR. (Note: `audience` is *not* a DistillConfig override — it is a request-shape input that participates in rule-table key resolution; see Acceptance #1.)
- **Semantic dedup.** `dedup: semantic` is reserved in the dataclass but not implemented. Real implementation needs an embedding-similarity step that's a separate FR (likely depends on librarian-primary indexing).
- **Auto-generation of examples** from past addressed-findings. Out of scope here; consumed by `fr_reviewer_570aad54` in MS-C.
- **Runtime example selection** ("pick 3 most-similar past findings as few-shot"). The loader stays static; dynamic retrieval is a later FR.
- **Custom severity scales.** Floor remains an enum over the three existing severities; no numeric scoring.

## Acceptance Criteria

1. **`audience` is an explicit request input that participates in rule-table key resolution — not a per-call override of an individual `DistillConfig` field.** `review_text({..., audience: "github_comment"})` returns a terser, severity-floored result matching the rule-table entry for the given `(kind, profile, size)` with `audience: "github_comment"`. Same call with `audience: "audit_corpus"` returns the raw findings with `dropped_findings == []` (regardless of other rule-table fields, since audit_corpus is the hard short-circuit).
2. The post-provider transforms (severity_floor, dedup, body_mode, max_findings) all plug into `reviewer.distill` — not into `review_text` directly. Each ships with a unit test for noise reduction *and* a feature-preservation test ("given an outlier concern alongside 20 nits, the concern survives unchanged"). Consensus runs in the selector layer ahead of the distill pipeline (per Open Question #2 resolution).
3. Rule-table evaluation returns both `(ProviderDecision, DistillConfig)` in a single call. No caller assembles the two halves from separate queries.
4. `.reviewer/config.yaml` loader supports `rules[].distill` with at least the field set listed in §Scope.
5. `ReviewResult.dropped_findings` is populated (empty list when nothing dropped). Audit corpus runs (`audience: audit_corpus`) always have `dropped_findings == []` *and* the full original finding list.
6. The first-corpus `.reviewer/prompts/examples/` set ships in this milestone. Existing prompt-loader unit tests already merged in PR#14 are extended to load from that real corpus and assert the merged prompt contains all expected sections in the documented order.
7. Three-tier evaluator gate behavior (per `fr_reviewer_cb081fa8`, runs in selector layer before distill):
   - When `DistillConfig.consensus == False`: selector picks one provider per the rule table, runs once. No escalation. Result flows into distill pipeline.
   - When `DistillConfig.consensus == True` and disagreement is below threshold: selector parallel-samples N=3 local-tier runs, converges findings via overlap. Consensus result flows into distill pipeline.
   - When `DistillConfig.consensus == True` and disagreement is above threshold: selector escalates to `claude_cli`; the escalation output (with `usage.disposition: "escalated-approved"` so the trailer convention can record it) flows into distill pipeline.
8. Performance: distill pipeline overhead is below 50ms for a 10-finding result on local hardware (no LLM calls in the post-provider transforms; consensus's LLM cost is metered separately via the provider's own `UsageEvent`).
9. Tests cover: each transform in isolation, the full pipeline ordering, the audit_corpus short-circuit, the consensus-disagreement escalation path, and the rule-table → DistillConfig → pipeline composition.

## Open Questions

1. ~~**Where does `audience` enter the rule-table inputs?**~~ **Resolved (rev 5): `audience` is a first-class rule-table input.** The rule table keys on `(kind, profile, size, audience)`, and the selected row provides the `DistillConfig` consumed by the distill pipeline. This keeps audience-specific output shaping declarative in policy rather than encoded as an ad-hoc request-time transform, and matches Acceptance #1's behavior (the same `(kind, profile, size)` with a different `audience` resolves to a different row → different `DistillConfig` → different output shape). Cost: the rule table is 5× wider with the five audience values, but most rows can use a default `DistillConfig` for "non-special" audiences and special-case the two outliers (`audit_corpus` short-circuit, `github_comment` aggressive shaping). Callers may still override provider/model explicitly where existing reviewer contracts allow, but `audience`-driven distill behavior is selected from the rule table.
2. ~~**Should `consensus` transforms run before the provider call (parallel sampling) or after (post-hoc voting on cached single result)?**~~ **Resolved (rev 3): consensus runs in the selector layer**, ahead of the distill pipeline. `fr_reviewer_cb081fa8` describes parallel sampling, which is a *replacement* for the regular provider call — not a post-hoc shaper. Distill stays a pure post-provider pipeline. The `DistillConfig.consensus` field stays as the rule-table → selector signal.
3. **`semantic` dedup placeholder** — should the dataclass field accept `semantic` and raise at runtime, or accept it and silently fall back to `title_substring`? Raising is more honest but causes hard breakage on misconfigured rules; falling back is friendlier but masks bugs. Tentative: raise with a clear "semantic dedup not implemented; use title_substring" error.

## Dependencies

- **Already merged:** `fr_reviewer_dfd27582` (severity_floor), `fr_reviewer_92453047` (`.reviewer/prompts/` loader), reviewer rule table (`fr_reviewer_2c02aaf8`).
- **Composes with:** MS-C (`fr_reviewer_570aad54` auto-promote excellent findings) — that work *consumes* the `.reviewer/prompts/examples/` directory this milestone *produces*. MS-C is sequenced after MS-B for that reason.
- **Blocks:** any future quality-improvement FR that needs to act on `ReviewResult` after the provider call. Until the pipeline exists, those FRs would have to add another bolted-on step.

## Implementation Notes (non-binding)

- The pipeline lives in `reviewer/distill/`:
  - `reviewer/distill/__init__.py` — module exports, `run_pipeline(result, config) -> ReviewResult`.
  - `reviewer/distill/transforms/dedup.py`
  - `reviewer/distill/transforms/severity_filter.py`
  - `reviewer/distill/transforms/body_mode.py`
  - `reviewer/distill/transforms/max_findings.py`
- Consensus runs in the selector (per Open Question #2 resolution) — likely as a new mode on `ProviderSelector` that returns a converged/escalated `ReviewResult` instead of calling a single provider once. The distill pipeline never sees the difference.
- `DistillConfig` ships in `reviewer/rules/distill.py` (alongside the existing rule-table policy code in `reviewer/rules/policy.py`).
- Audit-corpus short-circuit: `if config.audience == "audit_corpus": return result.with_dropped(())` — single early return at the top of `run_pipeline`.

## Revision history

- **rev 1** (2026-04-26): initial spec, author: Claude. Open questions flagged for first review pass.
- **rev 2** (2026-04-26): correct loader path from `.reviewer/examples/` to `.reviewer/prompts/examples/` (per Copilot R1 on PR#24, grounded in `reviewer/config/prompts.py:195`). Clarified the one-file-per-cell layout that the loader expects.
- **rev 3** (2026-04-26): resolve internal contradictions per Copilot R2 on PR#24. (a) Open Question #2 resolved: consensus is selector-layer, not a distill transform. Removed consensus from the §Scope pipeline list, updated Acceptance #2 + #7 to reflect the selector-layer placement, marked Open Question #2 as resolved with rationale, updated Implementation Notes accordingly. (b) Acceptance #1 reworded to not hard-code a `(github_comment, *, *, *)` shape (since rule-table key is `(kind, profile, size)` and audience routing is still an implementation choice).
- **rev 4** (2026-04-26): clarify `audience` semantics per Copilot R6 on PR#24. The Out-of-scope "no per-caller runtime overrides of DistillConfig" line conflicted with Acceptance #1's `review_text({..., audience: "github_comment"})` shape. Reconciled: `audience` is a request-shape input that participates in rule-table key resolution (a fourth keying dimension under Open Question #1's still-pending answer), NOT a per-call override of an individual DistillConfig field like `severity_floor`. The other fields stay rule-table-only.
- **rev 5** (2026-04-26): resolve Open Question #1 per Copilot R7 on PR#24. Earlier revs treated `audience` as a first-class rule-table input throughout (§Design Principle, §Scope, Acceptance #1) but Open Question #1 still said the audience→rule-table wiring decision was deferred — internally inconsistent. Decided in favor of "audience is a first-class rule-table input"; rule table keys on `(kind, profile, size, audience)`. Documented the cost (5× wider table) and the mitigation (most rows can share a default DistillConfig, special-cases for `audit_corpus` short-circuit and `github_comment` aggressive shaping).
- **rev 6** (2026-04-26): correct package path per Copilot R9 on PR#24. Earlier revs cited `reviewer/policy/distill.py` and `reviewer/policy/rules.py`, but rule-table code lives under `reviewer/rules/` (`reviewer/rules/policy.py` + `reviewer/rules/profile.py`); there is no top-level `reviewer/policy/` package. Updated Implementation Notes to land `DistillConfig` at `reviewer/rules/distill.py` so it's a sibling of the existing rule-table code rather than introducing a new top-level package.
