# MS-B: Reviewer distill pipeline + repo-aware prompts

**Milestone:** `ms_reviewer_e3f751cd`
**Target:** `reviewer`
**Status:** proposed
**FRs:**
- `fr_reviewer_de1694a8` — Uniform distill-before-handoff pipeline driven by rule-table DistillConfig (anchor)
- `fr_reviewer_cb081fa8` — Best-of-N local consensus + three-tier evaluator gate
- `fr_reviewer_afd4bab1` — Populate `.reviewer/examples/` with repo-specific invariant patterns

## Problem

Today every reviewer-output quality concern wires itself into `review_text` directly. `severity_floor` (merged via PR#13) is its own post-filter step bolted onto the return path; future transforms — consensus merging, dedup, body-mode shaping, max-findings caps — would each add another bolted-on step. The result is fragile:

1. **No composition order.** Two transforms that should run in a specific order (consensus before dedup, dedup before severity filter, severity before max-findings) have nowhere to live except inline in the agent.
2. **No audience awareness.** A finding emitted to `github_comment` should be terser than the same finding emitted to an `audit_corpus`. The rule table picks `(backend, model)` per `(kind, profile, size)`; it does not pick *output shape*. So callers either get one shape for everything or each caller hand-shapes the result.
3. **No prompt-side calibration data.** The `.reviewer/prompts/` loader (merged via PR#14) reads `severity_rubric.md`, `examples/<kind>/<severity>.md`, and `system_preamble.md` if they exist — but no consumer repo has any of those files. The loader has nothing real to validate against, and the calibration story stays theoretical.
4. **Consensus is structurally homeless.** `fr_reviewer_cb081fa8` (three-tier evaluator gate: local Best-of-N → claude_cli escalation) has no place to plug in; it would need its own pipeline, which is the third bolted-on step in a row.

The connective tissue is missing. Until it lands, every quality-improvement FR re-litigates "where in the call path does this run?" instead of "what does this transform do?"

## Design Principle

**One pipeline. Many transforms. Rule-table-driven config.**

The rule table's output shape grows from `PolicyDecision` to `(ProviderDecision, DistillConfig)`. A new `reviewer.distill` module owns an ordered pipeline of pluggable transforms, each with the signature `(ReviewResult, DistillConfig) -> ReviewResult`. Every quality concern — severity_floor, consensus, dedup, body shaping, max-findings — becomes a transform that plugs into that pipeline at a fixed position. New concerns add a new transform; they do not edit `review_text`.

**Every transform passes the "10×-outlier survives unchanged" test.** Per `project_reviewer_distill_principle` memory: aggressive smoothing destroys signal. Distillation reduces *noise* (low-value findings, duplicates, out-of-audience verbosity) without flattening *features* (the one concern hiding among 20 nits). Each transform ships with both a noise-reduction test and a feature-preservation test (typically: "given an outlier concern alongside 20 nits, the concern survives the transform unchanged").

**Audience is a first-class rule-table input.** The rule table already takes `(kind, profile, size)`; this milestone adds `audience: github_comment | developer_handoff | human_review | agent_consumption | audit_corpus`. The default audience is `agent_consumption` (low aggression — most reviews are still bot-to-bot). `audit_corpus` is the special case that always emits raw findings with zero distillation.

**Reversibility.** Dropped findings are not destroyed; they live in `ReviewResult.dropped_findings` so audit corpora and benchmarks can still see what the model produced before the pipeline shaped it. This is what makes aggressive `github_comment`-mode tuning safe: the raw output is still recoverable.

**Examples corpus seeds the prompt loader.** `fr_reviewer_afd4bab1` ships the first reference set of `.reviewer/examples/` files — drawn from milestone_store invariants since that's a real repo with real findings the reviewer agent has been generating. With concrete files in place, the prompts-loader's assertions become measurable: "given this severity rubric and these 3 examples, does qwen2.5-coder:14b's calibration improve on this held-out diff?"

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
- **`reviewer.distill` module** — pipeline order, fixed:
  1. **consensus** — when `DistillConfig.consensus`, run N=3 Best-of-N against the configured local-tier model; converge findings via overlap; on disagreement above a threshold escalate to `claude_cli` per `fr_reviewer_cb081fa8`'s three-tier rubric.
  2. **dedup** — collapse exact-text or title-substring duplicates into a single finding, preserving the highest severity among the merged set.
  3. **severity_filter** — drop findings below `DistillConfig.severity_floor` (existing behavior, lifted into the pipeline).
  4. **body_mode** — shape `summary` + finding `body` length per `body_mode`. `compact` strips finding bodies; `brief` keeps the first sentence; `full` no-op.
  5. **max_findings** — sort findings by severity desc + first-position then truncate.
- **`audit_corpus` audience** is a hard short-circuit: the pipeline returns `ReviewResult` with `findings` unchanged and `dropped_findings=[]`, regardless of the configured transforms.
- **`ReviewResult.dropped_findings`** — new field on the result dataclass. Persists across the pipeline. Always populated (empty list when nothing dropped).
- **`.reviewer/examples/` seed corpus** — populated against `tolldog/khonliang-reviewer-store` (or another representative milestone_store consumer). Files:
  - `examples/pr_diff/nit.md` — 3 examples drawn from real reviewer output.
  - `examples/pr_diff/comment.md` — 3 examples.
  - `examples/pr_diff/concern.md` — 3 examples.
  - `severity_rubric.md` — the 2026-04-22 dogfood-derived rubric (project_evaluator_gate_exp memory).
  - `system_preamble.md` — `khonliang-reviewer-store` repo invariants the reviewer should know about (e.g. "store agents own artifact lifecycle; non-store callers must not write store DB").

### Out of scope

- **Per-caller runtime overrides** of `DistillConfig`. First cut: rule-table produces a single config; callers cannot override individual fields inline. If real demand emerges, add a merge-with-caller-dict pass in a follow-up FR.
- **Semantic dedup.** `dedup: semantic` is reserved in the dataclass but not implemented. Real implementation needs an embedding-similarity step that's a separate FR (likely depends on librarian-primary indexing).
- **Auto-generation of examples** from past addressed-findings. Out of scope here; consumed by `fr_reviewer_570aad54` in MS-C.
- **Runtime example selection** ("pick 3 most-similar past findings as few-shot"). The loader stays static; dynamic retrieval is a later FR.
- **Custom severity scales.** Floor remains an enum over the three existing severities; no numeric scoring.

## Acceptance Criteria

1. `review_text({..., audience: "github_comment"})` returns a terser, severity-floored result matching the rule-table entry for `(github_comment, *, *, *)`. Same call with `audience: "audit_corpus"` returns the raw findings with `dropped_findings == []` (regardless of other rule-table fields).
2. The three transforms (severity_floor, consensus, dedup) all plug into `reviewer.distill` — not into `review_text` directly. Each ships with a unit test for noise reduction *and* a feature-preservation test ("given an outlier concern alongside 20 nits, the concern survives unchanged").
3. Rule-table evaluation returns both `(ProviderDecision, DistillConfig)` in a single call. No caller assembles the two halves from separate queries.
4. `.reviewer/config.yaml` loader supports `rules[].distill` with at least the field set listed in §Scope.
5. `ReviewResult.dropped_findings` is populated (empty list when nothing dropped). Audit corpus runs (`audience: audit_corpus`) always have `dropped_findings == []` *and* the full original finding list.
6. The first-corpus `.reviewer/examples/` set ships in this milestone. Existing prompt-loader unit tests already merged in PR#14 are extended to load from that real corpus and assert the merged prompt contains all expected sections in the documented order.
7. Three-tier evaluator gate behavior (per `fr_reviewer_cb081fa8`):
   - When `DistillConfig.consensus == False`: pipeline runs the configured single provider once, no escalation.
   - When `DistillConfig.consensus == True` and disagreement is below threshold: result is the consensus output of N=3 local runs.
   - When `DistillConfig.consensus == True` and disagreement is above threshold: result is the `claude_cli` escalation output (with `usage.disposition: "escalated-approved"` so the trailer convention can record it).
8. Performance: distill pipeline overhead is below 50ms for a 10-finding result on local hardware (no LLM calls in the post-provider transforms; consensus's LLM cost is metered separately via the provider's own `UsageEvent`).
9. Tests cover: each transform in isolation, the full pipeline ordering, the audit_corpus short-circuit, the consensus-disagreement escalation path, and the rule-table → DistillConfig → pipeline composition.

## Open Questions

1. **Where does `audience` enter the rule-table inputs?** The current rule table keys on `(kind, profile, size)`. Adding `audience` as a fourth dimension makes the table 5× wider. Alternative: keep the existing key, attach a default `DistillConfig` per row, and let callers override `audience` via the request. Decision deferred to implementation; document the chosen path in this spec's first revision.
2. **Should `consensus` transforms run before the provider call (parallel sampling) or after (post-hoc voting on cached single result)?** `fr_reviewer_cb081fa8` describes parallel sampling. That implies the "transform" is actually a *replacement* for the regular provider call — not a post-hoc shaper. Need to decide whether `consensus` lives in the pipeline at all, or whether it lives in the *selector* layer (with the pipeline running once on the consensus result). Tentative: consensus is a selector-layer concern; the pipeline runs once on whichever result the selector produces. Validate with `cb081fa8` author intent before coding.
3. **`semantic` dedup placeholder** — should the dataclass field accept `semantic` and raise at runtime, or accept it and silently fall back to `title_substring`? Raising is more honest but causes hard breakage on misconfigured rules; falling back is friendlier but masks bugs. Tentative: raise with a clear "semantic dedup not implemented; use title_substring" error.

## Dependencies

- **Already merged:** `fr_reviewer_dfd27582` (severity_floor), `fr_reviewer_92453047` (`.reviewer/prompts/` loader), reviewer rule table (`fr_reviewer_2c02aaf8`).
- **Composes with:** MS-C (`fr_reviewer_570aad54` auto-promote excellent findings) — that work *consumes* the `.reviewer/examples/` directory this milestone *produces*. MS-C is sequenced after MS-B for that reason.
- **Blocks:** any future quality-improvement FR that needs to act on `ReviewResult` after the provider call. Until the pipeline exists, those FRs would have to add another bolted-on step.

## Implementation Notes (non-binding)

- The pipeline lives in `reviewer/distill/`:
  - `reviewer/distill/__init__.py` — module exports, `run_pipeline(result, config) -> ReviewResult`.
  - `reviewer/distill/transforms/severity_filter.py`
  - `reviewer/distill/transforms/dedup.py`
  - `reviewer/distill/transforms/body_mode.py`
  - `reviewer/distill/transforms/max_findings.py`
- Consensus likely lives outside the distill module per Open Question #2.
- `DistillConfig` ships in `reviewer/policy/distill.py` (sibling to existing `reviewer/policy/rules.py`).
- Audit-corpus short-circuit: `if config.audience == "audit_corpus": return result.with_dropped(())` — single early return at the top of `run_pipeline`.

## Revision history

- **rev 1** (2026-04-26): initial spec, author: Claude. Open questions flagged for first review pass.
