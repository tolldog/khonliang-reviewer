# Built-in artifact-review rubrics

Markdown rubric prompts the reviewer agent loads when running an
**artifact review** (FR / spec / milestone — distinct from code review).
Mirrors the convention in `reviewer/data/default_pricing.yaml`:
shipped inside the package as bundled defaults. Repo-side overrides
for these artifact-review rubrics are planned as part of
`fr_reviewer_19c871ab` with the intended layout
`.reviewer/prompts/<rubric>.md`, but the override path is not
implemented yet — the existing `.reviewer/prompts/` loader at
`reviewer/config/prompts.py:195` reads only `system_preamble.md`,
`severity_rubric.md`, and `examples/<kind>/<severity>.md` today.

The artifact-review pipeline that consumes these prompts is the subject
of `fr_reviewer_19c871ab` (Artifact-review pipeline for FRs + specs +
milestones). Until that FR lands, the rubrics double as the **manual
self-review checklist** invoked before pushing planning-artifact
revisions, per `feedback_artifact_self_review_before_push` in the
user-level memory store. Working through them by hand is the workaround;
running them through the pipeline is the long-term shape.

## Files

- `spec_rubric.md` — full-document review for `specs/MS-*/spec.md`-shaped
  artifacts. Heaviest rubric: scope, acceptance, dependencies, design
  principle, internal consistency.
- `milestone_rubric.md` — review the FR cluster + summary + acceptance
  for a proposed milestone before any spec is authored.
- `fr_rubric.md` — review a freshly-promoted FR for duplicate detection,
  target / classification correctness, depends-on chain validity. Lightest
  touch: FRs are seed-stage exploration; the rubric stays narrow on
  purpose so it doesn't discourage idea capture.
- `schema_design_checklist.md` — cross-cuts the other three. Whenever
  an artifact proposes persisted state (SQL table, JSON schema,
  on-disk file shape), this checklist asks: can the proposed schema
  serve every query the artifact requires? Captures the lesson from
  PR #24 R13 — `address_rate` schema couldn't satisfy
  `address_rate_summary` because cohort columns were missing.
- `system_preamble.md` — *not present in this directory*; per-repo
  override only. A consuming repo can ship its own
  `.reviewer/prompts/system_preamble.md` with repo-specific
  invariants (e.g. "this repo's reviewer agent owns the rule
  table; non-reviewer callers must not write the rule-table store").
  The bundled default is intentionally empty — universal preamble
  content rarely earns its keep.

## How the rubrics get consumed

Each rubric is structured as:
1. **Purpose** — one sentence on what artifact shape it reviews.
2. **Categories** — the discrete finding categories the reviewer
   should look for, with examples drawn from real PR review rounds
   (see `project_dogfooding_log` Episode 12 for the source corpus).
3. **Self-review checklist** — the four-step manual pass humans run
   before pushing: identifier verification, cross-section sweep,
   aspirational-claim audit, patch-introduced fabrication check.
4. **Severity guidance** — when each finding category should be
   `nit` vs `comment` vs `concern` for that artifact kind.

The artifact-review pipeline (when `fr_reviewer_19c871ab` lands)
will pass each rubric as the system prompt for the corresponding
`kind=fr|spec|milestone` review. The schema-design checklist is
cross-cutting — every kind of artifact that proposes persisted
state runs through it as an additional pass on top of the kind-
specific rubric.

## Origin

Distilled from PR `tolldog/khonliang-reviewer#24` (the MS-B/C/D
spec PR), which ran 15 Copilot review rounds with ~30 cumulative
findings, all real. The categories below cover every finding
category that surfaced on that PR plus the schema-vs-query
consistency gap caught at R13. Future revisions should add new
categories whenever a Copilot finding falls outside the existing
set — that's the signal the rubric is missing a class of failure.
