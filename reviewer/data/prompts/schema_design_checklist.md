# Schema-design checklist (cross-cutting)

**Purpose**: cross-cuts the spec / milestone / FR rubrics. Whenever
a planning artifact proposes persisted state — a SQL table, a JSON
schema, a YAML config shape, an on-disk file layout — this
checklist asks: can the proposed schema serve every query the
artifact requires?

Distilled from PR `tolldog/khonliang-reviewer#24` R13: the
`address_rate` schema couldn't satisfy `address_rate_summary` or
the 90th-percentile cohort ranking because the cohort dimensions
(`repo`, `pr_number`, `model`, `kind`, `severity`) weren't in the
table. The artifact-spec was internally consistent (no cited
column was wrong) but designed-incomplete — neither the
self-review categories nor Copilot's first 12 rounds caught it
because it's not an *identifier* problem; it's a *missing piece*
problem. Flagging it requires reading the artifact's schema +
its query surface together and asking: does each query have the
columns it needs?

## Checklist (apply on every artifact that proposes persisted state)

1. **Enumerate query surfaces**: list every query the artifact
   describes — rollup skills, ranking computations, lookups by
   key, filters by dimension, joins to sibling tables, audit
   reads. For PR #24 R13 the query surfaces were
   `address_rate_summary(model, repo, kind, severity)`,
   `90th-percentile cohort scan`, `list_pending_examples`
   (reads sibling files, not this table — out of scope for
   this checklist).

2. **Enumerate persisted columns**: list every column the
   schema defines (NULL semantics, types, CHECK constraints).

3. **Cross-check**: for each query, can it run as pure SQL
   against the persisted columns? If a query needs a column
   that isn't persisted (e.g. "filter by model" against a
   schema with no `model` column), the schema is
   designed-incomplete. Either:
   - Add the column. (The fix in PR #24 R13.)
   - Fetch the column from a sibling table at query time
     (with the join cost made explicit).
   - Re-scope the query so it doesn't need the column
     (rare; usually means the artifact's acceptance criteria
     need updating too).

4. **Cohort-dimension audit**: when an artifact uses the words
   "cohort", "rank within", "filter by", "group by",
   "summary across" — each named dimension must be persisted.
   This is the specific case PR #24 R13 caught.

5. **Index-coverage check**: any rolling-window query
   (`since_days=30`, `last N records`) needs a covering index
   that includes the time column. Any cohort scan needs a
   covering index on the cohort columns + the filter column.
   Spec the index in §Implementation Notes; SQL without
   indexes is correct-but-slow.

6. **Privacy-invariant audit**: when the artifact claims
   "no third-party content stored" / "metadata only" / "no
   raw bodies persisted", enforce it structurally — closed
   enums via `CHECK`, fixed-length identifiers, the deliberate
   *absence* of any free-form `TEXT` column. Acceptance criteria
   that read "X is enforced structurally" must be backed by an
   actual structural mechanism, not by convention.

7. **Idempotency-key check**: any append-only or re-runnable
   computation needs a primary key whose components are stable
   across re-runs. Avoid nullable PK columns even when the
   target DB allows them — use sentinel values (`'no_merge_sha'`)
   and document the choice.

## When to apply

- During spec authoring: every spec that proposes a `CREATE TABLE`,
  a JSON schema, a YAML shape, a frontmatter shape, or any other
  persisted artifact.
- During the artifact-review pass: every revision that touches
  the §Implementation Notes schema OR the §Scope-level query
  surfaces.
- Whenever an Acceptance criterion says "queries / aggregates /
  ranks / filters" — re-run the cross-check with the current
  schema in hand.

## Severity guidance

- `concern`: query surface that can't be satisfied by the
  persisted schema (PR #24 R13 case); privacy-invariant
  claimed but not enforced structurally; nullable PK component.
- `comment`: missing covering index for a rolling-window query;
  idempotency key not explicitly named.
- `nit`: column ordering, comment wording on schema fields.

## Origin

PR #24 R13 (Copilot, 2026-04-26) — `address_rate` schema lacked
cohort columns required by `address_rate_summary` and 90th-percentile
ranking. The self-review pass didn't catch it because the rule's
categories (identifier verification, cross-section sweep,
aspirational-claim audit, patch-introduced fabrication) all assume
the artifact is self-consistent on existing identifiers; this was
a *design-completeness* gap, not a self-consistency gap. This
checklist is the rubric category that closes that gap.
