# FR-review rubric (lightest touch)

**Purpose**: review a freshly-promoted FR (functional requirement)
for obvious problems — duplicates, target/classification
misclassification, depends_on chains pointing at closed FRs.
FRs are seed-stage exploration artifacts; the rubric is
deliberately gentler than the spec rubric so it doesn't
discourage idea capture.

The point of FR review is **not** to enforce spec-level rigor
on a not-yet-designed idea. Half the value of a freshly-promoted
FR is that the author dumped a half-formed thought before it
evaporated. Reviewing it like a spec would defeat the purpose.

## Finding categories

### `duplicate_of_existing_fr`
The proposed FR overlaps materially with an open FR already in
the store. Detect via:
- Concept-string match against existing open FRs in the same
  target.
- Title token overlap > ~50% with an existing FR.
- Description sub-string match on the load-bearing claim.

When fired, the finding should cite the existing FR id so the
author can choose to merge, redirect, or supersede.

### `target_misclassified`
The FR's `target` field doesn't match the body of the description.
Example: `target=reviewer` but the description proposes a change
to developer's spec/milestone skills. Author probably picked the
wrong target on the promote_fr call.

### `classification_misclassified`
The FR's `classification` field is `library` but the description
proposes app-level state, OR `classification` is `app` but the
description proposes a generic primitive that should live in a
shared library (`khonliang-reviewer-lib`, `bus-lib`, etc.).

### `depends_on_closed_fr`
The FR's `depends_on: [...]` points at one or more FRs that are
already `merged` / `superseded` / `abandoned`. Either the
dependency is satisfied (drop it from `depends_on`), or the
FR was authored against a stale view (re-state the dependency
against current state).

### `acceptance_too_vague_to_close`
**Allowed at FR seed phase, but flag as `comment`** — not
`concern`. The point is to surface that the FR will need a
spec phase to close objectively, not to gate the FR's
promotion. Common at this phase since acceptance often gets
written during spec authoring.

### `scope_too_broad_for_one_fr`
The proposed FR describes work that should split into 2+ FRs.
Recommendation should name the natural split axis (code
locality, dependency ordering, theme).

## Self-review checklist (run before promote_fr)

1. **Duplicate sweep**: `developer-primary.list_frs(target=<target>,
   status='open')` and skim concepts/titles for overlap.
2. **Target sanity**: re-read the description's first paragraph;
   does the work actually live in the named target? If the
   description says "developer should expose a new skill", the
   target is `developer`, not `reviewer`.
3. **Classification sanity**: does the work land in
   `<target>-lib` (library) or `<target>` (app)? Library work
   is generic primitives consumable by sibling apps; app work
   is target-specific.
4. **depends_on validation**: each `fr_<id>` in `depends_on`
   resolves to an open FR.

## Severity guidance

- `concern`: `duplicate_of_existing_fr` (with a specific
  existing-FR citation), `target_misclassified`,
  `classification_misclassified`, `depends_on_closed_fr`.
  These block FR promotion or require redirect/merge.
- `comment`: `acceptance_too_vague_to_close` (allowed at FR
  phase but worth flagging for the spec author),
  `scope_too_broad_for_one_fr` (split recommendation).
- `nit`: title phrasing, concept string clarity. FR text is
  exploratory; cosmetic drift is fine until spec time.

## Out of scope for FR review

- Full design rigor — that's the spec phase's job.
- Implementation note completeness — the FR doesn't have to
  enumerate every file path it'll touch.
- Internal-consistency sweep across §Scope / §Acceptance /
  §Open Questions — FRs typically don't have those sections.
- Schema design — same, runs in `schema_design_checklist.md`
  during the spec phase.

The point is: a fresh FR with rough acceptance + half-articulated
scope is GOOD. Don't reject ideas for being half-formed.
