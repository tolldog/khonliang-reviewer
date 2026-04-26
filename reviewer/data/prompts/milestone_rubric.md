# Milestone-review rubric

**Purpose**: review a freshly-proposed milestone — the FR cluster,
title, summary, acceptance criteria — *before* a spec is authored.
Catches cluster-cohesion problems and upstream-FR drift early so
the spec doesn't have to backfill them.

A milestone is the unit of "work that ships together" — typically
a small group of related FRs that share code locality, dependency
ordering, or theme. The rubric is lighter than the spec rubric
because milestones are scoped artifacts, not full-design documents.

## Finding categories

### `fr_cluster_incoherent`
The listed FRs don't actually belong together — they share neither
code locality, nor dependency ordering, nor a unifying theme. A
milestone authored by `propose_milestone_from_work_unit` from
auto-clustering can land here if the auto-clusterer was relying on
weak signals (concept-string matching, all-medium-priority
fallback).

**Heuristic**: read the FR titles. If you can name the
milestone's *theme* in one short noun phrase ("provider polish",
"closed-loop calibration", "distill pipeline + repo prompts"),
the cluster is coherent. If the best you can do is "miscellaneous
reviewer FRs", the cluster is too broad.

### `fr_cluster_too_broad`
More than ~5-6 FRs OR an FR set spanning multiple subsystems.
Splits should be proposed: split by code locality first, by
dependency ordering second, by theme third.

### `fr_cluster_too_narrow`
A single FR that's not large enough to warrant a milestone of
its own (a 1-line bug fix doesn't need a spec). Rebundle into
the nearest theme milestone or hand-off as a quick PR.

### `missing_fr_dependency`
An FR in the cluster has `depends_on` pointing at an FR not in
the cluster AND not yet merged. Milestone sequencing breaks: the
spec can't satisfy its own preconditions.

### `closed_fr_in_active_cluster`
The cluster lists an FR whose status is already `merged` /
`superseded` / `abandoned`. Common with auto-clustering when the
FR store has stale `open` rows. Trigger an FR-store status sweep
before re-proposing the milestone.

### `summary_misaligned_with_frs`
The milestone's summary text describes work that isn't actually
in the FR cluster, OR omits work the FRs require. Often a tell
that the cluster was rebundled but the summary wasn't updated.

### `acceptance_too_vague_for_milestone`
The milestone's acceptance criteria are the deterministic stub
("Milestone scope is explicit and bounded to the listed FRs")
without any milestone-specific acceptance shape. That's fine
for a freshly-proposed milestone but should be flagged so the
spec phase fills it in.

## Cross-reference checks (when metadata supplies fr ids)

For each FR id in the cluster:
- Confirm the FR exists in the FR store.
- Confirm the FR's `target` matches the milestone's target.
- Confirm the FR's `status` is `open` (or explicitly note if
  re-clustering a `merged` FR for follow-up work).
- Walk `depends_on` one hop and confirm dependencies are
  either merged or in this same cluster.

## Self-review checklist (run before pushing the milestone)

1. **FR-id verification**: each `fr_<id>` cited in the milestone
   summary or work-unit FR list resolves to a real FR in the
   store. (`developer-primary.get_fr` per id.)
2. **Cluster-cohesion sweep**: name the theme in one phrase. If
   you can't, the cluster needs splitting or rebundling.
3. **Status sweep**: `developer-primary.list_frs(status='open',
   target='<target>')` and confirm the FRs in the cluster are
   actually still open.
4. **Title + summary consistency**: the milestone title's noun
   phrase matches the summary's first sentence; the summary's
   FR enumeration matches the work-unit's `frs` list.

## Severity guidance

- `concern`: `fr_cluster_incoherent`, `closed_fr_in_active_cluster`,
  `missing_fr_dependency`. These block spec authoring.
- `comment`: `fr_cluster_too_broad`, `fr_cluster_too_narrow`,
  `summary_misaligned_with_frs`. Cluster-shape suggestions; spec
  phase can recover.
- `nit`: title formatting, summary phrasing, deterministic-stub
  acceptance criteria. Most milestones land with stub acceptance
  and gain real acceptance during spec authoring.
