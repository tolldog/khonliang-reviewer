# Spec-review rubric

**Purpose**: review a `specs/MS-*/spec.md`-shaped full-document
planning artifact (problem, design principle, scope, acceptance,
open questions, dependencies, implementation notes, revision
history). Spec authors land code work after a spec is approved;
findings here gate that hand-off, so the rubric is the heaviest
of the artifact-review set.

## Finding categories

Each finding the reviewer surfaces should fall into one of these
categories. New categories are added when a real flag surfaces
that doesn't fit — that's how the rubric grows.

### `fabricated_identifier`
A concrete name in the spec — file path, class name, table column,
function id, config key, FR/spec/milestone id, external CLI flag —
that does NOT exist in the repository. Distinguish from
proposed-new identifiers, which the spec must explicitly tag (e.g.
"new module `reviewer/distill/`").

**Examples (drawn from PR #24)**:
- `.reviewer/examples/` cited as a loader path; real path is
  `.reviewer/prompts/examples/`.
- `qwen2.5-coder_14b.yaml` cited as model-config filename; real
  filename strips the `:tag` to `qwen2.5-coder.yaml` per
  `_model_stem`.
- `UsageStore.review_records` cited as a persisted table; only
  `reviewer_usage` and `model_pricing` exist.
- `ProviderDecision` cited as the existing rule-table output type;
  real type is `PolicyDecision`.
- `GitHubClient` cited as the HTTP wrapper; real class is
  `ReviewerGithubClient`.

### `inconsistent_internal_reference`
Two sections of the same spec say contradictory things — typically
because a revision touched section A but left analogous section B
on the old wording. The most reliable place to catch this is a
re-read sweep across §Scope, §Acceptance, §Open Questions,
§Out of scope, §Implementation Notes, §Dependencies, §Revision
history.

**Examples**:
- §Out-of-scope says "no per-caller runtime overrides of
  DistillConfig" but Acceptance #1 has callers passing `audience`
  inline (resolved by declaring `audience` a request-shape input,
  not a per-call field override).
- §Scope says "consensus is step 1 of the distill pipeline" but
  Open Question #2 + Implementation Notes say consensus likely
  lives at the selector layer.
- Acceptance #6 says "finding ids and SHAs only" while the schema
  also stores classification + timestamp + rationale_code.
- §Implementation Notes says "three new MCP skills" while §Scope
  enumerates four.

### `ambiguous_acceptance`
An acceptance criterion that can't be objectively closed —
prose that doesn't name a function call, a return shape, an
observable side effect, or a test pattern. "Works correctly"
without a check; "is fast enough" without a number;
"behaves consistently" without a contract.

**Examples**:
- "The pipeline runs efficiently" — no number, no measurement
  point.
- "Existing callers continue to work" — no enumeration of which
  callers, no test invocation.

### `missing_dependency`
The spec depends on persisted state, an upstream FR, or a code
path that the §Dependencies section does not call out. Common
failure: implicit reliance on a sibling FR's data shape.

**Examples (PR #24)**:
- MS-C address-rate scraper cited `UsageStore.review_records` as
  the seed source without naming a dependency on that table
  existing — table didn't exist; corrected to use the existing
  `reviewer_usage` rows + GitHub recovery.
- MS-D `num_ctx` per-model config-file override implicitly
  depended on the `.reviewer/` loader being extended; original
  text didn't call that out and made the dependency invisible.

### `open_question_should_be_decision`
An §Open Questions entry that the rest of the spec already acts on
as if it had a specific resolution. The §Design Principle / §Scope /
§Acceptance has implicitly committed; leaving the question "open"
mis-signals reviewer scope.

**Examples (PR #24)**:
- Open Question #1 said "audience→rule-table wiring is deferred"
  while §Design Principle, §Scope, and Acceptance #1 all treated
  audience as a first-class rule-table input. Resolved.
- Open Question #2 said "consensus placement deferred" while
  §Scope listed consensus as a pipeline step. Resolved.

### `unjustified_out_of_scope`
A §Out-of-scope item that's listed without a reason, OR an item
that's *in* scope but materially relies on something §Out-of-scope
declares missing. The rubric enforces that out-of-scope items
either explain "why later" OR get a follow-up FR pointer.

### `design_principle_violation`
The §Scope or §Acceptance proposes work that violates a stated
§Design Principle. Typically a "we keep transforms loosely
coupled" principle followed by a transform that hard-codes a
sibling transform's behavior.

### `aspirational_claim_unbacked`
A statement of the form "X is enforced", "Y is guaranteed", "Z
never happens" without a concrete mechanism (a `CHECK` constraint,
a type-system guarantee, a test, an architectural invariant).
Reword the claim or back it with a citation.

**Examples (PR #24)**:
- "Structurally enforced" claimed without a `CHECK` constraint —
  the column was free-form `TEXT`, no schema enforcement at all.
- "No warnings emitted" claimed in Acceptance while §Scope
  describes a warning being logged.
- "Only finding ids and SHAs persisted" claimed while the schema
  also stores classification + timestamps + rationale codes.

### `patch_introduced_fabrication`
A fix-the-flagged-line patch introduces a new fabricated
identifier in its replacement text. The most reliable indicator
this happened: a "rev N: align with code" entry in §Revision
history, paired with a name in the patch text that doesn't
exist in the repo.

**Example (PR #24)**: rev10 was an explicit "switch from `gh api`
to the SDK" commit and in the same commit invented
`GitHubClient` as the wrapper class — but the real class is
`ReviewerGithubClient`. Patch-introduced fabrication is the
hardest category to catch via patch-only review; it requires a
re-read of the patch text itself.

## Self-review checklist (run before pushing)

1. **Identifier verification**: every concrete name in the patch
   gets `grep`-checked against the repo, OR explicitly tagged as
   proposed-new.
2. **Cross-section sweep**: when a section changes, re-read
   §Scope / §Acceptance / §Open Questions / §Out-of-scope /
   §Implementation Notes / §Dependencies / §Revision history
   for analogous claims that may now be inconsistent.
3. **Aspirational-claim audit**: every "enforced" / "guaranteed"
   / "never" needs a citation or a reword.
4. **Patch-introduced fabrication check**: re-read the patch
   text itself for new identifiers; verify those.

If a step surfaces nothing, that's a real signal — the most
common churn driver is sweep #2 returning a contradiction the
patch-author missed.

## Severity guidance for spec findings

- `concern`: any of `fabricated_identifier`,
  `inconsistent_internal_reference`, `missing_dependency`,
  `aspirational_claim_unbacked` whose claim is load-bearing
  (e.g. a privacy invariant, a backward-compat promise).
- `comment`: `ambiguous_acceptance`,
  `open_question_should_be_decision`,
  `unjustified_out_of_scope`, `aspirational_claim_unbacked`
  on a non-load-bearing claim.
- `nit`: prose-clarity rewords, ordering tweaks, formatting
  asymmetries that don't change the contract.

Specs gate code work; severity skews higher than for code
review because a wrong spec costs implementer time downstream.
