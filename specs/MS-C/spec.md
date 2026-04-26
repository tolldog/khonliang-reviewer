# MS-C: Closed-loop reviewer calibration

**Milestone:** `ms_reviewer_d6f9a46c`
**Target:** `reviewer`
**Status:** proposed
**FRs:**
- `fr_reviewer_eeebcba2` — Finding-address-rate tracking (scrape GitHub reactions/replies, aggregate per `(model, repo, kind)`)
- `fr_reviewer_570aad54` — Flag excellent findings → auto-promote to `.reviewer/prompts/examples` library

## Problem

The reviewer ships findings; some get acted on, most do not. Today there is no signal telling us *which* findings developers actually addressed — only the raw output and the LLM token cost. Two consequences:

1. **Quality is invisible.** Two models could produce the same number of findings at the same severity distribution; one's findings get fixed, the other's get ignored. The benchmark sweep harness (PR#23) captures latency, token cost, and severity histograms — but not the only metric that ultimately matters: did the developer change the code in response?
2. **The examples corpus stays static.** MS-B seeds `.reviewer/prompts/examples/` from human-curated milestone_store invariants. That's a fine starting point, but the corpus does not grow as the reviewer improves. Real-world high-signal findings stay buried in old PRs; the prompt loader keeps shipping the same handful of examples regardless of which kinds of findings actually move the needle.

The two FRs in this milestone close the loop: address-rate becomes the quality signal, and that signal is what flags findings as "excellent" and worth promoting into the live examples library.

## Design Principle

**Address-rate is a product metric, not a model metric.** A finding is "addressed" when the diff between the reviewed commit and a later merge commit shows that the file/region the finding pointed at was modified in a way consistent with the suggestion. That's a heuristic — false positives (developer changed that region for unrelated reasons) and false negatives (developer chose a different fix elsewhere) are unavoidable. The metric is useful in *aggregate* per `(model, repo, kind)`; it is *not* a per-finding ground truth. Treat it as a population statistic.

**One signal source: GitHub.** Address-rate scraping reads PR review comments, commit history, and reactions/replies via the existing `reviewer.github_client.GitHubClient` wrapper around `githubkit` (the same path the reviewer already uses to post review comments — no new dependency, no new auth surface). We do NOT shell out to `gh api`: per the user-level engineering directive, GitHub work goes through the Python SDK rather than subprocess, and `githubkit>=0.13` is already a declared dep (`pyproject.toml:15`). We also do not stand up a separate webhook listener or write a CI plugin — that's a different scale of investment. The reviewer agent already runs on dev hardware; periodic batched scrapes against repos the reviewer has touched are sufficient.

**Promotion is conservative.** Auto-promote only fires when (a) the finding has a high address-rate against its severity-and-kind cohort, *and* (b) it is structurally distinct from existing examples (per a simple title/body similarity check). The library does not double up on the same lesson; the promotion threshold stays high enough that ~5–10 findings/quarter clear the bar, not hundreds.

**Promoted examples are reviewable.** The existing prompt loader expects ONE file per `(kind, severity)` cell at `.reviewer/prompts/examples/<kind>/<severity>.md` — the entire file content becomes the few-shot block for that cell. Auto-promotion can't be a per-finding `git mv`; it has to *append* a new example block into the existing severity file. To keep the human-in-the-loop step explicit, candidates first land as standalone files at `.reviewer/prompts/examples/_pending/<finding_id>.md` (one file per candidate, with frontmatter naming the target cell). Nothing reaches the live cell file until an operator (or a future bus skill) runs the promotion step, which appends the candidate's body into `<kind>/<severity>.md` and removes the pending file. The pending area is the staging line; promote-to-live is a deliberate separate action.

## Scope

### In scope

- **`reviewer.scrape.address_rate`** module — periodic scrape against repos / PRs the reviewer has reviewed, seeded from existing persisted `reviewer_usage` rows plus GitHub PR metadata (NOT from a new `review_records` table — `UsageStore` today persists only `reviewer_usage` + `model_pricing`):
  - Use `reviewer_usage` to identify review runs that posted output for a GitHub PR, including at minimum the repo, PR number (or URL), and model. Those rows are the scrape seed set.
  - For each seeded PR, fetch PR review comments, comment threads, and the commits between the reviewed SHA and the merge SHA via `reviewer.github_client.GitHubClient` (githubkit). Concretely: `pulls.async_get(owner, repo, pull_number)` for PR metadata + merge_commit_sha; `pulls.async_list_review_comments(owner, repo, pull_number)` for the comment list; `reactions.async_list_for_pull_request_review_comment(...)` for thumbs-up/thumbs-down + ad-hoc reaction signals; `repos.async_compare_commits(owner, repo, base=reviewed_sha, head=merge_sha)` for diff hunks between reviewed-SHA and merge-SHA. Replies to review comments come back inline on the review-comment list (`in_reply_to_id` field). All calls reuse the existing `GitHubClient` token discovery (delegating to `reviewer.credentials.get_github_token`).
  - Recover individual findings from the posted PR review comments rather than from local per-finding persistence. Each scraped finding gets a stable `finding_id` derived deterministically from `(repo, pr_number, review_comment_id, reviewed_sha)` — concretely, `sha256(f"{repo}|{pr_number}|{review_comment_id}|{reviewed_sha}")[:16]`, prefixed `finding_` for human readability. The same id is used as the `address_rate` table key, the `_pending/<id>.md` candidate filename, and the `source_finding_id` frontmatter value, so a candidate can always be traced back to its source comment without ambiguity. Classification inputs come from GitHub metadata (`path`, `line` or diff hunk position, thread replies/reactions, reviewed SHA) together with the finding `kind` / `severity` parsed from the comment-body format the reviewer already posts. (The reviewer's `ReviewFinding` library type does NOT carry an id field today; `finding_id` is a scraper-derived value, not a `ReviewFinding` attribute.)
  - If a PR is closed without merge, the review comment was deleted, or the posted comment body no longer contains enough structured data to recover `kind` / `severity`, skip classification for that finding and record it as `inconclusive` rather than inventing missing fields.
  - Heuristic-classify each recovered finding as `addressed` / `not_addressed` / `inconclusive`:
    - `addressed`: the file mentioned in `finding.path` has a diff hunk between reviewed-SHA and merge-SHA that overlaps `finding.line ± 5` *and* the comment thread shows no rebuttal reaction (👎 / "out of scope" reply).
    - `inconclusive`: file modified but region untouched, comment thread shows mixed signal, or GitHub metadata is insufficient to recover the finding cleanly.
    - `not_addressed`: file unchanged, or comment thread shows explicit "won't fix" / "out of scope" signal.
  - Persist results in a new `address_rate` table in the existing reviewer SQLite store, keyed on `(finding_id, reviewed_sha, merge_sha)` so re-runs are idempotent. `finding_id` here is the same scraper-derived id defined above (single source of truth across the table, candidate filenames, and candidate frontmatter).
- **Aggregation rollup** — `address_rate_summary(model, repo, kind, severity)` returns the rolling 30-day percentage of `addressed` over `(addressed + not_addressed)` (excludes `inconclusive` from both numerator and denominator). Surface the summary via a new `reviewer.usage_summary` extension.
- **Auto-promote candidate generator** — for each finding marked `addressed` in the last N=30 days:
  - Compute its rank within its `(model, repo, kind, severity)` cohort.
  - When rank ≥ 90th percentile, run a structural-similarity check against the body of `.reviewer/prompts/examples/<kind>/<severity>.md` (the loaded few-shot block for that cell — concatenated example content). The check is deliberately simple: title token overlap < 40% *and* body token overlap < 30% (Jaccard, lowercased, stop-words removed) against the existing block as a whole.
  - When both checks pass, write the candidate to `.reviewer/prompts/examples/_pending/<finding_id>.md` — one file per candidate. Frontmatter declares the target cell (`kind`, `severity`) so the promote step knows where to append.
- **Operator skills** — four new MCP skills to drive the loop:
  - `compute_address_rates(repo, since_days=30)` — runs the scrape + classification + storage step; idempotent.
  - `address_rate_summary(model?, repo?, kind?, severity?, since_days=30)` — read-only rollup.
  - `list_pending_examples()` — enumerates `.reviewer/prompts/examples/_pending/` so the operator can see what's queued; surfaces age + cohort + percentile per candidate.
  - `promote_pending_example(pending_id, *, separator='\n\n---\n\n')` — appends the candidate's body content into the target cell file (`.reviewer/prompts/examples/<kind>/<severity>.md`), creating the file + parent dirs if missing, with the configured separator between existing and new content; then removes the pending file. Returns the post-append target file path + a diff hash so the operator can verify the change.
- **Composition with MS-B**: MS-B's `.reviewer/prompts/examples/` corpus is the seed; MS-C *appends* into the same cell files via the promote step. No prompt-loader changes required (it already concatenates the file contents into the few-shot block). MS-B is sequenced first.

### Out of scope

- **GitHub-side webhook listener** or CI integration. Pull-only scrape from dev hardware is sufficient; webhook architecture is a separate FR if scale demands it.
- **Per-finding ground truth.** The classification is a population heuristic; we explicitly do not surface per-finding `addressed: true/false` to end users. The summary skill returns aggregates only.
- **Demoting / removing existing examples** when their address rate drops. First cut: examples are append-only via promotion; the corpus is curated by humans on the way out. Future FR if examples ever go stale.
- **Cross-repo signal pooling.** A finding from `tolldog/khonliang-reviewer` and a textually similar one from a sibling repo are treated as independent samples in this cut. Cross-repo "this lesson generalizes" detection is a follow-up.
- **Auto-promotion to the live examples directory** without operator review. The `_pending/` staging area is the line; promotion-to-live is a deliberate separate action so the operator (currently the user) sees the candidate set and can reject anything off-base.

## Acceptance Criteria

1. `compute_address_rates(repo="tolldog/khonliang-reviewer", since_days=30)` runs end-to-end against the live repo and persists at least 5 classified rows in the `address_rate` table without raising. Re-running with the same args is a no-op (idempotent on `(finding_id, reviewed_sha, merge_sha)`).
2. `address_rate_summary(model="qwen2.5-coder:14b", repo="tolldog/khonliang-reviewer", kind="pr_diff", severity="concern")` returns a percentage (0.0–100.0) plus the underlying `(addressed, not_addressed, inconclusive)` counts. Returns `null` percentage with non-zero counts when the denominator is zero.
3. The auto-promote step writes at least one candidate file to `.reviewer/prompts/examples/_pending/<finding_id>.md` when run against the seed reviewer-store dataset. Frontmatter on the candidate names the target cell (`kind`, `severity`); body is the few-shot block content suitable for appending into the live cell file.
3a. `promote_pending_example(pending_id)` reads a pending candidate, appends its body (separated by the configured rule) into the target `.reviewer/prompts/examples/<kind>/<severity>.md`, removes the pending file, and returns the post-append target path. Re-running the prompt loader sees the new content as part of the cell's few-shot block without any loader code changes.
4. The structural-similarity check rejects a candidate that overlaps an existing example at >40% title-Jaccard or >30% body-Jaccard. Tested with: pasting an existing example back in as a candidate → rejected with a clear "duplicate of <existing-id>" reason in the candidate file's frontmatter (so the operator sees why it was suppressed if they go looking).
5. The 30-day rolling window respects the SQLite store's `created_at` semantics — re-running computation 90 days later does not re-classify findings already classified, and the rollup correctly excludes finding outside the window.
6. **Privacy/leakage check**: scraped data persists only non-content metadata needed for classification/rollup — finding ids, SHAs, and small fixed enums/timestamps (`classification`, `classified_at`, `rationale_code`). The `address_rate` table never stores PR comment or reaction bodies; the `rationale_code` column is `CHECK`-constrained to a closed enum vocabulary so raw text can't sneak in. (Reasoning: even within local trusted env, persisting third-party comment text bloats the DB and complicates eventual cross-machine sync.)
7. The new `usage_summary` extension surfaces address-rate alongside existing latency / token / cost metrics, gated behind an `include_address_rate: true` skill argument (so existing summary callers don't pay the rollup cost they don't ask for). `usage_summary` is an MCP/bus skill taking structured JSON args, not a CLI; the gate is a boolean skill kwarg, not a CLI flag.
8. Tests cover: classification heuristic (each of the three categories given a constructed mini-repo fixture), rollup math (zero-denominator, mixed cohort), structural-similarity false positive (different finding, same boilerplate phrasing), candidate-file formatting round-trip.

## Open Questions

1. **Address-rate vs. accept-rate for cross-vendor reviews.** Today reviewer findings post as PR comments. When a Copilot review *also* posts on the same PR, the address signal could conflate the two (a developer addressing a Copilot finding might be miscredited to a reviewer finding nearby). The classifier needs to disambiguate by author. Tentative: filter the comment-author check to `khonliang-reviewer-bot` (or whatever the configured PR-comment author is) before counting `addressed`. Document the assumption in the spec's first revision.
2. **What counts as "the merge SHA"** when a PR is rebased + squashed? The reviewed SHA comes from GitHub review metadata / PR head SHA at review time — not from `UsageStore` persistence (the current `reviewer_usage` schema does not store a commit SHA). The merge SHA is the squash-merge commit on `main`. Heuristic: walk `git log --first-parent main` for the squash that introduced the PR's content, identified by `Merged-via` trailer or PR-number reference in the commit message. Edge: PRs merged via fast-forward have no distinct merge SHA — the table stores the sentinel `no_merge_sha` (see §Implementation Notes schema) so primary-key idempotency stays well-defined. Tentative: collapse `addressed` and `inconclusive` into the same bucket when no distinct merge SHA exists. If we later need the reviewed SHA in usage persistence, that would require explicitly extending `UsageEvent` + `reviewer_usage` (separate FR).
3. **Promotion threshold tuning.** 90th percentile within the cohort is a starting guess. Worth instrumenting the candidate generator to log the rank distribution for the first month so we can adjust before too many candidates accumulate (or too few).
4. **`.reviewer/prompts/examples/_pending/` lifecycle.** When a candidate sits there for >90 days unreviewed, should it auto-expire? Tentative: surface in `list_pending_examples()` with an age field, but do not auto-delete (deletion is operator's choice).

## Dependencies

- **Hard-blocks on:** MS-B (`fr_reviewer_afd4bab1` — `.reviewer/prompts/examples/` seed corpus). Auto-promote needs the live corpus structure to compare candidates against.
- **Composes with:** existing `UsageStore` schema — `reviewer_usage` rows seed the scraper, and the new `address_rate` table is a sibling table in the same SQLite store. No migration of existing rows needed; no new persistence required for individual findings (they are recovered from GitHub on each scrape).
- **External:** None new. Uses the existing `githubkit` dep (already in `pyproject.toml:15`) and the existing `reviewer.github_client.GitHubClient` wrapper. Auth resolves through `reviewer.credentials.get_github_token` exactly like the existing review-comment-posting path; no additional credentials or host binaries required. (Notably: NOT shelling out to `gh api` — see §Design Principle "One signal source: GitHub".)

## Implementation Notes (non-binding)

- Module layout:
  - `reviewer/scrape/address_rate.py` — scrape + classify + persist.
  - `reviewer/scrape/promote.py` — candidate generator + similarity check.
  - `reviewer/skills/address_rate.py` — four new MCP skills (`compute_address_rates`, `address_rate_summary`, `list_pending_examples`, `promote_pending_example`).
- New `address_rate` table:
  ```sql
  CREATE TABLE address_rate (
    finding_id TEXT NOT NULL,
    reviewed_sha TEXT NOT NULL,
    merge_sha TEXT NOT NULL,        -- sentinel 'no_merge_sha' when no merge commit exists (e.g. fast-forward merge)
    classification TEXT NOT NULL CHECK (
      classification IN ('addressed', 'not_addressed', 'inconclusive')
    ),
    classified_at REAL NOT NULL,
    rationale_code TEXT NOT NULL CHECK (
      rationale_code IN (
        'file_unchanged',
        'region_overlap',
        'rebuttal_reaction',
        'wont_fix_reply',
        'mixed_signal',
        'no_merge_sha'
      )
    ),
    PRIMARY KEY (finding_id, reviewed_sha, merge_sha)
  );
  ```
  Persists only finding ids, SHAs, classification, and a fixed enum code naming the heuristic that fired. The PK is `(finding_id, reviewed_sha, merge_sha)` and `merge_sha` is `NOT NULL` with a sentinel value (`'no_merge_sha'`) for the fast-forward case so re-runs stay idempotent under the composite key without nullable-PK semantics ambiguity. Any human-readable rationale is derived at runtime from the enum + the row's other fields. **Raw GitHub comment / reaction text is never stored** in this table (or anywhere else by this milestone) — Acceptance Criterion #6 enforced structurally by the `CHECK`-constrained closed-enum vocabulary, not by convention.
- Similarity check via `re.findall(r"\w+", text.lower())` + Python `set` Jaccard. No external dep on tokenizers / embeddings.
- Candidate file format (`.reviewer/prompts/examples/_pending/<finding_id>.md` — where `<finding_id>` is the deterministic scraper-derived id defined in §Scope above: `sha256(f"{repo}|{pr_number}|{review_comment_id}|{reviewed_sha}")[:16]` with the `finding_` prefix). NOT an FR id, NOT a `ReviewFinding` attribute (that library type doesn't carry an id today). The frontmatter `source_finding_id` is the same value:
  ```markdown
  ---
  source_finding_id: finding_<16-hex>
  target_kind: pr_diff
  target_severity: concern
  cohort: ollama/qwen2.5-coder:14b/pr_diff/concern
  rank_percentile: 94
  proposed_at: 2026-04-26
  similarity_check: passed
  ---

  ## Title
  ...

  ## Body
  ...
  ```

## Revision history

- **rev 1** (2026-04-26): initial spec, author: Claude. MS-B sequencing dependency flagged. Open questions on cross-vendor disambiguation + merge-SHA semantics flagged for first review pass.
- **rev 2** (2026-04-26): correct loader path from `.reviewer/examples/` to `.reviewer/prompts/examples/` and rework promotion model: the loader expects ONE file per `(kind, severity)` cell — not one file per finding — so promote-to-live can't be a `git mv`. Revised model: candidates land at `_pending/<finding_id>.md` (one file per candidate); a new `promote_pending_example` skill APPENDS the candidate's body into the target cell file with a configurable separator. Both fixes per Copilot R1 on PR#24, grounded in `reviewer/config/prompts.py:195-260`.
- **rev 3** (2026-04-26): correct skill count in §Implementation Notes from "three" to "four" — rev2 added `promote_pending_example` to §Scope but didn't update the module ownership line (per Copilot R2 on PR#24).
- **rev 4** (2026-04-26): two structural cleanups per Copilot R3 on PR#24. (a) Replaced free-form `rationale TEXT` column in the `address_rate` table with a closed-enum `rationale_code` column — Acceptance #6's "finding ids and SHAs only" constraint is now enforced structurally by the schema rather than relying on convention to keep raw GitHub comment text out of the column. Human-readable rationale is derived at runtime from the enum. (b) Renamed candidate-file frontmatter `source_finding_id: fr_<id>` to `source_finding_id: finding_<id>` and explicitly noted the field is a `ReviewFinding` id (not an FR id) — earlier wording read like a functional-requirement reference.
- **rev 5** (2026-04-26): correct scrape-seed reference per Copilot R4 on PR#24. Earlier revs cited `UsageStore.review_records` as the scrape seed, but that table doesn't exist — `UsageStore` today persists `reviewer_usage` + `model_pricing` only. Reworked §Scope to seed from existing `reviewer_usage` rows + GitHub PR metadata, recover findings from GitHub PR review comments (rather than per-finding local persistence), and key the new `address_rate` table by GitHub review-comment id. Removed implicit dependency on a new persistence layer; MS-C now requires only the new `address_rate` sibling table.
- **rev 6** (2026-04-26): three schema-rigor cleanups per Copilot R5 on PR#24. (a) Open Question #2 still cited `UsageStore.review_records.commit_sha` despite rev5 already removing that reference from §Scope. Updated to source `reviewed_sha` from GitHub review metadata / PR head SHA at review time. (b) `merge_sha TEXT` (nullable) inside `PRIMARY KEY` left idempotency semantics ambiguous; bumped to `NOT NULL` with sentinel `'no_merge_sha'` for the fast-forward case. (c) `rationale_code` was claimed as "structural enforcement" but had no `CHECK` constraint; added explicit `CHECK (rationale_code IN (...))` plus `CHECK` on `classification` so the privacy-and-vocabulary invariants are DB-enforced rather than convention-enforced. Renamed `won't_fix_reply` → `wont_fix_reply` for SQL-literal compatibility.
- **rev 7** (2026-04-26): correct Acceptance #6 wording per Copilot R6 on PR#24. Earlier wording said "scraped data persists finding ids and SHAs only" but the schema also stores `classification`, `classified_at`, `rationale_code` — non-content metadata that the privacy invariant is fine with. Reworded to "non-content metadata (finding ids, SHAs, fixed enums, timestamps); never PR comment/reaction bodies" so the criterion matches the actual schema.
- **rev 8** (2026-04-26): correct Acceptance #7 surface shape per Copilot R8 on PR#24. Earlier wording proposed `--include-address-rate` as a CLI flag, but `usage_summary` is an MCP/bus skill (structured JSON args, no CLI). Reshaped the gate as `include_address_rate: true` boolean skill argument so the API shape stays consistent with the rest of the reviewer surface.
- **rev 9** (2026-04-26): unify `finding_id` definition per Copilot R10 on PR#24. Earlier revs left it ambiguous — §Scope keyed scraped findings by GitHub review-comment id, but the candidate-file format said the id came from a `ReviewFinding` (a library type that does NOT carry an id field today). Standardized on a deterministic scraper-derived id: `sha256(f"{repo}|{pr_number}|{review_comment_id}|{reviewed_sha}")[:16]` with the `finding_` prefix. The same id flows through the `address_rate` table key, the `_pending/<id>.md` candidate filename, and the `source_finding_id` frontmatter — single source of truth, traceable back to the source comment without ambiguity.
- **rev 10** (2026-04-26): replace all `gh api` subprocess references with `githubkit` SDK calls via the existing `reviewer.github_client.GitHubClient` wrapper. Surfaced during a user-prompted re-validation pass against user-level engineering directive "GitHub: PyGithub or githubkit — not gh". `githubkit>=0.13` is already a declared dep (`pyproject.toml:15`) and the reviewer already uses it for posting review comments; the address-rate scraper inherits that path. Documented the specific githubkit endpoints used (pulls.async_get, async_list_review_comments, reactions.async_list_for_pull_request_review_comment, repos.async_compare_commits) and removed the "gh CLI on the host" external dependency.
