# canswim agent rules

## Pull requests and merge

- **Never merge a PR unless CI is green.** Do not use `gh pr merge` (or merge via API/UI automation) while checks are failing, pending, or missing when required.
- Before merging: run `gh pr checks <n>` (or `gh pr view --json statusCheckRollup`) and confirm success.
- If CI is red, fix the failure on the PR branch, push, wait for green, then merge.
- Prefer merge only after the latest commit on the PR has a successful Tests workflow (or equivalent required checks).
- **Preferred merge helper:** `./scripts/gh-pr-merge-safe.sh <pr#>` — runs local CI, verifies remote checks are green, then merges.

## Local CI (before push / merge)

- Run the same suite as GitHub Actions: `./scripts/ci-local.sh`
- Agents should run `./scripts/ci-local.sh` **before** `git push` and **before** `gh pr merge`.
- Git hooks (after `./scripts/install-git-hooks.sh`):
  - `pre-push` — blocks push if local CI fails
  - `pre-merge-commit` — blocks local `git merge` commits if local CI fails
  - Bypass (rare): `git push --no-verify` or `SKIP_LOCAL_CI=1`
- `gh pr merge` is **not** a git hook path; always use `gh-pr-merge-safe.sh` or manually confirm `gh pr checks` is green.

## Local verification

- Prefer the `canswim` conda env for CLI and pytest (also used by `ci-local.sh` when present).
- Keep `hfhub_sync=False` for local gather/forecast unless explicitly testing HF sync.
- Do not commit local slimmed `symbol_lists/all_stocks.csv` or gitignored `data/` artifacts from few-symbol runs.
