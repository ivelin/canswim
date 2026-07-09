# canswim agent rules

## Pull requests and merge

- **Never merge a PR unless CI is green.** Do not use `gh pr merge` (or merge via API/UI automation) while checks are failing, pending, or missing when required.
- Before merging: run `gh pr checks <n>` (or `gh pr view --json statusCheckRollup`) and confirm success.
- If CI is red, fix the failure on the PR branch, push, wait for green, then merge.
- Prefer merge only after the latest commit on the PR has a successful Tests workflow (or equivalent required checks).

## Local verification

- Prefer the `canswim` conda env for CLI and pytest.
- Keep `hfhub_sync=False` for local gather/forecast unless explicitly testing HF sync.
- Do not commit local slimmed `symbol_lists/all_stocks.csv` or gitignored `data/` artifacts from few-symbol runs.
