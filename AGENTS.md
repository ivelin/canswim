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

## Documentation (keep in sync with code)

User-facing docs live under `docs/` and the root `README.md`. **Update them in the same PR** as the behavior they describe—before push/merge (same bar as tests).

| Doc | Source of truth for |
|-----|---------------------|
| `python -m canswim -h` (`src/canswim/__main__.py`) | CLI task names and flags |
| `docs/cli.md` | CLI recipes / env / workflows |
| `docs/run_triggers.md` | Gather/forecast policy (CLI · GUI · MCP) |
| `docs/mcp.md` | MCP tools + opt-in writes |
| `docs/data_store.md` | Parquet vs DuckDB |
| `README.md` | Landing links + short tables |
| `docs/images/{charts,scans,run}.png` | Dashboard screenshots |

**When a PR changes…**

| Change | Also update |
|--------|-------------|
| CLI task/flag or help text | `__main__.py` help; `docs/cli.md` and README if top-level |
| Gather/forecast policy or Run tab labels | `docs/run_triggers.md`; UX string tests; README table if labels change |
| MCP tools add/rename/remove | `docs/mcp.md` tool table (+ README link only if needed) |
| Charts / Scans / Run layout or primary copy | Replace affected `docs/images/*.png` in the same PR |
| Data paths / DuckDB vs parquet semantics | `docs/data_store.md` |

Do **not** maintain a separate design-doc tree that diverges from `docs/run_triggers.md`. Prefer tests that lock consumer strings (`tests/canswim/test_ux_split_labels.py`) and MCP tool registration checks (`tests/canswim/test_docs_sync.py`) over stale prose.
