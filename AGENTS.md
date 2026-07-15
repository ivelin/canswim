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
- **Tests must not write production data.** Reading optional local parquet (e.g. under `data/` even if it symlinks to `~/.canswim/data`) is allowed; all writes go to the per-test temp tree. Enforced by `tests/conftest.py` + `tests/isolation_policy.py` (autouse `data_dir`, block writes under `~/.canswim`). Do not weaken those guards.

## Documentation (keep in sync with code)

User-facing docs live under `docs/` and the root `README.md`. **Update them in the same PR** as the behavior they describe—before push/merge (same bar as tests).

| Doc | Source of truth for |
|-----|---------------------|
| `python -m canswim -h` (`src/canswim/__main__.py`) | CLI task names and flags |
| `docs/cli.md` | CLI recipes / env / workflows |
| `docs/run_triggers.md` | Gather/forecast policy (CLI · GUI · MCP) |
| `docs/mcp.md` | MCP tools + opt-in writes + Streamable HTTP flags |
| `docs/deploy_service.md` | User systemd: private GUI + public apikey MCP |
| `docs/data_store.md` | Parquet vs DuckDB **and schema migrations** |
| `README.md` | Landing links + short tables |
| `docs/images/{charts,scans,run}.png` | Dashboard screenshots |

**When a PR changes…**

| Change | Also update |
|--------|-------------|
| CLI task/flag or help text | `__main__.py` help; `docs/cli.md` and README if top-level |
| Gather/forecast policy or Run tab labels | `docs/run_triggers.md`; UX string tests; README table if labels change |
| MCP tools add/rename/remove | `docs/mcp.md` tool table (+ README link only if needed) |
| Prod deploy / systemd / Tailscale-Funnel MCP path | `docs/deploy_service.md` + README sketch |
| Charts / Scans / Run layout or primary copy | Replace affected `docs/images/*.png` in the same PR |
| Data paths / DuckDB vs parquet semantics | `docs/data_store.md` |
| **Search DB / DuckDB schema** | See **Schema migrations** below |

## Schema migrations (required between app versions)

**Every change that affects the DuckDB search schema must ship with documented, coded, and tested migration steps.** Do not rely on “just rebuild” alone without a versioned path for operators who reuse local DBs.

| Requirement | Where |
|-------------|--------|
| Schema version constant + migration functions | `src/canswim/db_migrations.py` (`CURRENT_SCHEMA_VERSION`, `MIGRATIONS`) |
| Apply on reuse; stamp on full rebuild | `init_search_db` in `src/canswim/db.py` |
| Operator upgrade steps + migration log | `docs/data_store.md` |
| Automated tests for upgrade path | `tests/canswim/test_db_migrations.py` |

**Checklist for a schema-changing PR:**

1. Bump `CURRENT_SCHEMA_VERSION` and add a `Migration` (name + description + `upgrade`).
2. Append a row to the **Migration log** in `docs/data_store.md`.
3. Add/extend tests: old version (or legacy pre-meta DB) → new version.
4. Prefer **additive** migrations; if a full rebuild is required, document that explicitly as the operator step.
5. Run `./scripts/ci-local.sh` before push/merge.

Parquet under `data/` remains the system of record; migrations evolve the **search cache**. Full rebuild (`--same_data False` / Rebuild Charts database) remains the escape hatch and must stamp the current schema version.

Do **not** maintain a separate design-doc tree that diverges from `docs/run_triggers.md`. Prefer tests that lock consumer strings (`tests/canswim/test_ux_split_labels.py`) and MCP tool registration checks (`tests/canswim/test_docs_sync.py`) over stale prose.

## GitHub Pages (https://ivelin.github.io/canswim/)

- The public docs site is **generated from `main`** (`docs/` + `mkdocs.yml`) via `.github/workflows/docs.yml`.
- **Do not** maintain operator docs on a parallel `website` branch (legacy Pages source). That branch is frozen; new content goes on `main` only.
- Merging docs to `main` (paths under `docs/` or `mkdocs.yml`) triggers a Pages redeploy. Use **Actions → Docs → Run workflow** if a manual rebuild is needed.
- After changing operator docs, `./scripts/ci-local.sh` (includes `test_docs_sync`) must pass before push/merge, same as code.
- Local preview: `pip install mkdocs-material && mkdocs serve` (from repo root).
