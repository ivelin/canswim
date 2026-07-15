#!/usr/bin/env bash
# Local mirror of .github/workflows/tests.yml (Tests job).
# Usage: ./scripts/ci-local.sh
# Skip: SKIP_LOCAL_CI=1 ./scripts/ci-local.sh   (or git push --no-verify)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${SKIP_LOCAL_CI:-}" == "1" ]]; then
  echo "SKIP_LOCAL_CI=1 — skipping local CI"
  exit 0
fi

echo "==> local CI (matches GitHub Actions Tests workflow)"

# Prefer project conda env when available (faster than bare pip -e . every push)
PY=()
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  if conda env list 2>/dev/null | awk '{print $1}' | grep -qx 'canswim'; then
    # conda run keeps isolation without mutating the caller shell
    PY=(conda run -n canswim --no-capture-output python)
    echo "    using: conda env canswim"
  fi
fi
if [[ ${#PY[@]} -eq 0 ]]; then
  PY=(python)
  echo "    using: $(command -v python) ($("${PY[@]}" -V 2>&1))"
fi

# Tests use per-test temp data dirs (tests/conftest.py). Never mkdir/write under
# repo data/ when it is a symlink to production ~/.canswim/data.
if [[ -L data ]]; then
  echo "    note: data/ is a symlink — not creating dirs there (prod isolation)"
elif [[ -d data ]]; then
  # Real directory (CI / fresh checkout): empty scaffolding only, no prod home.
  mkdir -p data/data-3rd-party data/forecast
else
  mkdir -p data/data-3rd-party data/forecast
fi

# Ensure editable install has test deps (pytest moved to extras_require[dev])
if ! "${PY[@]}" -c "import pytest" 2>/dev/null; then
  echo "==> installing package with [dev] extras for pytest"
  "${PY[@]}" -m pip install -q -e ".[dev]"
fi

echo "==> pytest tests/canswim/"
# Recurse package (includes tests/canswim/dashboard/, etc.)
# Isolation is enforced by tests/conftest.py (autouse temp data_dir + write guards).
"${PY[@]}" -m pytest tests/canswim/ -q --tb=short

echo "==> local CI OK"
