#!/usr/bin/env bash
# Local mirror of .github/workflows/tests.yml (Tests job).
#
# Must stay identical to remote for merge decisions:
#   GHA steps:  pip install -e ".[dev]"
#               mkdir -p data/data-3rd-party data/forecast
#               python -m pytest tests/canswim/ -v
#   GHA python: 3.10 (matrix) — local uses best available interpreter with
#               working darts/torch (prefer CI_PYTHON / python3.10 / project env).
#
# Usage: ./scripts/ci-local.sh
# Skip: SKIP_LOCAL_CI=1 ./scripts/ci-local.sh   (or git push --no-verify)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${SKIP_LOCAL_CI:-}" == "1" ]]; then
  echo "SKIP_LOCAL_CI=1 — skipping local CI"
  exit 0
fi

echo "==> local CI (identical suite to GitHub Actions Tests workflow)"

# Interpreter selection: explicit override, then common CI mirrors, then conda, then PATH.
PY=()
if [[ -n "${CI_PYTHON:-}" ]]; then
  PY=("${CI_PYTHON}")
  echo "    using: CI_PYTHON=${CI_PYTHON} ($("${PY[@]}" -V 2>&1))"
elif [[ -x "${HOME}/.venvs/canswim-ci310/bin/python" ]] \
  && "${HOME}/.venvs/canswim-ci310/bin/python" -c "from darts.models import TiDEModel" 2>/dev/null; then
  PY=("${HOME}/.venvs/canswim-ci310/bin/python")
  echo "    using: canswim-ci310 (Python 3.10 GHA mirror) ($("${PY[@]}" -V 2>&1))"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  if conda env list 2>/dev/null | awk '{print $1}' | grep -qx 'canswim'; then
    PY=(conda run -n canswim --no-capture-output python)
    echo "    using: conda env canswim"
  fi
fi
if [[ ${#PY[@]} -eq 0 ]]; then
  PY=(python)
  echo "    using: $(command -v python) ($("${PY[@]}" -V 2>&1))"
fi

# Same scaffolding as GHA "Create data directories". Never mkdir under a symlink
# to production ~/.canswim/data (tests/conftest.py uses per-test temps).
if [[ -L data ]]; then
  echo "    note: data/ is a symlink — not creating dirs there (prod isolation)"
elif [[ -d data ]]; then
  mkdir -p data/data-3rd-party data/forecast
else
  mkdir -p data/data-3rd-party data/forecast
fi

# Same as GHA "Install dependencies" when pytest missing
if ! "${PY[@]}" -c "import pytest" 2>/dev/null; then
  echo "==> installing package with [dev] extras for pytest (GHA: pip install -e \".[dev]\")"
  "${PY[@]}" -m pip install -q -e ".[dev]"
fi

echo "==> python -m pytest tests/canswim/ -v   # same path + flags as GHA Tests"
# Isolation: tests/conftest.py (autouse temp data_dir + write guards).
"${PY[@]}" -m pytest tests/canswim/ -v --tb=short

echo "==> local CI OK (suite matches .github/workflows/tests.yml)"
