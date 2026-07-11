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

mkdir -p data/data-3rd-party data/forecast

echo "==> pytest tests/canswim/"
# Recurse package (includes tests/canswim/dashboard/, etc.)
"${PY[@]}" -m pytest tests/canswim/ -q --tb=short

echo "==> local CI OK"
