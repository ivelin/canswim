#!/usr/bin/env bash
# CANSWIM weekend forecast — all symbols in the Charts/search DB (stock_tickers).
# One-shot / manual. For recurring runs, use the in-process APScheduler inside
# canswim-mcp (MCP_ALLOW_RUNS=1) — see docs/deploy_service.md §3b.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

export CANSWIM_DIR="${CANSWIM_DIR:-${ROOT}}"
export CANSWIM_HOME="${CANSWIM_HOME:-${HOME}/.canswim}"
export data_dir="${data_dir:-${CANSWIM_HOME}/data}"
export db_file="${db_file:-canswim_local.duckdb}"
export hfhub_sync="${hfhub_sync:-False}"
export PYTHONPATH="${CANSWIM_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -r "${HOME}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${HOME}/.env"
  set +a
  export FMP_API_KEY="${FMP_API_KEY:-${FMP_API_Key:-}}"
fi

PYTHON="${CANSWIM_PYTHON:-${PYTHON:-python3}}"
echo "CANSWIM Weekend: Starting (data_dir=${data_dir}) …"
# Live week start for all stock_tickers (add --catchup for monthly origins).
exec "${PYTHON}" -m canswim weekend "$@"
