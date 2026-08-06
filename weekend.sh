#!/usr/bin/env bash
# CANSWIM weekend forecast — all symbols in the Charts/search DB (stock_tickers).
# Prefer systemd: service/canswim-weekend.timer (see docs/deploy_service.md).
# Cron example (Saturday 06:00):
#   0 6 * * 6 /home/YOU/canswim/weekend.sh >> /tmp/canswim-weekend.log 2>&1
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
