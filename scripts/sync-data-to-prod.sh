#!/usr/bin/env bash
# Safely merge local canswim data into a remote prod host.
#
# Design (no silent wipe of prod-only symbols):
#   1. Snapshot remote data_dir
#   2. rsync forecast/ + most 3rd-party files WITHOUT --delete
#   3. Stage local price parquet; merge on remote (union by Symbol,Date)
#   4. Brief stop of canswim systemd units, rebuild DuckDB search cache, restart
#
# Usage:
#   ./scripts/sync-data-to-prod.sh [user@host]
# Env:
#   LOCAL_DATA   default: <repo>/data
#   REMOTE_HOME  default: ~/.canswim  (resolved on remote)
#   DRY_RUN=1    print plan only
set -euo pipefail

REMOTE="${1:-openclaw@spark-9045}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DATA="${LOCAL_DATA:-$ROOT/data}"
DRY_RUN="${DRY_RUN:-0}"

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "$LOCAL_DATA/data-3rd-party" ]] || die "missing $LOCAL_DATA/data-3rd-party"
[[ -d "$LOCAL_DATA/forecast" ]] || die "missing $LOCAL_DATA/forecast"

RSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=15"

# Absolute remote data root (rsync does not expand $HOME on the far side)
REMOTE_DATA="$("${RSH[@]}" "$REMOTE" 'echo "${HOME}/.canswim/data"')"
[[ -n "$REMOTE_DATA" ]] || die "could not resolve remote data dir"
REMOTE_DATA="${REMOTE_DATA//$'\r'/}"

log "Local data:  $LOCAL_DATA"
log "Remote host: $REMOTE"
log "Remote data: $REMOTE_DATA"

if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY_RUN=1 — no changes"
  "${RSH[@]}" "$REMOTE" "du -sh '${REMOTE_DATA}' '${REMOTE_DATA}/forecast' '${REMOTE_DATA}/data-3rd-party' 2>/dev/null; systemctl --user is-active canswim-dashboard.service canswim-mcp.service 2>/dev/null || true"
  exit 0
fi

# --- 1) Backup remote ---
log "Creating remote backup under ~/.canswim/backups/"
"${RSH[@]}" "$REMOTE" bash -s <<'EOS'
set -euo pipefail
HOME_DATA="${HOME}/.canswim"
STAMP=$(date +%Y%m%d-%H%M%S)
BAK="${HOME_DATA}/backups/data-${STAMP}"
mkdir -p "${HOME_DATA}/backups"
# hardlink-friendly copy of tree (fast, space-efficient where FS allows)
if command -v rsync >/dev/null; then
  rsync -a --info=stats1 "${HOME_DATA}/data/" "${BAK}/"
else
  cp -a "${HOME_DATA}/data" "${BAK}"
fi
echo "BACKUP=${BAK}"
du -sh "${BAK}"
EOS

# --- 2) rsync forecasts (additive, no --delete) ---
log "Rsync forecast/ (no --delete — keeps prod-only symbols)"
rsync -az --info=stats1 -e "$RSYNC_SSH" \
  "$LOCAL_DATA/forecast/" \
  "$REMOTE:${REMOTE_DATA}/forecast/"

# --- 3) rsync 3rd-party except price history (merged separately) ---
log "Rsync data-3rd-party (excluding price hist — will merge)"
rsync -az --info=stats1 -e "$RSYNC_SSH" \
  --exclude 'all_stocks_price_hist_1d.parquet' \
  --exclude 'all_stocks_price_hist_1d.csv' \
  --exclude '*.bak*' \
  --exclude 'run_tickers.csv' \
  "$LOCAL_DATA/data-3rd-party/" \
  "$REMOTE:${REMOTE_DATA}/data-3rd-party/"

# Stage local prices for merge
if [[ -f "$LOCAL_DATA/data-3rd-party/all_stocks_price_hist_1d.parquet" ]]; then
  log "Stage local price parquet for merge"
  rsync -az -e "$RSYNC_SSH" \
    "$LOCAL_DATA/data-3rd-party/all_stocks_price_hist_1d.parquet" \
    "$REMOTE:${REMOTE_DATA}/data-3rd-party/_incoming_prices_local.parquet"
fi

# --- 4) Merge prices on remote (union; keep latest row per Symbol,Date) ---
log "Merge price parquet on remote (prod ∪ local)"
"${RSH[@]}" "$REMOTE" bash -s <<'EOS'
set -euo pipefail
D="${HOME}/.canswim/data/data-3rd-party"
PROD="${D}/all_stocks_price_hist_1d.parquet"
INCOMING="${D}/_incoming_prices_local.parquet"
[[ -f "$INCOMING" ]] || { echo "No incoming prices; skip merge"; exit 0; }
python3 - <<'PY'
import os
from pathlib import Path
import pandas as pd

d = Path(os.path.expanduser("~/.canswim/data/data-3rd-party"))
prod = d / "all_stocks_price_hist_1d.parquet"
inc = d / "_incoming_prices_local.parquet"
frames = []
for p in (prod, inc):
    if p.is_file():
        df = pd.read_parquet(p)
        frames.append(df)
        print(f"loaded {p.name}: shape={df.shape} index={df.index.names}")
if not frames:
    raise SystemExit("no price frames")
merged = pd.concat(frames, axis=0)
# Normalize MultiIndex (Symbol, Date) if present
if isinstance(merged.index, pd.MultiIndex):
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
else:
    # flat columns
    sym = "Symbol" if "Symbol" in merged.columns else "symbol"
    dt = "Date" if "Date" in merged.columns else "date"
    merged = merged.drop_duplicates(subset=[sym, dt], keep="last")
    merged = merged.sort_values([sym, dt])
out = d / "all_stocks_price_hist_1d.parquet"
merged.to_parquet(out)
print(f"wrote {out}: shape={merged.shape}")
nsym = (
    merged.index.get_level_values("Symbol").nunique()
    if isinstance(merged.index, pd.MultiIndex)
    else merged[sym].nunique()
)
print(f"unique symbols≈{nsym}")
inc.unlink(missing_ok=True)
PY
EOS

# --- 5) Brief service window: stop → rebuild DuckDB → start ---
# Mixed forecast schemas (time vs date) need a resilient rebuild helper.
REBUILD_HELPER="${ROOT}/scripts/rebuild_search_db.py"
if [[ ! -f "$REBUILD_HELPER" ]]; then
  # fallback: copy from /tmp path used during ops if not yet committed
  REBUILD_HELPER="/tmp/canswim_rebuild_search_db.py"
fi
[[ -f "$REBUILD_HELPER" ]] || die "missing rebuild helper scripts/rebuild_search_db.py"
log "Upload rebuild helper + stop services for DuckDB rebuild"
rsync -az -e "$RSYNC_SSH" "$REBUILD_HELPER" "$REMOTE:/tmp/canswim_rebuild_search_db.py"

"${RSH[@]}" "$REMOTE" bash -s <<'EOS'
set -euo pipefail
systemctl --user stop canswim-dashboard.service canswim-mcp.service || true
for i in 1 2 3 4 5 6 7 8 9 10; do
  pgrep -f 'python3 -m canswim (dashboard|mcp)' >/dev/null 2>&1 || break
  sleep 1
done
DATA="${HOME}/.canswim/data"
STAMP=$(date +%Y%m%d-%H%M%S)
if [[ -f "${DATA}/canswim_local.duckdb" ]]; then
  mv -f "${DATA}/canswim_local.duckdb" "${DATA}/canswim_local.duckdb.bak-${STAMP}"
fi
export data_dir="${HOME}/.canswim/data"
export db_file="canswim_local.duckdb"
export PYTHONPATH="${HOME}/canswim/src${PYTHONPATH:+:$PYTHONPATH}"
/usr/bin/python3 /tmp/canswim_rebuild_search_db.py
systemctl --user reset-failed canswim-dashboard.service 2>/dev/null || true
systemctl --user start canswim-mcp.service
systemctl --user start canswim-dashboard.service
sleep 3
systemctl --user is-active canswim-dashboard.service canswim-mcp.service
echo "DONE services restored"
EOS

log "Sync complete. Prod-only forecasts kept; local forecasts/prices merged; DuckDB rebuilt."
log "If anything looks wrong: restore from ~/.canswim/backups/data-* on the host."
