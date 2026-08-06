#!/usr/bin/env python3
"""Delete forecast hive partitions for symbols lacking real fundamentals.

Hard rule: forecasts/backtests require real local earnings + key metrics +
analyst estimates (not zero-filled placeholders). Run after policy change or
when FMP fund endpoints failed during a bulk refresh.

Usage (prod home layout)::

    data_dir=$HOME/.canswim/data \\
      python scripts/purge_forecasts_without_fundamentals.py [--dry-run]

Then rebuild the DuckDB search cache::

    data_dir=$HOME/.canswim/data python scripts/rebuild_search_db.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Allow running from repo root without install
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from canswim.eligibility import (  # noqa: E402
    fundamentals_are_ready,
    load_fundamentals_symbol_sets,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        default=os.getenv("data_dir", str(Path.home() / ".canswim" / "data")),
        help="canswim data_dir (default: env data_dir or ~/.canswim/data)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List symbols that would be purged; do not delete",
    )
    args = p.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    forecast_root = data_dir / "forecast"
    if not forecast_root.is_dir():
        print(f"No forecast dir at {forecast_root}", file=sys.stderr)
        return 1

    fund_sets = load_fundamentals_symbol_sets(data_dir)
    dirs = sorted(forecast_root.glob("symbol=*"))
    purge: list[str] = []
    keep: list[str] = []
    for d in dirs:
        sym = d.name.split("=", 1)[-1].strip().upper()
        ok, reason = fundamentals_are_ready(sym, fund_sets=fund_sets)
        if ok:
            keep.append(sym)
        else:
            purge.append(sym)
            print(f"{'WOULD PURGE' if args.dry_run else 'PURGE'} {sym}: {reason}")
            if not args.dry_run:
                shutil.rmtree(d)

    print(
        f"done: keep={len(keep)} purge={len(purge)} dry_run={args.dry_run} "
        f"data_dir={data_dir}"
    )
    if purge and not args.dry_run:
        print(
            "Next: rebuild search DB, e.g.\n"
            f"  data_dir={data_dir} python scripts/rebuild_search_db.py"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
