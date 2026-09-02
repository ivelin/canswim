# TimesFM-3 vs TIDE — paired IBD50 backtest

Research / non-production. Scored from existing DuckDB only (`canswim_local.duckdb`). No new training.

## Window
- Universe: live IBD50 (`~/.canswim/data/data-3rd-party/IBD50.csv`) minus `ITCI` (no TimesFM-3 backtest rows)
- Shared origins: **11** monthly starts, **2025-09-29 through 2026-08-03**
- Paired symbol-origin rows: **n = 539** each
- TimesFM-3 backtests start on 2025-09-29. TIDE alone goes back to 2025-02-11 (94 origins); **0 paired rows before 2025-09-29**
- Live origin **2026-08-31** has TimesFM-3 forecasts (50 symbols) but no backtest yet (42-day horizon still open)

Stored native metric is `backtest_error.mal_error` (mean absolute log error per symbol-origin). MAE / RMSE / MAPE / directional hit rate are the q50 (`close_quantile_0.5`) vs actual close on those same pairs.

## Stored MAL (n=539)
| Model | n | MAL mean | MAL median |
|---|---:|---:|---:|
| TIDE | 539 | 0.085 | 0.043 |
| TimesFM-3 | 539 | 0.050 | 0.041 |

## Point forecast (q50 vs close)
| Model | n (day rows) | MAE | RMSE | MAPE | dir-hit |
|---|---:|---:|---:|---:|---:|
| TIDE | 21364 | 21.91 | 49.08 | 12.5% | 51.2% |
| TimesFM-3 | 21511 | 22.76 | 53.77 | 11.9% | 49.1% |

## Horizon split (MAPE / dir-hit)
| Horizon | TIDE | TimesFM-3 |
|---|---|---|
| 1–7d | 5.8% / 49.6% | 4.7% / 39.3% |
| 8–21d | 10.6% / 49.6% | 9.3% / 50.3% |
| 22–42d | 15.5% / 52.3% | 14.6% / 50.9% |

## Takeaway
TimesFM-3 is better on canswim MAL and on MAPE. TIDE is slightly better in dollars (MAE/RMSE). Direction is a coin flip for both. Neither model is a buy/sell edge. Keep as research charts, not investment analysis.
