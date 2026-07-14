# Production-style user service deploy

Stand up canswim as **user-level systemd** units on a Linux host with:

| Surface | Exposure | Auth |
|---------|----------|------|
| **Gradio dashboard** | **Private** — Tailscale (or other VPN/LAN) only | Network isolation; optional app password via env |
| **MCP** | **Public** via reverse proxy + Funnel (or similar) | **API key** at the gateway (`?apikey=` / owner key whitelist) |

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

This page is the operator guide. CLI/MCP flags: [cli.md](cli.md), [mcp.md](mcp.md). Data layout: [data_store.md](data_store.md).

## Architecture (recommended)

```text
  Tailscale peers ──HTTP──► Gradio :7860          (dashboard; not on public Funnel)
                              ▲
                              │ shared data_dir (parquet + DuckDB)
                              │
  Internet ──Funnel──► Caddy :8080 ──apikey──► FastMCP 127.0.0.1:3472/mcp
```

**Separate processes:**

1. **Dashboard** — `python -m canswim dashboard --same_data True` (bind `0.0.0.0:7860` or Tailscale IP only).
2. **MCP** — `python -m canswim mcp --http --host 127.0.0.1 --port 3472` (Streamable HTTP / `streamable-http` transport; localhost only; gateway is the public edge).

Do **not** put the Gradio UI on the public Funnel. Do **not** expose the MCP bind port directly on the public internet without the gateway apikey check.

## Prerequisites

- Linux user account with **systemd --user** (and preferably `loginctl enable-linger $USER` so services survive logout).
- Checkout of canswim (or `pip install canswim`) and a Python env with dependencies (torch/darts for forecast, Gradio for UI, `mcp` for FastMCP).
- Shared data directory (example: `~/.canswim/data`) with `data-3rd-party/` parquet + `forecast/` partitions.
- Model checkpoint available for forecast (`canswim_model.pt` in the process cwd, or HF download when `hfhub_sync=True`).
- Secrets in **`~/.env`** (not committed): e.g. `FMP_API_KEY`, `CANSWIM_MCP_KEY`, optional `HF_TOKEN` / dashboard password.
- Optional public MCP: reverse proxy (Caddy recommended) and a single Tailscale Funnel (or TLS terminator) to the proxy port.

## 1. Layout and env

```bash
# Example paths — adjust to your host
export CANSWIM_DIR=~/canswim          # git checkout
export CANSWIM_HOME=~/.canswim
export data_dir=$CANSWIM_HOME/data
export db_file=canswim_local.duckdb
export hfhub_sync=False

mkdir -p "$data_dir/data-3rd-party" "$data_dir/forecast"
# optional: symlink checkout data/ → shared data so relative paths resolve
# ln -sfn "$data_dir" "$CANSWIM_DIR/data"
```

**`~/.env` (examples — use strong random values):**

```bash
FMP_API_KEY=...                 # market data (name may also be FMP_API_Key on some hosts)
CANSWIM_MCP_KEY=canswim_...     # only key accepted by the gateway for /mcp/canswim*
# optional UI password if your Gradio build honors it:
# DASHBOARD_SECRET_KEY=...
```

Wrappers should map legacy names if needed:

```bash
export FMP_API_KEY="${FMP_API_KEY:-${FMP_API_Key:-}}"
```

## 2. Dashboard unit (private GUI)

**Wrapper** `~/.canswim-dashboard/run.sh` (illustrative):

```bash
#!/usr/bin/env bash
set -euo pipefail
set -a; source "$HOME/.env" 2>/dev/null || true; set +a
export FMP_API_KEY="${FMP_API_KEY:-${FMP_API_Key:-}}"
: "${CANSWIM_DIR:=$HOME/canswim}"
: "${CANSWIM_HOME:=$HOME/.canswim}"
: "${CANSWIM_DASHBOARD_HOST:=0.0.0.0}"
: "${CANSWIM_DASHBOARD_PORT:=7860}"
cd "$CANSWIM_DIR"
export PYTHONPATH="${CANSWIM_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
export GRADIO_SERVER_NAME="$CANSWIM_DASHBOARD_HOST"
export GRADIO_SERVER_PORT="$CANSWIM_DASHBOARD_PORT"
export data_dir="${CANSWIM_HOME}/data"
export db_file=canswim_local.duckdb
export hfhub_sync=False
# Conservative threads on shared hosts
export OMP_NUM_THREADS=2 TORCH_NUM_THREADS=2
exec env data-3rd-party=data-3rd-party python -m canswim dashboard --same_data True
```

**Unit** `~/.config/systemd/user/canswim-dashboard.service`:

```ini
[Unit]
Description=CANSWIM Gradio Dashboard (Tailscale UI)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/canswim
EnvironmentFile=-%h/.env
Environment=CANSWIM_DASHBOARD_HOST=0.0.0.0
Environment=CANSWIM_DASHBOARD_PORT=7860
ExecStart=%h/.canswim-dashboard/run.sh
Restart=always
RestartSec=10
MemoryMax=4G
CPUQuota=150%
Nice=10

[Install]
WantedBy=default.target
```

```bash
chmod +x ~/.canswim-dashboard/run.sh
systemctl --user daemon-reload
systemctl --user enable --now canswim-dashboard
systemctl --user status canswim-dashboard --no-pager
```

**Access (private):**

```text
http://<tailscale-ip>:7860/
http://<magicdns-name>:7860/
```

Confirm the UI is **not** listed on public Funnel routes. Prefer binding only the Tailscale IP if the host also has a public interface.

## 3. MCP unit (localhost Streamable HTTP)

```bash
python -m canswim mcp --http --host 127.0.0.1 --port 3472
```

**Wrapper** `~/.canswim-dashboard/run-mcp.sh` (illustrative):

```bash
#!/usr/bin/env bash
set -euo pipefail
set -a; source "$HOME/.env" 2>/dev/null || true; set +a
export FMP_API_KEY="${FMP_API_KEY:-${FMP_API_Key:-}}"
: "${CANSWIM_DIR:=$HOME/canswim}"
: "${CANSWIM_HOME:=$HOME/.canswim}"
: "${CANSWIM_MCP_HOST:=127.0.0.1}"
: "${CANSWIM_MCP_PORT:=3472}"
cd "$CANSWIM_DIR"
export PYTHONPATH="${CANSWIM_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
export data_dir="${CANSWIM_HOME}/data"
export db_file=canswim_local.duckdb
export hfhub_sync=False
# Leave MCP_ALLOW_RUNS unset for read-only public tools
exec env data-3rd-party=data-3rd-party python -m canswim mcp \
  --http --host "$CANSWIM_MCP_HOST" --port "$CANSWIM_MCP_PORT"
```

**Unit** `~/.config/systemd/user/canswim-mcp.service`:

```ini
[Unit]
Description=CANSWIM FastMCP (Streamable HTTP for gateway)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/canswim
EnvironmentFile=-%h/.env
Environment=CANSWIM_MCP_HOST=127.0.0.1
Environment=CANSWIM_MCP_PORT=3472
ExecStart=%h/.canswim-dashboard/run-mcp.sh
Restart=always
RestartSec=5
MemoryMax=2G
Nice=10

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now canswim-mcp
curl -sS -X POST "http://127.0.0.1:3472/mcp" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"local","version":"1"}}}'
```

FastMCP serves the protocol at **`/mcp`** on that port. Details: [mcp.md](mcp.md).

## 4. Public MCP via gateway + apikey

Use one edge (e.g. **Tailscale Funnel → Caddy :8080**) for all MCPs. Pattern:

1. Public path prefix: `/mcp/canswim*`
2. Require `?apikey=` equal to `CANSWIM_MCP_KEY` from the host env (owner whitelist).
3. On success: strip prefix and `reverse_proxy 127.0.0.1:3472`.
4. Missing key → **401**; wrong key → **403**.

**Connector URL shape:**

```text
https://<funnel-host>/mcp/canswim/mcp?apikey=<CANSWIM_MCP_KEY>
```

**Auth matrix smoke:**

```bash
source ~/.env
BASE="https://<funnel-host>/mcp/canswim/mcp"
# no key → 401
curl -sS -o /dev/null -w '%{http_code}\n' -X POST "$BASE" \
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}'
# wrong key → 403
curl -sS -o /dev/null -w '%{http_code}\n' -X POST "$BASE?apikey=wrong" \
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}'
# correct key → 200
curl -sS -o /dev/null -w '%{http_code}\n' -X POST "$BASE?apikey=$CANSWIM_MCP_KEY" \
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}'
```

Keep **`MCP_ALLOW_RUNS` unset** on the public MCP process unless you intentionally allow remote gather/forecast (heavy; prefer CLI on the host for data population).

## 5. Data population (after services exist)

```bash
cd "$CANSWIM_DIR"
export PYTHONPATH="$CANSWIM_DIR/src" data_dir="$CANSWIM_HOME/data" hfhub_sync=False
# example: checked-in IBD50 list
TICKERS=$(tail -n +2 symbol_lists/IBD50.csv | paste -sd, -)
python -m canswim gatherdata --tickers "$TICKERS"
python -m canswim forecast --tickers "$TICKERS"   # or a smaller subset first
# rebuild search DB so Charts/MCP see new symbols (dashboard --same_data False once, or Run-tab rebuild)
```

Search DB semantics: [data_store.md](data_store.md). Forecast needs complete covariates (industry funds / broad market); yfinance must not use a custom session that breaks Yahoo (default canswim leaves session to yfinance).

## 6. Operations

```bash
systemctl --user restart canswim-dashboard canswim-mcp
systemctl --user status canswim-dashboard canswim-mcp --no-pager
journalctl --user -u canswim-dashboard -u canswim-mcp --since "5 min ago" -q
```

Resource limits (MemoryMax / CPUQuota / thread env) are strongly recommended on shared hosts.

## Security checklist

- [ ] Dashboard **not** on public Funnel / open internet without VPN.
- [ ] MCP process binds **127.0.0.1** only.
- [ ] Gateway enforces **owner** `CANSWIM_MCP_KEY` (not an open `apikey=*` alone).
- [ ] Secrets only in `~/.env` (mode 600); never in git.
- [ ] Public MCP stays **read-only** unless you deliberately set `MCP_ALLOW_RUNS=1`.
- [ ] Unit `ReadWritePaths` limited to data dirs when using hardening options.

## Related

- [mcp.md](mcp.md) — tools and HTTP flags  
- [cli.md](cli.md) — tasks and env  
- [data_store.md](data_store.md) — parquet vs DuckDB  
- [run_triggers.md](run_triggers.md) — gather/forecast policy  
