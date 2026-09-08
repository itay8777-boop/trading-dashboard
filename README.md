# Trading Dashboard

A local trading-strategy tester and live chart for Interactive Brokers, built around a
TradingView-style UI. Two front ends share the same data/backtest layer:

- **`dashboard/`** (Dash + TradingView [Lightweight Charts](https://github.com/tradingview/lightweight-charts)) —
  the actively developed app: live candles, technical indicators, a Pine-Editor-style
  strategy panel with vectorized backtest engines, and a full Strategy Tester
  (Key Stats, Performance Summary, List of Trades).
- **`app.py`** (Streamlit) — an earlier full TradingView-page replica (toolbar, drawing
  tools, watchlists chrome) sharing the same `ib_data.py` fetch/cache layer.

## Requirements

- Python 3.12+
- Interactive Brokers TWS or IB Gateway running locally with API access enabled
  (*Configuration → API → Settings → Enable ActiveX and Socket Clients*)
- (Optional) a [Databento](https://databento.com) `.dbn` export if you want to seed
  history older than what IB exposes for expired futures contracts

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running

**Dash dashboard** (primary app — MNQ futures, live candles, indicators, strategy
backtesting):

```bash
.venv/bin/python -m dashboard.app
```

Then open <http://127.0.0.1:8050>. See [`dashboard/README.md`](dashboard/README.md) for
the module layout and how to extend it (new indicators, timeframes, etc.).

**Streamlit app** (equities/ETFs, full TradingView-page replica):

```bash
.venv/bin/streamlit run app.py
```

Both connect to TWS/IB Gateway on `127.0.0.1:7497` (paper) by default — see `ib_data.py`
for connection settings.

## Project layout

| Path | Role |
|---|---|
| `dashboard/` | Dash app — the active app. See its own README for details. |
| `app.py` | Streamlit app — TradingView-page replica for equities/ETFs. |
| `ib_data.py` | Shared IB data-fetch layer (pacing, chunking, parquet cache) used by both front ends. |
| `chart.py` / `ui_components.py` | Plotly chart building and TradingView-styled HTML components, shared by both front ends. |
| `backtest_runner.py` | Standalone CLI entry point for running a backtest without either UI. |
| `import_databento.py` | One-time import of a Databento `.dbn` export into the local parquet cache (for futures history older than IB exposes). |
| `scripts/build_mnq_history.py` | Helper script for building deep MNQ futures history. |

## Data caching

Historical bars are cached to Parquet under `data_cache/` (per symbol/timeframe/session),
and raw Databento exports live under `databento_data/`. Both are gitignored — they're
regenerated automatically by fetching from IB (and are large: order of 100MB+).

## Notes

- This is a local development/research tool, not a production trading system. It does
  not place live orders — IB connections are read-only (market data + historical bars).
- `strategies.json` (user-saved Pine-Editor strategies) is created at runtime and
  gitignored; the dashboard's built-in strategies are seeded in code, not that file.
