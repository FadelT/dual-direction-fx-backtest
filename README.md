# Dual-Direction FX & Crypto Backtest

A research engine testing the idea:

**compression → arm both directions → breakout reveals direction → kill losing leg → manage winner**

against the cleaner benchmark:

**wait for breakout → enter only the revealed direction**

→ See [REPORT.md](REPORT.md) for the full analysis and results.

---

## Strategies

1. `BREAKOUT` — no position during compression, enters 1x at breakout
2. `HEDGE_1x1` — long 1x + short 1x at setup, close losing leg at breakout
3. `HALF_HEDGE_ADD` — long 0.5x + short 0.5x, close loser, add 0.5x to winner
4. `OCO` — semantic benchmark for buy-stop/sell-stop, same economics as BREAKOUT

**Key finding**: hedge variants never beat BREAKOUT after transaction costs on any asset or timeframe tested.

---

## Install

```bash
pip install -r requirements.txt
```

---

## Usage

### Smoke test (synthetic data)

```bash
python dual_direction_backtest.py --synthetic --symbol EURUSD --out results_synthetic
```

### FX pairs (M15)

```bash
python dual_direction_backtest.py \
  --csv EURUSD_M15.csv \
  --symbol EURUSD \
  --spread-pips 0.8 \
  --slippage-pips 0.05 \
  --out results_eurusd
```

### Crypto H4 — recommended setup

Pip size is inferred automatically: `1.0` for BTC/ETH, `0.0001` for XRP/SOL/etc.

```bash
python dual_direction_backtest.py \
  --csv XRPUSDT_H4.csv \
  --symbol XRPUSDT \
  --spread-pips 5 \
  --slippage-pips 2 \
  --atr-quantile 0.30 \
  --max-range-atr 3.5 \
  --buffer-atr 0.0 \
  --stop-atr 2.0 \
  --tp-r 3.0 \
  --out results_xrp
```

### Download Binance H4 data (free, no API key)

```python
import requests, time, pandas as pd
from datetime import datetime

def fetch_binance_h4(symbol, start_str='2018-01-01'):
    url = 'https://api.binance.com/api/v3/klines'
    start_ms = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ms   = int(datetime.now().timestamp() * 1000)
    rows = []
    while start_ms < end_ms:
        r = requests.get(url, params={
            'symbol': symbol, 'interval': '4h',
            'startTime': start_ms, 'limit': 1000
        }, timeout=15)
        data = r.json()
        if not data or isinstance(data, dict): break
        rows.extend(data)
        start_ms = data[-1][0] + 1
        time.sleep(0.1)
    df = pd.DataFrame(rows, columns=[
        'timestamp','open','high','low','close','volume',
        'close_time','qv','trades','tbb','tbq','ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for c in ['open','high','low','close']: df[c] = df[c].astype(float)
    return df[['timestamp','open','high','low','close']]

# Examples
fetch_binance_h4('XRPUSDT').to_csv('XRPUSDT_H4.csv', index=False)
fetch_binance_h4('BTCUSDT').to_csv('BTCUSDT_H4.csv', index=False)
fetch_binance_h4('AVAXUSDT').to_csv('AVAXUSDT_H4.csv', index=False)
```

---

## CV Walk-Forward results (H4, 5% risk/trade)

Best performing assets with time-series cross-validation parameter selection:

| Asset | OOS Period | Trades | Win% | Return | Max DD |
|---|---|---|---|---|---|
| XRPUSDT | 2020-07 → 2026-08 | 131 | 55.0% | **+957%** | -27.3% |
| BTCUSDT | 2020-07 → 2026-08 | 148 | 44.6% | **+373%** | -38.3% |
| AVAXUSDT | 2022-03 → 2026-08 | 94 | 56.4% | **+252%** | -27.5% |

vs benchmarks over same period:
- XRP buy & hold: **+477%** (strategy: +957%)
- BTC buy & hold: **+892%** (strategy: +373%)
- S&P 500 buy & hold: **+148%**

---

## Parameters worth sweeping first

```
--atr-quantile     0.15, 0.20, 0.30, 0.40
--max-range-atr    2.0, 2.5, 3.0, 3.5
--buffer-atr       0.0, 0.1, 0.2
--stop-atr         1.0, 1.5, 2.0
--tp-r             2, 3, 4
```

**Important**: use time-series cross-validation (3 IS folds) to select parameters — do not optimize on the same window you test on.

---

## Outputs

- `trades.csv` — one row per trade per strategy
- `summary.csv` — aggregated metrics per strategy
- `summary_by_session.csv` — breakdown by trading session

Key column: `delta_vs_breakout_pips` — if negative for hedge variants, hedging lost after costs.

---

## CSV format

```csv
timestamp,open,high,low,close
2025-01-02 07:00:00,1.0351,1.0357,1.0349,1.0355
```

Also supports `Date,Time,Open,High,Low,Close` and `datetime,Open,High,Low,Close`.

---

## Limitations (V1)

- Bar-based, not tick-based
- Constant spread (no spread widening during events)
- No swap/overnight financing
- No session filter
- No trend filter (Ichimoku, EMA)
- No walk-forward optimizer built-in
- Same exit engine across all strategies
