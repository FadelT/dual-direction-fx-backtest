
# Dual-Direction FX Backtest

A deliberately simple research engine for testing the idea:

**compression -> arm both directions -> breakout reveals direction -> kill losing leg -> manage winner**

against the cleaner benchmark:

**wait for breakout -> enter only the revealed direction**

## Strategies

1. `BREAKOUT`
   - no position during compression
   - enters 1x in breakout direction

2. `HEDGE_1x1`
   - at compression setup: long 1x + short 1x
   - at breakout: close losing 1x leg
   - keep winning 1x leg

3. `HALF_HEDGE_ADD`
   - at compression setup: long 0.5x + short 0.5x
   - at breakout: close losing 0.5x
   - add 0.5x to winning side
   - final directional exposure = 1x

4. `OCO`
   - modeled as a one-sided filled breakout entry
   - currently has the same filled-volume economics as `BREAKOUT`
   - useful as a semantic benchmark for a buy-stop/sell-stop approach

## Important expected result

With linear spot-FX arithmetic and identical exit rules, all four approaches have
essentially the same **gross directional PnL** once a breakout occurs.

The hedge variants differ mainly because:
- they trade more volume;
- they pay more spread/slippage/commission;
- they also pay costs when a compression setup never breaks out.

So the first experiment is intentionally trying to falsify the hedge idea.

## CSV format

The loader accepts e.g.

```csv
timestamp,open,high,low,close
2025-01-02 07:00:00,1.0351,1.0357,1.0349,1.0355
...
```

or:

```csv
Date,Time,Open,High,Low,Close
2025-01-02,07:00:00,1.0351,1.0357,1.0349,1.0355
...
```

Use M15 data for the default parameters.

## Install

```bash
pip install -r requirements.txt
```

## Smoke test

```bash
python dual_direction_backtest.py \
  --synthetic \
  --symbol EURUSD \
  --out results_synthetic
```

## Real EURUSD data

```bash
python dual_direction_backtest.py \
  --csv EURUSD_M15.csv \
  --symbol EURUSD \
  --spread-pips 0.8 \
  --slippage-pips 0.05 \
  --out results_eurusd
```

## JPY pair

Pip size is inferred automatically:

```bash
python dual_direction_backtest.py \
  --csv USDJPY_M15.csv \
  --symbol USDJPY \
  --spread-pips 0.9 \
  --out results_usdjpy
```

## Parameters worth sweeping first

Do not optimize everything at once.

Start with:

- `--atr-quantile`: 0.15, 0.20, 0.30, 0.40
- `--max-range-atr`: 2.0, 2.5, 3.0, 3.5
- `--buffer-atr`: 0.0, 0.1, 0.2, 0.3
- `--stop-atr`: 1.0, 1.5, 2.0
- `--tp-r`: 2, 3, 4
- `--trail-atr`: 1.5, 2.0, 2.5

Example:

```bash
python dual_direction_backtest.py \
  --csv EURUSD_M15.csv \
  --symbol EURUSD \
  --atr-quantile 0.20 \
  --max-range-atr 2.5 \
  --buffer-atr 0.20 \
  --stop-atr 1.5 \
  --tp-r 4 \
  --trail-atr 2 \
  --out results_test
```

## Outputs

- `trades.csv`
- `summary.csv`
- `summary_by_session.csv`

The key column is:

`delta_vs_breakout_pips`

If a hedge variant does not beat `BREAKOUT` after costs, the hedge has not earned
its additional complexity.

## Current limitations

This V1 is intentionally strict and simple:

- bar-based, not tick-based;
- assumes timestamps are already in a consistent timezone;
- spread is constant;
- same exit engine across variants;
- no swap;
- no Ichimoku yet;
- no Dow/swing filter yet;
- no H1/H4 zone filter yet;
- no asymmetric long/short sizing yet;
- no walk-forward optimizer yet.

Those should be added only after the baseline is measured.
