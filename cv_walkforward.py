"""
Time-series cross-validation walk-forward optimizer.

For each OOS window of 6 months:
  - Evaluates all parameter combos across 3 IS folds (12 months each)
  - Selects the combo with the best average score across all folds
  - Tests the selected combo on the OOS window (never seen during selection)

Score: expectancy_pips * sqrt(trades)  — penalizes high-variance low-sample combos

Usage:
    python cv_walkforward.py --symbol XRPUSDT --csv XRPUSDT_H4_full.csv --risk 0.05
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dual_direction_backtest import (
    BacktestConfig,
    backtest,
    load_ohlc_csv,
    pip_size,
    summarize,
)

GRID = {
    "atr_quantile":        [0.20, 0.30, 0.40],
    "max_range_atr":       [2.5, 3.0, 3.5],
    "breakout_buffer_atr": [0.0, 0.1],
    "stop_atr":            [1.5, 2.0],
    "take_profit_r":       [3.0, 4.0],
}


def score_combo(df_slice: pd.DataFrame, params: dict, symbol: str) -> float | None:
    cfg = BacktestConfig(
        symbol=symbol,
        spread_pips=5,
        slippage_per_side_pips=2,
        compression_lookback_bars=360,
        **params,
    )
    try:
        trades = backtest(df_slice, cfg)
        s = summarize(trades)
        if s.empty:
            return None
        row = s[s["strategy"] == "BREAKOUT"].iloc[0]
        if row["trades"] < 5:
            return None
        return float(row["expectancy_pips"]) * np.sqrt(float(row["trades"]))
    except Exception:
        return None


def build_windows(df: pd.DataFrame):
    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))

    start = max(df["timestamp"].min().normalize(), pd.Timestamp("2019-01-01"))
    windows = []
    while True:
        is_splits = [
            (str((start + pd.DateOffset(months=m * 6)).date()),
             str((start + pd.DateOffset(months=m * 6 + 12)).date()))
            for m in range(3)
        ]
        oos_start = start + pd.DateOffset(months=18)
        oos_end = oos_start + pd.DateOffset(months=6)
        if oos_end > df["timestamp"].max():
            break
        windows.append((is_splits, str(oos_start.date()), str(oos_end.date())))
        start += pd.DateOffset(months=6)

    return windows, keys, combos


def run(symbol: str, df: pd.DataFrame, risk_pct: float, initial_capital: float):
    PS = pip_size(symbol)
    windows, keys, combos = build_windows(df)

    print(f"\n{len(combos)} combos | {len(windows)} OOS windows | risk={risk_pct:.0%}/trade\n")
    print(f"{'OOS Window':<25} {'Trades':>7} {'Win%':>6} {'Net $':>9} {'Capital':>10} {'':>3}")
    print("-" * 60)

    capital = initial_capital
    profitable = total = 0
    all_records = []

    for is_splits, oos_start, oos_end in windows:
        # Cross-validation: score each combo on all 3 IS folds
        scores: dict[tuple, float] = {}
        for combo in combos:
            params = dict(zip(keys, combo))
            fold_scores = []
            for is_start, is_end in is_splits:
                df_is = df[(df["timestamp"] >= is_start) & (df["timestamp"] < is_end)].reset_index(drop=True)
                if len(df_is) < 300:
                    continue
                sc = score_combo(df_is, params, symbol)
                if sc is not None:
                    fold_scores.append(sc)
            if len(fold_scores) == 3:
                scores[combo] = float(np.mean(fold_scores))

        if not scores:
            continue

        best_params = dict(zip(keys, max(scores, key=scores.get)))

        # OOS evaluation
        df_test = df[(df["timestamp"] >= oos_start) & (df["timestamp"] < oos_end)].reset_index(drop=True)
        if len(df_test) < 30:
            continue

        cfg = BacktestConfig(
            symbol=symbol, spread_pips=5, slippage_per_side_pips=2,
            compression_lookback_bars=360, **best_params,
        )
        trades = backtest(df_test, cfg)
        oos = trades[(trades["strategy"] == "BREAKOUT") & (trades["direction"] != "NONE")].copy()
        oos["stop_dollar"] = (oos["stop_initial"] - oos["breakout_price"]).abs()
        oos = oos[oos["stop_dollar"] > 0]

        window_net = 0.0
        pnls = []
        for _, row in oos.iterrows():
            pos_size = (capital * risk_pct) / row["stop_dollar"]
            pnl = row["net_pips"] * PS * pos_size
            capital += pnl
            window_net += pnl
            pnls.append(pnl)
            all_records.append({"oos_start": oos_start, "oos_end": oos_end,
                                 "entry_time": row["entry_time"], "pnl": pnl,
                                 "capital": capital, **best_params})
            if capital <= 0:
                break

        total += 1
        if window_net > 0:
            profitable += 1

        wr = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
        icon = "✅" if window_net > 0 else "❌"
        print(f"{oos_start}->{oos_end}  {len(oos):>7} {wr:>6.1%} {window_net:>9,.0f} ${capital:>9,.0f} {icon}")

    ret = (capital - initial_capital) / initial_capital
    cap_vals = [initial_capital] + [r["capital"] for r in all_records]
    eq = pd.Series(cap_vals)
    peak = eq.cummax()
    max_dd = float(((eq - peak) / peak).min())

    print(f"\n{'=' * 60}")
    print(f"Capital initial  : ${initial_capital:,.0f}")
    print(f"Capital final    : ${capital:,.2f}")
    print(f"Return           : {ret:+.1%}")
    print(f"Max drawdown     : {max_dd:.1%}")
    print(f"Fenêtres profit. : {profitable}/{total}")

    return pd.DataFrame(all_records)


def parse_args():
    p = argparse.ArgumentParser(description="CV walk-forward optimizer for dual-direction backtest.")
    p.add_argument("--csv", required=True, help="Path to H4 OHLC CSV.")
    p.add_argument("--symbol", required=True, help="Asset symbol (e.g. XRPUSDT, BTCUSDT).")
    p.add_argument("--risk", type=float, default=0.05, help="Risk per trade as fraction (default: 0.05).")
    p.add_argument("--capital", type=float, default=10_000, help="Initial capital (default: 10000).")
    p.add_argument("--out", type=str, default=None, help="Output CSV for trade log.")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_ohlc_csv(args.csv)
    print(f"Loaded {len(df):,} bars | {df.timestamp.min().date()} -> {df.timestamp.max().date()}")

    records = run(args.symbol.upper(), df, args.risk, args.capital)

    if args.out and not records.empty:
        records.to_csv(args.out, index=False)
        print(f"\nTrade log saved to {args.out}")


if __name__ == "__main__":
    main()
