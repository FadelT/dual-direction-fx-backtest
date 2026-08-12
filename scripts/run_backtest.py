"""
Run a single backtest and save results.

Usage:
    python scripts/run_backtest.py --csv data/crypto/H4/BTCUSDT_H4_full.csv --symbol BTCUSDT
    python scripts/run_backtest.py --synthetic --symbol EURUSD
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest import (
    BacktestConfig,
    backtest,
    generate_synthetic_fx,
    load_ohlc_csv,
    summarize,
    summarize_by_session,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compression -> breakout backtest: BREAKOUT vs hedge variants."
    )
    p.add_argument("--csv", type=str, default=None, help="OHLC CSV path.")
    p.add_argument("--symbol", type=str, default="EURUSD")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--synthetic", action="store_true", help="Run on generated M15 data.")

    p.add_argument("--spread-pips", type=float, default=0.8)
    p.add_argument("--slippage-pips", type=float, default=0.05)
    p.add_argument("--commission-side-pips", type=float, default=0.0)

    p.add_argument("--atr-quantile", type=float, default=0.30)
    p.add_argument("--max-range-atr", type=float, default=3.0)
    p.add_argument("--buffer-atr", type=float, default=0.10)
    p.add_argument("--stop-atr", type=float, default=1.5)
    p.add_argument("--tp-r", type=float, default=3.0)
    p.add_argument("--trail-atr", type=float, default=2.0)
    p.add_argument("--max-wait-bars", type=int, default=24)
    p.add_argument("--max-hold-bars", type=int, default=96)

    return p.parse_args()


def main():
    args = parse_args()

    cfg = BacktestConfig(
        symbol=args.symbol.upper(),
        spread_pips=args.spread_pips,
        slippage_per_side_pips=args.slippage_pips,
        commission_per_side_pips=args.commission_side_pips,
        atr_quantile=args.atr_quantile,
        max_range_atr=args.max_range_atr,
        breakout_buffer_atr=args.buffer_atr,
        stop_atr=args.stop_atr,
        take_profit_r=args.tp_r,
        trailing_atr=args.trail_atr,
        max_wait_bars=args.max_wait_bars,
        max_hold_bars=args.max_hold_bars,
    )

    if args.synthetic:
        df = generate_synthetic_fx()
        source = "synthetic"
    elif args.csv:
        df = load_ohlc_csv(args.csv)
        source = args.csv
    else:
        raise SystemExit("Use either --csv PATH or --synthetic")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = backtest(df, cfg)
    summary = summarize(trades)
    by_session = summarize_by_session(trades)

    trades.to_csv(out_dir / "trades.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    by_session.to_csv(out_dir / "summary_by_session.csv", index=False)

    print(f"\nSource: {source}")
    print(f"Bars:   {len(df):,}")
    print(f"Symbol: {cfg.symbol}")
    print(f"Trades rows: {len(trades):,}")

    if summary.empty:
        print("\nNo setups/trades found with these parameters.")
    else:
        show_cols = [
            "strategy", "trades", "win_rate", "net_pips",
            "expectancy_pips", "profit_factor", "max_drawdown_pips",
            "total_cost_pips", "delta_vs_breakout_pips",
        ]
        print("\n=== SUMMARY ===")
        print(summary[show_cols].to_string(index=False))

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
