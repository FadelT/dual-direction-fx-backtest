
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class BacktestConfig:
    symbol: str = "EURUSD"
    timeframe_minutes: int = 15

    # Setup
    atr_period: int = 14
    compression_lookback_bars: int = 96 * 20   # ~20 trading days on M15
    atr_quantile: float = 0.30
    range_bars: int = 12                        # 3 hours on M15
    max_range_atr: float = 3.0
    breakout_buffer_atr: float = 0.10
    max_wait_bars: int = 24                     # 6 hours on M15

    # Exit
    stop_atr: float = 1.5
    take_profit_r: Optional[float] = 3.0
    trailing_atr: Optional[float] = 2.0
    trailing_activate_r: float = 1.0
    max_hold_bars: int = 96                     # 24 hours on M15

    # Costs (expressed in pips)
    spread_pips: float = 0.8
    commission_per_side_pips: float = 0.0
    slippage_per_side_pips: float = 0.05

    # Execution
    execute_breakout_on_next_open: bool = True


@dataclass
class Trade:
    strategy: str
    symbol: str
    direction: str

    arm_time: pd.Timestamp
    breakout_signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp

    arm_price: float
    breakout_price: float
    exit_price: float

    upper_range: float
    lower_range: float
    atr: float
    atr_threshold: float
    range_atr_ratio: float

    stop_initial: float
    take_profit_initial: Optional[float]

    gross_pips: float
    cost_pips: float
    net_pips: float
    r_multiple_net: float

    mfe_pips: float
    mae_pips: float
    holding_bars: int
    exit_reason: str
    session: str


# ----------------------------
# Data loading
# ----------------------------

def load_ohlc_csv(path: str | Path) -> pd.DataFrame:
    """
    Accepts common OHLC CSV layouts.

    Required logical fields:
        timestamp (or datetime, or date+time), open, high, low, close

    Examples supported:
        timestamp,open,high,low,close
        Date,Time,Open,High,Low,Close
        datetime,Open,High,Low,Close
    """
    path = Path(path)
    df = pd.read_csv(path)

    original = list(df.columns)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Timestamp detection
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    elif "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"], errors="coerce", utc=False)
    elif "date" in df.columns and "time" in df.columns:
        ts = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce",
            utc=False,
        )
    elif "date" in df.columns:
        ts = pd.to_datetime(df["date"], errors="coerce", utc=False)
    else:
        raise ValueError(
            f"Could not detect timestamp column. Columns found: {original}"
        )

    rename_map = {}
    aliases = {
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c", "last"],
    }
    for target, names in aliases.items():
        found = next((x for x in names if x in df.columns), None)
        if found is None:
            raise ValueError(f"Missing required OHLC field '{target}'. Columns: {original}")
        rename_map[found] = target

    df = df.rename(columns=rename_map)
    df["timestamp"] = ts

    keep = ["timestamp", "open", "high", "low", "close"]
    df = df[keep].copy()

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna()
          .drop_duplicates(subset=["timestamp"], keep="last")
          .sort_values("timestamp")
          .reset_index(drop=True)
    )

    if len(df) < 500:
        raise ValueError(f"Only {len(df)} valid bars found; provide more history.")

    return df


# ----------------------------
# Indicators / setup
# ----------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    # Wilder ATR == EMA alpha=1/period
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def enrich_features(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    x = df.copy()
    x["atr"] = wilder_atr(x, cfg.atr_period)

    # Past-only rolling ATR threshold; at bar t, all inputs are <= t.
    x["atr_threshold"] = (
        x["atr"]
        .rolling(cfg.compression_lookback_bars, min_periods=cfg.compression_lookback_bars)
        .quantile(cfg.atr_quantile)
    )

    x["range_high"] = x["high"].rolling(cfg.range_bars, min_periods=cfg.range_bars).max()
    x["range_low"] = x["low"].rolling(cfg.range_bars, min_periods=cfg.range_bars).min()
    x["range_width"] = x["range_high"] - x["range_low"]
    x["range_atr_ratio"] = x["range_width"] / x["atr"]

    x["compression"] = (
        (x["atr"] <= x["atr_threshold"])
        & (x["range_atr_ratio"] <= cfg.max_range_atr)
    )

    # A setup is armed only on the FIRST bar of a new compression episode.
    x["compression_start"] = x["compression"] & (~x["compression"].shift(1).fillna(False))
    return x


# ----------------------------
# Helpers
# ----------------------------

def pip_size(symbol: str) -> float:
    s = symbol.upper().replace("/", "")
    if "BTC" in s or "ETH" in s:
        return 1.0   # 1 pip = $1 for crypto
    return 0.01 if "JPY" in s else 0.0001


def px_to_pips(delta_price: float, symbol: str) -> float:
    return delta_price / pip_size(symbol)


def execution_cost_pips(cfg: BacktestConfig, executed_unit_sides: float) -> float:
    """
    Mid-price model:
      each execution side pays:
        half spread + commission + slippage
    """
    per_side = (
        0.5 * cfg.spread_pips
        + cfg.commission_per_side_pips
        + cfg.slippage_per_side_pips
    )
    return executed_unit_sides * per_side


def session_name(ts: pd.Timestamp) -> str:
    # Assumes timestamps are UTC or at least consistently interpreted.
    h = ts.hour
    if 0 <= h < 7:
        return "Asia"
    if 7 <= h < 12:
        return "London_Open"
    if 12 <= h < 16:
        return "London_NY_Overlap"
    if 16 <= h < 21:
        return "NY"
    return "Late_NY"


def strategy_cost_pips(strategy: str, cfg: BacktestConfig) -> float:
    """
    Executed volume in unit-sides:
      BREAKOUT: open 1 + close 1 = 2.0
      OCO:      same filled-volume assumption = 2.0
      HEDGE_1x1:
          open long 1 + open short 1
          close loser 1 + close winner 1 = 4.0
      HALF_HEDGE_ADD:
          open long .5 + open short .5
          close loser .5 + add winner .5
          final close total winner 1.0 = 3.0
    """
    sides = {
        "BREAKOUT": 2.0,
        "OCO": 2.0,
        "HEDGE_1x1": 4.0,
        "HALF_HEDGE_ADD": 3.0,
    }[strategy]
    return execution_cost_pips(cfg, sides)


def gross_pips_for_strategy(
    strategy: str,
    direction: str,
    arm_price: float,
    breakout_price: float,
    exit_price: float,
    symbol: str,
) -> float:
    """
    Gross PnL under linear spot-FX mid-price arithmetic.

    BREAKOUT/OCO:
        enter 1x at breakout, exit 1x later

    HEDGE_1x1:
        arm: +1 long and +1 short
        breakout: close loser
        survivor exits later

    HALF_HEDGE_ADD:
        arm: +0.5 long and +0.5 short
        breakout: close loser and add +0.5 to winner
        total winner exposure after breakout = 1.0
    """
    sign = 1.0 if direction == "LONG" else -1.0

    if strategy in ("BREAKOUT", "OCO"):
        pnl_px = sign * (exit_price - breakout_price)

    elif strategy == "HEDGE_1x1":
        # Pre-breakout long+short cancel grossly.
        winner = sign * (exit_price - arm_price)
        loser = -sign * (breakout_price - arm_price)
        pnl_px = winner + loser

    elif strategy == "HALF_HEDGE_ADD":
        pre_winner = 0.5 * sign * (exit_price - arm_price)
        pre_loser = -0.5 * sign * (breakout_price - arm_price)
        added_winner = 0.5 * sign * (exit_price - breakout_price)
        # Existing 0.5 winner also continues after breakout, already included in pre_winner.
        pnl_px = pre_winner + pre_loser + added_winner

    else:
        raise ValueError(strategy)

    return px_to_pips(pnl_px, symbol)


# ----------------------------
# Exit engine
# ----------------------------

def simulate_exit(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    entry_price: float,
    atr_at_breakout: float,
    cfg: BacktestConfig,
):
    sign = 1.0 if direction == "LONG" else -1.0

    risk_px = cfg.stop_atr * atr_at_breakout
    if direction == "LONG":
        stop = entry_price - risk_px
        tp = None if cfg.take_profit_r is None else entry_price + cfg.take_profit_r * risk_px
    else:
        stop = entry_price + risk_px
        tp = None if cfg.take_profit_r is None else entry_price - cfg.take_profit_r * risk_px

    initial_stop = stop
    initial_tp = tp

    mfe_px = 0.0
    mae_px = 0.0

    last_idx = min(len(df) - 1, entry_idx + cfg.max_hold_bars)

    for k in range(entry_idx, last_idx + 1):
        row = df.iloc[k]
        high = float(row.high)
        low = float(row.low)
        atr_now = float(row.atr) if np.isfinite(row.atr) else atr_at_breakout

        if direction == "LONG":
            mfe_px = max(mfe_px, high - entry_price)
            mae_px = min(mae_px, low - entry_price)

            # Conservative same-bar ordering: stop is checked before TP.
            if low <= stop:
                return k, stop, "STOP", initial_stop, initial_tp, mfe_px, mae_px

            if tp is not None and high >= tp:
                return k, tp, "TAKE_PROFIT", initial_stop, initial_tp, mfe_px, mae_px

            if cfg.trailing_atr is not None:
                open_profit = high - entry_price
                if open_profit >= cfg.trailing_activate_r * risk_px:
                    candidate = high - cfg.trailing_atr * atr_now
                    stop = max(stop, candidate)

        else:
            mfe_px = max(mfe_px, entry_price - low)
            mae_px = min(mae_px, entry_price - high)

            if high >= stop:
                return k, stop, "STOP", initial_stop, initial_tp, mfe_px, mae_px

            if tp is not None and low <= tp:
                return k, tp, "TAKE_PROFIT", initial_stop, initial_tp, mfe_px, mae_px

            if cfg.trailing_atr is not None:
                open_profit = entry_price - low
                if open_profit >= cfg.trailing_activate_r * risk_px:
                    candidate = low + cfg.trailing_atr * atr_now
                    stop = min(stop, candidate)

    # Time exit at close of final bar
    return (
        last_idx,
        float(df.iloc[last_idx].close),
        "TIME",
        initial_stop,
        initial_tp,
        mfe_px,
        mae_px,
    )


# ----------------------------
# Backtest
# ----------------------------

def backtest(df_raw: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    df = enrich_features(df_raw, cfg)

    strategies = ["BREAKOUT", "HEDGE_1x1", "HALF_HEDGE_ADD", "OCO"]
    trades: list[Trade] = []

    i = max(
        cfg.compression_lookback_bars,
        cfg.range_bars,
        cfg.atr_period,
    )

    while i < len(df) - 2:
        row = df.iloc[i]

        if not bool(row.compression_start):
            i += 1
            continue

        signal_idx = i
        arm_idx = i + 1  # signal known at close i; hedge can be opened at next bar open

        arm_price = float(df.iloc[arm_idx].open)
        arm_time = pd.Timestamp(df.iloc[arm_idx].timestamp)

        upper = float(row.range_high)
        lower = float(row.range_low)
        atr_signal = float(row.atr)
        atr_threshold = float(row.atr_threshold)
        range_atr_ratio = float(row.range_atr_ratio)

        if not all(np.isfinite(v) for v in [upper, lower, atr_signal, atr_threshold, range_atr_ratio]):
            i += 1
            continue

        buffer_px = cfg.breakout_buffer_atr * atr_signal

        breakout_signal_idx = None
        direction = None

        max_j = min(len(df) - 2, arm_idx + cfg.max_wait_bars)
        for j in range(arm_idx, max_j + 1):
            c = float(df.iloc[j].close)

            bullish = c > upper + buffer_px
            bearish = c < lower - buffer_px

            # If one extreme somehow crosses both thresholds in malformed data,
            # skip rather than invent an ordering.
            if bullish and bearish:
                continue
            if bullish:
                breakout_signal_idx = j
                direction = "LONG"
                break
            if bearish:
                breakout_signal_idx = j
                direction = "SHORT"
                break

        # No breakout: hedge variants would have to flatten.
        # We explicitly account for that "failed arm" cost as a synthetic no-move trade.
        if breakout_signal_idx is None:
            flatten_idx = max_j
            flatten_price = float(df.iloc[flatten_idx].close)
            flatten_time = pd.Timestamp(df.iloc[flatten_idx].timestamp)

            # Equal long+short gross PnL is zero at the same flatten price.
            for strategy, executed_sides in [("HEDGE_1x1", 4.0), ("HALF_HEDGE_ADD", 2.0)]:
                # HALF_HEDGE_ADD before breakout only opened .5 + .5 and closes .5 + .5 => 2 unit-sides.
                cost = execution_cost_pips(cfg, executed_sides)
                trades.append(
                    Trade(
                        strategy=strategy,
                        symbol=cfg.symbol,
                        direction="NONE",
                        arm_time=arm_time,
                        breakout_signal_time=pd.NaT,
                        entry_time=arm_time,
                        exit_time=flatten_time,
                        arm_price=arm_price,
                        breakout_price=np.nan,
                        exit_price=flatten_price,
                        upper_range=upper,
                        lower_range=lower,
                        atr=atr_signal,
                        atr_threshold=atr_threshold,
                        range_atr_ratio=range_atr_ratio,
                        stop_initial=np.nan,
                        take_profit_initial=np.nan,
                        gross_pips=0.0,
                        cost_pips=cost,
                        net_pips=-cost,
                        r_multiple_net=np.nan,
                        mfe_pips=0.0,
                        mae_pips=0.0,
                        holding_bars=flatten_idx - arm_idx + 1,
                        exit_reason="NO_BREAKOUT_FLATTEN",
                        session=session_name(arm_time),
                    )
                )

            i = flatten_idx + 1
            continue

        # Confirmation known at close breakout_signal_idx.
        # Execute directional transition/entry at next bar open.
        if cfg.execute_breakout_on_next_open:
            entry_idx = breakout_signal_idx + 1
            breakout_price = float(df.iloc[entry_idx].open)
            entry_time = pd.Timestamp(df.iloc[entry_idx].timestamp)
        else:
            entry_idx = breakout_signal_idx
            breakout_price = float(df.iloc[entry_idx].close)
            entry_time = pd.Timestamp(df.iloc[entry_idx].timestamp)

        breakout_signal_time = pd.Timestamp(df.iloc[breakout_signal_idx].timestamp)

        (
            exit_idx,
            exit_price,
            exit_reason,
            stop_initial,
            tp_initial,
            mfe_px,
            mae_px,
        ) = simulate_exit(
            df=df,
            entry_idx=entry_idx,
            direction=direction,
            entry_price=breakout_price,
            atr_at_breakout=atr_signal,
            cfg=cfg,
        )

        exit_time = pd.Timestamp(df.iloc[exit_idx].timestamp)
        risk_pips = px_to_pips(cfg.stop_atr * atr_signal, cfg.symbol)

        for strategy in strategies:
            gross = gross_pips_for_strategy(
                strategy=strategy,
                direction=direction,
                arm_price=arm_price,
                breakout_price=breakout_price,
                exit_price=exit_price,
                symbol=cfg.symbol,
            )
            cost = strategy_cost_pips(strategy, cfg)
            net = gross - cost

            trades.append(
                Trade(
                    strategy=strategy,
                    symbol=cfg.symbol,
                    direction=direction,
                    arm_time=arm_time,
                    breakout_signal_time=breakout_signal_time,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    arm_price=arm_price,
                    breakout_price=breakout_price,
                    exit_price=exit_price,
                    upper_range=upper,
                    lower_range=lower,
                    atr=atr_signal,
                    atr_threshold=atr_threshold,
                    range_atr_ratio=range_atr_ratio,
                    stop_initial=stop_initial,
                    take_profit_initial=tp_initial,
                    gross_pips=gross,
                    cost_pips=cost,
                    net_pips=net,
                    r_multiple_net=(net / risk_pips) if risk_pips > 0 else np.nan,
                    mfe_pips=px_to_pips(mfe_px, cfg.symbol),
                    mae_pips=px_to_pips(mae_px, cfg.symbol),
                    holding_bars=exit_idx - entry_idx + 1,
                    exit_reason=exit_reason,
                    session=session_name(entry_time),
                )
            )

        # Non-overlapping experiment: wait until this trade is over before arming another.
        i = exit_idx + 1

    return pd.DataFrame([asdict(t) for t in trades])


# ----------------------------
# Metrics
# ----------------------------

def max_drawdown_from_pips(pnls: pd.Series) -> float:
    equity = pnls.fillna(0).cumsum()
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min()) if len(dd) else np.nan


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    rows = []
    for strategy, g in trades.groupby("strategy", sort=False):
        pnl = g["net_pips"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())

        rows.append(
            {
                "strategy": strategy,
                "trades": len(g),
                "directional_trades": int((g["direction"] != "NONE").sum()),
                "no_breakout_arms": int((g["direction"] == "NONE").sum()),
                "win_rate": float((pnl > 0).mean()),
                "net_pips": float(pnl.sum()),
                "avg_net_pips": float(pnl.mean()),
                "median_net_pips": float(pnl.median()),
                "avg_win_pips": float(wins.mean()) if len(wins) else np.nan,
                "avg_loss_pips": float(losses.mean()) if len(losses) else np.nan,
                "profit_factor": (
                    float(gross_profit / gross_loss) if gross_loss > 0 else np.inf
                ),
                "expectancy_pips": float(pnl.mean()),
                "avg_R": float(g["r_multiple_net"].mean(skipna=True)),
                "max_drawdown_pips": max_drawdown_from_pips(pnl),
                "total_cost_pips": float(g["cost_pips"].sum()),
                "avg_holding_bars": float(g["holding_bars"].mean()),
            }
        )

    out = pd.DataFrame(rows)

    # Make the key comparison easy to see.
    breakout_net = out.loc[out["strategy"] == "BREAKOUT", "net_pips"]
    if len(breakout_net):
        b = float(breakout_net.iloc[0])
        out["delta_vs_breakout_pips"] = out["net_pips"] - b
    else:
        out["delta_vs_breakout_pips"] = np.nan

    return out


def summarize_by_session(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby(["strategy", "session"], dropna=False)
        .agg(
            trades=("net_pips", "size"),
            net_pips=("net_pips", "sum"),
            avg_net_pips=("net_pips", "mean"),
            win_rate=("net_pips", lambda s: (s > 0).mean()),
            avg_R=("r_multiple_net", "mean"),
        )
        .reset_index()
    )


# ----------------------------
# Synthetic data for smoke test
# ----------------------------

def generate_synthetic_fx(
    n: int = 8000,
    start: str = "2024-01-01",
    seed: int = 7,
    base_price: float = 1.10,
) -> pd.DataFrame:
    """
    Generates M15-like OHLC data with alternating low/high volatility regimes.
    It is NOT market-realistic; it exists only to verify that the engine runs.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="15min")

    rets = np.zeros(n)
    block = 240
    for start_i in range(0, n, block):
        end_i = min(n, start_i + block)
        block_id = start_i // block

        if block_id % 3 == 0:
            sigma = 0.00004   # compression
            drift = 0.0
        elif block_id % 3 == 1:
            sigma = 0.00018   # expansion
            drift = rng.choice([-1, 1]) * 0.000015
        else:
            sigma = 0.00009
            drift = 0.0

        rets[start_i:end_i] = drift + rng.normal(0, sigma, end_i - start_i)

    close = base_price + np.cumsum(rets)
    open_ = np.r_[base_price, close[:-1]]

    wick = np.abs(rng.normal(0.00005, 0.000025, n))
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick

    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compression -> breakout comparison: Breakout vs dual-direction hedge variants."
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
            "strategy",
            "trades",
            "win_rate",
            "net_pips",
            "expectancy_pips",
            "profit_factor",
            "max_drawdown_pips",
            "total_cost_pips",
            "delta_vs_breakout_pips",
        ]
        print("\n=== SUMMARY ===")
        print(summary[show_cols].to_string(index=False))

    print(f"\nSaved:")
    print(f"  {out_dir / 'trades.csv'}")
    print(f"  {out_dir / 'summary.csv'}")
    print(f"  {out_dir / 'summary_by_session.csv'}")


if __name__ == "__main__":
    main()
