from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BacktestConfig


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

    return (
        last_idx,
        float(df.iloc[last_idx].close),
        "TIME",
        initial_stop,
        initial_tp,
        mfe_px,
        mae_px,
    )
