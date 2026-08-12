from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import BacktestConfig, Trade
from .indicators import enrich_features
from .utils import (
    execution_cost_pips,
    gross_pips_for_strategy,
    px_to_pips,
    session_name,
    strategy_cost_pips,
)
from .exit import simulate_exit


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

        arm_idx = i + 1
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

        if breakout_signal_idx is None:
            flatten_idx = max_j
            flatten_price = float(df.iloc[flatten_idx].close)
            flatten_time = pd.Timestamp(df.iloc[flatten_idx].timestamp)

            for strategy, executed_sides in [("HEDGE_1x1", 4.0), ("HALF_HEDGE_ADD", 2.0)]:
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

        i = exit_idx + 1

    return pd.DataFrame([asdict(t) for t in trades])
