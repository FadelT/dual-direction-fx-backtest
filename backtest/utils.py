from __future__ import annotations

import pandas as pd

from .config import BacktestConfig


def pip_size(symbol: str) -> float:
    s = symbol.upper().replace("/", "")
    if "BTC" in s or "ETH" in s:
        return 1.0
    return 0.01 if "JPY" in s else 0.0001


def px_to_pips(delta_price: float, symbol: str) -> float:
    return delta_price / pip_size(symbol)


def execution_cost_pips(cfg: BacktestConfig, executed_unit_sides: float) -> float:
    per_side = (
        0.5 * cfg.spread_pips
        + cfg.commission_per_side_pips
        + cfg.slippage_per_side_pips
    )
    return executed_unit_sides * per_side


def session_name(ts: pd.Timestamp) -> str:
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
    sign = 1.0 if direction == "LONG" else -1.0

    if strategy in ("BREAKOUT", "OCO"):
        pnl_px = sign * (exit_price - breakout_price)

    elif strategy == "HEDGE_1x1":
        winner = sign * (exit_price - arm_price)
        loser = -sign * (breakout_price - arm_price)
        pnl_px = winner + loser

    elif strategy == "HALF_HEDGE_ADD":
        pre_winner = 0.5 * sign * (exit_price - arm_price)
        pre_loser = -0.5 * sign * (breakout_price - arm_price)
        added_winner = 0.5 * sign * (exit_price - breakout_price)
        pnl_px = pre_winner + pre_loser + added_winner

    else:
        raise ValueError(strategy)

    return px_to_pips(pnl_px, symbol)
