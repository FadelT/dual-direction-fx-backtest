from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


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
