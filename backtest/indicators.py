from __future__ import annotations

import pandas as pd

from .config import BacktestConfig


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
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def enrich_features(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    x = df.copy()
    x["atr"] = wilder_atr(x, cfg.atr_period)

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
