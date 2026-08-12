from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_ohlc_csv(path: str | Path) -> pd.DataFrame:
    """
    Accepts common OHLC CSV layouts.

    Required logical fields:
        timestamp (or datetime, or date+time), open, high, low, close
    """
    path = Path(path)
    df = pd.read_csv(path)

    original = list(df.columns)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

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
        raise ValueError(f"Could not detect timestamp column. Columns found: {original}")

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


def generate_synthetic_fx(
    n: int = 8000,
    start: str = "2024-01-01",
    seed: int = 7,
    base_price: float = 1.10,
) -> pd.DataFrame:
    """
    Generates M15-like OHLC data with alternating low/high volatility regimes.
    Exists only to verify that the engine runs — not market-realistic.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="15min")

    rets = np.zeros(n)
    block = 240
    for start_i in range(0, n, block):
        end_i = min(n, start_i + block)
        block_id = start_i // block

        if block_id % 3 == 0:
            sigma = 0.00004
            drift = 0.0
        elif block_id % 3 == 1:
            sigma = 0.00018
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
