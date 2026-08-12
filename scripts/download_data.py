"""
Download H4 OHLC data from Binance (free, no API key required).
Saves one CSV per symbol under data/crypto/H4/.

Usage:
    python scripts/download_data.py
"""

import requests
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "BNBUSDT",
    "DOGEUSDT",
]

START_DATE = "2018-01-01"
OUT_DIR = Path(__file__).parent.parent / "data" / "crypto" / "H4"


def fetch_binance_h4(symbol: str, start_str: str = START_DATE) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    start_ms = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ms = int(datetime.now().timestamp() * 1000)
    rows = []

    while start_ms < end_ms:
        r = requests.get(
            url,
            params={"symbol": symbol, "interval": "4h", "startTime": start_ms, "limit": 1000},
            timeout=15,
        )
        data = r.json()
        if not data or isinstance(data, dict):
            break
        rows.extend(data)
        start_ms = data[-1][0] + 1
        time.sleep(0.1)

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume",
                 "close_time", "qv", "trades", "tbb", "tbq", "ignore"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df[["timestamp", "open", "high", "low", "close"]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in SYMBOLS:
        out = OUT_DIR / f"{symbol}_H4_full.csv"
        print(f"Downloading {symbol}...", end=" ", flush=True)
        df = fetch_binance_h4(symbol)
        df.to_csv(out, index=False)
        print(f"{len(df):,} bars | {df.timestamp.min().date()} -> {df.timestamp.max().date()}")


if __name__ == "__main__":
    main()
