"""
Live signal checker — run after each H4 bar close.

Detects:
  ARMED     — compression just started, watching for breakout
  BREAKOUT  — price broke out of compressed range (actionable)
  NONE      — no active setup

Usage:
    python scripts/signal_check.py --symbol XRPUSDT --csv data/crypto/H4/XRPUSDT_H4_full.csv
    python scripts/signal_check.py --symbol XRPUSDT  # downloads fresh data automatically
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest import BacktestConfig, enrich_features, pip_size


# Params that performed consistently well across CV walk-forward runs
DEFAULT_PARAMS = {
    "atr_quantile": 0.30,
    "max_range_atr": 3.0,
    "breakout_buffer_atr": 0.10,
    "stop_atr": 1.5,
    "take_profit_r": 3.0,
    "compression_lookback_bars": 360,
    "spread_pips": 5,
    "slippage_per_side_pips": 2,
}


def fetch_recent_bars(symbol: str, n: int = 500) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    rows = []
    end_ms = int(datetime.now().timestamp() * 1000)
    # Each H4 bar = 4h = 14_400_000 ms
    start_ms = end_ms - n * 14_400_000

    while start_ms < end_ms:
        r = requests.get(url, params={
            "symbol": symbol, "interval": "4h",
            "startTime": start_ms, "limit": 1000,
        }, timeout=15)
        data = r.json()
        if not data or isinstance(data, dict):
            break
        rows.extend(data)
        start_ms = data[-1][0] + 1
        time.sleep(0.1)

    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "qv", "trades", "tbb", "tbq", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df[["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)


def check_signal(df: pd.DataFrame, symbol: str, params: dict | None = None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    cfg = BacktestConfig(symbol=symbol, **p)

    x = enrich_features(df, cfg)

    # Use the last fully closed bar (-2 avoids the still-forming current bar)
    last = x.iloc[-2]

    base = {
        "symbol": symbol,
        "timestamp": str(last.timestamp),
        "close": float(last.close),
        "atr": float(last.atr) if np.isfinite(last.atr) else None,
        "range_high": float(last.range_high) if np.isfinite(last.range_high) else None,
        "range_low": float(last.range_low) if np.isfinite(last.range_low) else None,
        "signal": "NONE",
        "direction": None,
        "entry_hint": None,
        "stop_hint": None,
        "tp_hint": None,
        "stop_pips": None,
    }

    # Look back within max_wait_bars for the most recent compression start
    lookback = x.iloc[-(cfg.max_wait_bars + 2):-1]
    armed_rows = lookback[lookback["compression_start"] == True]

    if armed_rows.empty:
        return base

    arm_row = armed_rows.iloc[-1]
    upper = float(arm_row.range_high)
    lower = float(arm_row.range_low)
    atr_at_arm = float(arm_row.atr)
    buffer = cfg.breakout_buffer_atr * atr_at_arm

    if not all(np.isfinite([upper, lower, atr_at_arm])):
        return base

    PS = pip_size(symbol)
    close = float(last.close)

    if close > upper + buffer:
        direction = "LONG"
        entry = close
        stop = entry - cfg.stop_atr * atr_at_arm
        tp = entry + cfg.take_profit_r * cfg.stop_atr * atr_at_arm
        stop_pips = (entry - stop) / PS
        base.update({
            "signal": "BREAKOUT",
            "direction": direction,
            "entry_hint": round(entry, 6),
            "stop_hint": round(stop, 6),
            "tp_hint": round(tp, 6),
            "stop_pips": round(stop_pips, 1),
        })
    elif close < lower - buffer:
        direction = "SHORT"
        entry = close
        stop = entry + cfg.stop_atr * atr_at_arm
        tp = entry - cfg.take_profit_r * cfg.stop_atr * atr_at_arm
        stop_pips = (stop - entry) / PS
        base.update({
            "signal": "BREAKOUT",
            "direction": direction,
            "entry_hint": round(entry, 6),
            "stop_hint": round(stop, 6),
            "tp_hint": round(tp, 6),
            "stop_pips": round(stop_pips, 1),
        })
    else:
        base["signal"] = "ARMED"

    return base


def send_telegram(token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=10)
    return r.status_code == 200


def format_telegram(sig: dict) -> str | None:
    if sig["signal"] == "NONE":
        return None

    symbol = sig["symbol"]
    ts = sig["timestamp"]

    if sig["signal"] == "ARMED":
        return (
            f"⚡ *COMPRESSION — {symbol}*\n"
            f"Range: {sig['range_low']} → {sig['range_high']}\n"
            f"ATR: {sig['atr']:.4f}\n"
            f"Watching for breakout...\n"
            f"`{ts} UTC`"
        )

    arrow = "📈" if sig["direction"] == "LONG" else "📉"
    return (
        f"{arrow} *BREAKOUT {sig['direction']} — {symbol}*\n"
        f"Entry (next open): ~{sig['entry_hint']}\n"
        f"Stop: {sig['stop_hint']}  ({sig['stop_pips']} pips)\n"
        f"TP: {sig['tp_hint']}  (3R)\n"
        f"`{ts} UTC`"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--csv", default=None, help="Local CSV. If omitted, fetches from Binance.")
    p.add_argument("--bars", type=int, default=500, help="Bars to fetch when no CSV given.")
    p.add_argument("--json", action="store_true", help="Print signal as JSON.")
    p.add_argument("--params", default=None, help="JSON file with override params.")
    args = p.parse_args()

    symbol = args.symbol.upper()

    if args.csv:
        from backtest import load_ohlc_csv
        df = load_ohlc_csv(args.csv)
    else:
        print(f"Fetching last {args.bars} H4 bars for {symbol}...", flush=True)
        df = fetch_recent_bars(symbol, n=args.bars)
        print(f"  Got {len(df)} bars ({df.timestamp.iloc[0].date()} → {df.timestamp.iloc[-1].date()})")

    params = None
    if args.params:
        params = json.loads(Path(args.params).read_text())

    sig = check_signal(df, symbol, params)

    if args.json:
        print(json.dumps(sig, indent=2))
        return

    print(f"\n{'='*40}")
    print(f"Signal  : {sig['signal']}")
    print(f"Symbol  : {sig['symbol']}")
    print(f"Time    : {sig['timestamp']}")
    print(f"Close   : {sig['close']}")
    if sig["signal"] == "BREAKOUT":
        print(f"Direction: {sig['direction']}")
        print(f"Entry   : {sig['entry_hint']}")
        print(f"Stop    : {sig['stop_hint']}  ({sig['stop_pips']} pips)")
        print(f"TP      : {sig['tp_hint']}  (3R)")
    elif sig["signal"] == "ARMED":
        print(f"Range   : {sig['range_low']} → {sig['range_high']}")
    print(f"{'='*40}\n")

    # Telegram (only if env vars are set)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        msg = format_telegram(sig)
        if msg:
            ok = send_telegram(token, chat_id, msg)
            print(f"Telegram: {'sent ✓' if ok else 'failed ✗'}")

    # Exit code 0 = signal (ARMED or BREAKOUT), 1 = no signal
    # Lets GitHub Actions detect signals via step outcome
    sys.exit(0 if sig["signal"] != "NONE" else 1)


if __name__ == "__main__":
    main()
