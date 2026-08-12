"""
Live signal checker — run after each H4 bar close.

Detects:
  ARMED     — compression just started, watching for breakout
  BREAKOUT  — price broke out of compressed range (actionable)
  NONE      — no active setup

Usage:
    python scripts/signal_check.py --symbol XRPUSDT
    python scripts/signal_check.py --symbol XRPUSDT --csv data/crypto/H4/XRPUSDT_H4_full.csv
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


def log(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


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

KRAKEN_SYMBOL_MAP = {
    "BTCUSDT":  "XBTUSDT",
    "ETHUSDT":  "ETHUSDT",
    "XRPUSDT":  "XRPUSDT",
    "SOLUSDT":  "SOLUSDT",
    "AVAXUSDT": "AVAXUSDT",
    "BNBUSDT":  "BNBUSDT",
    "DOGEUSDT": "DOGEUSDT",
    "LINKUSDT": "LINKUSDT",
}

MAX_RETRIES = 3
RETRY_DELAY = 15  # seconds


def fetch_recent_bars(symbol: str, n: int = 500) -> pd.DataFrame:
    """Fetch recent H4 bars from Kraken with retry logic."""
    kraken_pair = KRAKEN_SYMBOL_MAP.get(symbol.upper(), symbol.upper())
    since = int(datetime.now().timestamp()) - n * 4 * 3600

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            rows = []
            s = since
            while True:
                r = requests.get(
                    "https://api.kraken.com/0/public/OHLC",
                    params={"pair": kraken_pair, "interval": 240, "since": s},
                    timeout=20,
                )
                r.raise_for_status()
                body = r.json()

                if body.get("error"):
                    log(f"Kraken API error: {body['error']}")
                    break

                result = body.get("result", {})
                pair_key = next((k for k in result if k != "last"), None)
                if pair_key is None:
                    break

                batch = result[pair_key]
                if not batch:
                    break

                rows.extend(batch)
                if len(batch) < 720:
                    break
                s = result.get("last", 0)
                time.sleep(0.5)

            if not rows:
                raise ValueError("Empty response from Kraken")

            df = pd.DataFrame(
                rows,
                columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
            for c in ["open", "high", "low", "close"]:
                df[c] = df[c].astype(float)
            df = df[["timestamp", "open", "high", "low", "close"]].tail(n).reset_index(drop=True)

            if df.empty:
                raise ValueError("DataFrame empty after processing")

            return df

        except Exception as e:
            log(f"Fetch attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])


def empty_signal(symbol: str, error: str = "") -> dict:
    return {
        "symbol": symbol, "signal": "NONE", "error": error,
        "direction": None, "timestamp": None, "close": None,
        "bar_high": None, "bar_low": None,
        "atr": None, "range_high": None, "range_low": None,
        "entry_hint": None, "stop_hint": None, "tp_hint": None, "stop_pips": None,
    }


def check_signal(df: pd.DataFrame, symbol: str, params: dict | None = None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    cfg = BacktestConfig(symbol=symbol, **p)

    min_bars = cfg.compression_lookback_bars + cfg.range_bars + 2
    if len(df) < min_bars:
        log(f"Not enough bars: got {len(df)}, need {min_bars}")
        return empty_signal(symbol, "insufficient_bars")

    x = enrich_features(df, cfg)
    last = x.iloc[-2]  # last fully closed bar (-1 may still be forming)

    # Validate core values are finite
    if not all(np.isfinite([float(last.close), float(last.high), float(last.low)])):
        return empty_signal(symbol, "non_finite_bar")

    base = {
        "symbol":     symbol,
        "timestamp":  str(last.timestamp),
        "close":      float(last.close),
        "bar_high":   float(last.high),   # actual bar high — used by paper trading for exit checks
        "bar_low":    float(last.low),    # actual bar low  — used by paper trading for exit checks
        "atr":        float(last.atr)        if np.isfinite(last.atr)        else None,
        "range_high": float(last.range_high) if np.isfinite(last.range_high) else None,
        "range_low":  float(last.range_low)  if np.isfinite(last.range_low)  else None,
        "signal":     "NONE",
        "direction":  None,
        "entry_hint": None,
        "stop_hint":  None,
        "tp_hint":    None,
        "stop_pips":  None,
    }

    lookback = x.iloc[-(cfg.max_wait_bars + 2):-1]
    armed_rows = lookback[lookback["compression_start"] == True]

    if armed_rows.empty:
        return base

    arm_row    = armed_rows.iloc[-1]
    upper      = float(arm_row.range_high)
    lower      = float(arm_row.range_low)
    atr_at_arm = float(arm_row.atr)

    if not all(np.isfinite([upper, lower, atr_at_arm])):
        return base

    buffer = cfg.breakout_buffer_atr * atr_at_arm
    PS     = pip_size(symbol)
    close  = float(last.close)

    if close > upper + buffer:
        direction = "LONG"
        entry     = close
        stop      = entry - cfg.stop_atr * atr_at_arm
        tp        = entry + cfg.take_profit_r * cfg.stop_atr * atr_at_arm
        stop_px   = entry - stop
    elif close < lower - buffer:
        direction = "SHORT"
        entry     = close
        stop      = entry + cfg.stop_atr * atr_at_arm
        tp        = entry - cfg.take_profit_r * cfg.stop_atr * atr_at_arm
        stop_px   = stop - entry
    else:
        base["signal"] = "ARMED"
        return base

    if stop_px <= 0:
        return base

    base.update({
        "signal":     "BREAKOUT",
        "direction":  direction,
        "entry_hint": round(entry, 6),
        "stop_hint":  round(stop, 6),
        "tp_hint":    round(tp, 6),
        "stop_pips":  round(stop_px / PS, 1),
    })
    return base


def send_telegram(token: str, chat_id: str, text: str) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            )
            if r.status_code == 200:
                return True
            log(f"Telegram HTTP {r.status_code}: {r.text[:100]}")
        except Exception as e:
            log(f"Telegram attempt {attempt} failed: {e}")
        if attempt < MAX_RETRIES:
            time.sleep(5)
    return False


def format_telegram(sig: dict) -> str | None:
    if sig["signal"] == "NONE":
        return None
    symbol = sig["symbol"]
    ts     = sig["timestamp"]
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
    p.add_argument("--csv",    default=None)
    p.add_argument("--bars",   type=int, default=500)
    p.add_argument("--json",   action="store_true")
    p.add_argument("--params", default=None)
    args = p.parse_args()

    symbol = args.symbol.upper()

    if args.csv:
        from backtest import load_ohlc_csv
        df = load_ohlc_csv(args.csv)
    else:
        log(f"Fetching last {args.bars} H4 bars for {symbol}...")
        df = fetch_recent_bars(symbol, n=args.bars)
        if df.empty:
            log("ERROR: no data returned after retries — writing NONE signal")
            sig = empty_signal(symbol, "api_failure")
            if args.json:
                print(json.dumps(sig, indent=2))
            sys.exit(2)
        log(f"  Got {len(df)} bars ({df.timestamp.iloc[0].date()} → {df.timestamp.iloc[-1].date()})")

    params = json.loads(Path(args.params).read_text()) if args.params else None
    sig    = check_signal(df, symbol, params)

    # Telegram always fires before any early return
    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        msg = format_telegram(sig)
        if msg:
            ok = send_telegram(token, chat_id, msg)
            log(f"Telegram: {'sent ✓' if ok else 'failed ✗'}")

    if args.json:
        print(json.dumps(sig, indent=2))
        sys.exit(0 if sig["signal"] != "NONE" else 1)

    print(f"\n{'='*40}")
    print(f"Signal   : {sig['signal']}")
    print(f"Symbol   : {sig['symbol']}")
    print(f"Time     : {sig['timestamp']}")
    print(f"Close    : {sig['close']}")
    if sig["signal"] == "BREAKOUT":
        print(f"Direction: {sig['direction']}")
        print(f"Entry    : {sig['entry_hint']}")
        print(f"Stop     : {sig['stop_hint']}  ({sig['stop_pips']} pips)")
        print(f"TP       : {sig['tp_hint']}  (3R)")
    elif sig["signal"] == "ARMED":
        print(f"Range    : {sig['range_low']} → {sig['range_high']}")
    print(f"{'='*40}\n")

    sys.exit(0 if sig["signal"] != "NONE" else 1)


if __name__ == "__main__":
    main()
