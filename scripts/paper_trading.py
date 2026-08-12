"""
Paper trading engine — runs after each signal check.

Reads signal JSONs produced by signal_check.py, updates paper_trades.json:
  - Opens a new paper trade when a BREAKOUT fires (if no trade already open for that symbol)
  - Checks open trades against the latest bar (high/low) to detect stop or TP hits
  - Sends Telegram notifications on open/close events
  - Commits updated paper_trades.json back to the repo

Usage (called by GitHub Actions after signal jobs):
    python scripts/paper_trading.py --signals signal_XRPUSDT.json signal_BTCUSDT.json ...
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest import pip_size

STATE_FILE = Path(__file__).parent.parent / "paper_trades.json"
RISK_PER_TRADE = 0.05   # 5% of capital per trade
INITIAL_CAPITAL = 10_000.0


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def send_telegram(token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=10)
        return r.status_code == 200
    except Exception as e:
        log(f"Telegram error: {e}")
        return False


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "open": [], "closed": [],
        "summary": {
            "total_trades": 0, "wins": 0, "losses": 0,
            "net_pips": 0.0, "capital": INITIAL_CAPITAL,
            "initial_capital": INITIAL_CAPITAL,
        },
    }


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def open_symbols(state: dict) -> set:
    return {t["symbol"] for t in state["open"]}


def check_bar_exit(trade: dict, bar: dict) -> tuple[str, float] | None:
    """
    Check if the latest bar (with high/low) triggers stop or TP.
    Returns (reason, exit_price) or None if still open.
    Conservative: stop checked before TP on same bar.
    """
    direction = trade["direction"]
    stop  = trade["stop"]
    tp    = trade["tp"]
    high  = bar["high"]
    low   = bar["low"]

    if direction == "LONG":
        if low <= stop:
            return "STOP", stop
        if high >= tp:
            return "TAKE_PROFIT", tp
    else:
        if high >= stop:
            return "STOP", stop
        if low <= tp:
            return "TAKE_PROFIT", tp
    return None


def pips(price_diff: float, symbol: str) -> float:
    return price_diff / pip_size(symbol)


def format_open_msg(trade: dict) -> str:
    arrow = "📈" if trade["direction"] == "LONG" else "📉"
    capital = trade.get("capital_at_open", INITIAL_CAPITAL)
    size = trade.get("size", 1.0)
    return (
        f"{arrow} *PAPER TRADE OPENED — {trade['symbol']}*\n"
        f"Direction: {trade['direction']}\n"
        f"Entry: {trade['entry_price']}\n"
        f"Stop: {trade['stop']}  ({trade['stop_pips']:.0f} pips)\n"
        f"TP: {trade['tp']}  (3R)\n"
        f"Size: {size:.4f} units  |  Risk: ${trade['risk_dollar']:.0f}\n"
        f"`{trade['entry_time']} UTC`"
    )


def format_close_msg(trade: dict, exit_price: float, reason: str, pnl_pips: float, pnl_dollar: float, capital: float) -> str:
    icon = "✅" if reason == "TAKE_PROFIT" else "❌"
    sign = "+" if pnl_pips >= 0 else ""
    return (
        f"{icon} *PAPER TRADE CLOSED — {trade['symbol']}*\n"
        f"Direction: {trade['direction']}  |  Reason: {reason}\n"
        f"Entry: {trade['entry_price']}  →  Exit: {exit_price}\n"
        f"P&L: {sign}{pnl_pips:.1f} pips  |  {sign}${pnl_dollar:.2f}\n"
        f"Capital: ${capital:,.2f}\n"
        f"`{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC`"
    )


def format_summary_msg(state: dict) -> str:
    s = state["summary"]
    ret = (s["capital"] - s["initial_capital"]) / s["initial_capital"] * 100
    wr = s["wins"] / s["total_trades"] * 100 if s["total_trades"] else 0
    open_symbols = ", ".join(t["symbol"] for t in state["open"]) or "none"
    return (
        f"📊 *Paper Trading Summary*\n"
        f"Capital: ${s['capital']:,.2f}  ({ret:+.1f}%)\n"
        f"Trades: {s['total_trades']}  |  Win rate: {wr:.0f}%\n"
        f"Net pips: {s['net_pips']:+.1f}\n"
        f"Open positions: {open_symbols}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--signals", nargs="+", required=True, help="Signal JSON files from signal_check.py")
    p.add_argument("--bar-jsons", nargs="*", default=[], help="Latest bar JSON files (optional, for exit check)")
    args = p.parse_args()

    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    def notify(text: str):
        if token and chat_id:
            ok = send_telegram(token, chat_id, text)
            log(f"Telegram: {'sent ✓' if ok else 'failed ✗'}")

    state = load_state()
    events = []
    changed = False

    # --- 1. Load all signals ---
    signals = {}
    for f in args.signals:
        try:
            sig = json.loads(Path(f).read_text())
            signals[sig["symbol"]] = sig
        except Exception as e:
            log(f"Could not read {f}: {e}")

    # --- 2. Check open trades for stop/TP against latest signal bar ---
    still_open = []
    for trade in state["open"]:
        symbol = trade["symbol"]
        sig = signals.get(symbol)
        if not sig or sig.get("close") is None:
            still_open.append(trade)
            continue

        # Approximate latest bar using signal close/range as proxy
        # (signal_check already fetched latest bar data)
        bar = {
            "high": sig.get("range_high") or sig["close"],
            "low":  sig.get("range_low")  or sig["close"],
            "close": sig["close"],
        }

        result = check_bar_exit(trade, bar)
        if result is None:
            still_open.append(trade)
            continue

        reason, exit_price = result
        PS = pip_size(symbol)
        sign = 1.0 if trade["direction"] == "LONG" else -1.0
        pnl_pips = sign * (exit_price - trade["entry_price"]) / PS
        pnl_dollar = pnl_pips * PS * trade["size"]

        state["summary"]["capital"]    += pnl_dollar
        state["summary"]["net_pips"]   += pnl_pips
        state["summary"]["total_trades"] += 1
        if pnl_pips > 0:
            state["summary"]["wins"] += 1
        else:
            state["summary"]["losses"] += 1

        trade.update({
            "exit_price": exit_price,
            "exit_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "exit_reason": reason,
            "pnl_pips": round(pnl_pips, 1),
            "pnl_dollar": round(pnl_dollar, 2),
            "status": "closed",
        })
        state["closed"].append(trade)
        events.append(("close", trade, exit_price, reason, pnl_pips, pnl_dollar))
        changed = True
        log(f"Closed {symbol} {reason}  pnl={pnl_pips:+.1f} pips  ${pnl_dollar:+.2f}")

    state["open"] = still_open

    # --- 3. Open new trades on BREAKOUT (one position per symbol) ---
    open_syms = open_symbols(state)
    for symbol, sig in signals.items():
        if sig.get("signal") != "BREAKOUT":
            continue
        if symbol in open_syms:
            log(f"{symbol}: already has an open trade, skipping")
            continue

        capital = state["summary"]["capital"]
        PS      = pip_size(symbol)
        stop_px = abs(sig["entry_hint"] - sig["stop_hint"])
        if stop_px <= 0:
            continue

        risk_dollar = capital * RISK_PER_TRADE
        size        = risk_dollar / stop_px
        stop_pips   = stop_px / PS

        trade = {
            "id":              str(uuid.uuid4())[:8],
            "symbol":          symbol,
            "direction":       sig["direction"],
            "entry_time":      sig["timestamp"],
            "entry_price":     sig["entry_hint"],
            "stop":            sig["stop_hint"],
            "tp":              sig["tp_hint"],
            "stop_pips":       round(stop_pips, 1),
            "size":            round(size, 6),
            "risk_dollar":     round(risk_dollar, 2),
            "capital_at_open": round(capital, 2),
            "status":          "open",
        }
        state["open"].append(trade)
        events.append(("open", trade))
        changed = True
        log(f"Opened {symbol} {sig['direction']} @ {sig['entry_hint']}  size={size:.4f}")

    # --- 4. Send Telegram notifications ---
    for event in events:
        if event[0] == "open":
            notify(format_open_msg(event[1]))
        else:
            _, trade, exit_price, reason, pnl_pips, pnl_dollar = event
            notify(format_close_msg(trade, exit_price, reason, pnl_pips, pnl_dollar, state["summary"]["capital"]))
        time.sleep(0.5)

    # Send summary if anything changed
    if changed:
        notify(format_summary_msg(state))

    # --- 5. Save state ---
    if changed:
        save_state(state)
        log("State saved.")
    else:
        log("No changes.")

    # Print state for CI log
    print(json.dumps(state["summary"], indent=2))


if __name__ == "__main__":
    main()
