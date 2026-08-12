"""
Paper trading engine — runs after each signal check.

- Opens trades on BREAKOUT (one per symbol, no duplicates)
- Checks open trades against actual bar high/low for stop/TP
- Enforces max hold of 96 bars (~16 days on H4) — closes at market if exceeded
- Tracks capital with 5% risk per trade
- Sends Telegram on open/close/error events
- Writes updated state atomically to paper_trades.json

Usage:
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

STATE_FILE    = Path(__file__).parent.parent / "paper_trades.json"
RISK_PER_TRADE = 0.05   # 5% of capital per trade
INITIAL_CAPITAL = 10_000.0
MAX_HOLD_BARS   = 96    # matches backtest config — force-close after 16 days on H4
BAR_SECONDS     = 4 * 3600


def log(*args):
    print(*args, file=sys.stderr, flush=True)


# ── Telegram ────────────────────────────────────────────────────────────────

def send_telegram(token: str, chat_id: str, text: str) -> bool:
    for attempt in range(1, 4):
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            )
            if r.status_code == 200:
                return True
            log(f"Telegram HTTP {r.status_code}")
        except Exception as e:
            log(f"Telegram attempt {attempt} failed: {e}")
        if attempt < 3:
            time.sleep(5)
    return False


def notify(token, chat_id, text):
    if token and chat_id:
        ok = send_telegram(token, chat_id, text)
        log(f"Telegram: {'sent ✓' if ok else 'failed ✗'}")


# ── State ────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception as e:
            log(f"WARNING: could not parse state file: {e} — starting fresh")
    return {
        "open": [], "closed": [],
        "summary": {
            "total_trades": 0, "wins": 0, "losses": 0,
            "net_pips": 0.0, "capital": INITIAL_CAPITAL,
            "initial_capital": INITIAL_CAPITAL,
        },
    }


def save_state(state: dict):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)  # atomic on POSIX


# ── Exit logic ───────────────────────────────────────────────────────────────

def bars_elapsed(entry_time_str: str) -> int:
    try:
        entry_ts = datetime.fromisoformat(entry_time_str.replace(" ", "T"))
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=timezone.utc)
        elapsed_s = (datetime.now(timezone.utc) - entry_ts).total_seconds()
        return int(elapsed_s / BAR_SECONDS)
    except Exception:
        return 0


def check_exit(trade: dict, sig: dict) -> tuple[str, float] | None:
    """
    Check the actual bar high/low (from signal JSON) for stop/TP.
    Returns (reason, exit_price) or None if still open.
    Conservative: stop checked before TP on same bar.
    """
    direction = trade["direction"]
    stop  = trade["stop"]
    tp    = trade["tp"]

    # Use actual bar high/low — added to signal JSON in signal_check.py
    bar_high = sig.get("bar_high") or sig.get("close")
    bar_low  = sig.get("bar_low")  or sig.get("close")

    if bar_high is None or bar_low is None:
        return None

    if direction == "LONG":
        if bar_low <= stop:
            return "STOP", stop
        if bar_high >= tp:
            return "TAKE_PROFIT", tp
    else:
        if bar_high >= stop:
            return "STOP", stop
        if bar_low <= tp:
            return "TAKE_PROFIT", tp

    # Max hold enforcement
    if bars_elapsed(trade["entry_time"]) >= MAX_HOLD_BARS:
        return "TIME", sig.get("close") or trade["entry_price"]

    return None


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt_open(trade: dict) -> str:
    arrow = "📈" if trade["direction"] == "LONG" else "📉"
    return (
        f"{arrow} *PAPER OPEN — {trade['symbol']}*\n"
        f"{trade['direction']}  @  {trade['entry_price']}\n"
        f"Stop: {trade['stop']}  ({trade['stop_pips']:.0f} pips)\n"
        f"TP:   {trade['tp']}  (3R)\n"
        f"Size: {trade['size']:.4f}  |  Risk: ${trade['risk_dollar']:.0f}\n"
        f"`{trade['entry_time']} UTC`"
    )


def fmt_close(trade: dict, exit_price: float, reason: str,
              pnl_pips: float, pnl_dollar: float, capital: float) -> str:
    icon = "✅" if pnl_dollar >= 0 else "❌"
    sign = "+" if pnl_pips >= 0 else ""
    return (
        f"{icon} *PAPER CLOSE — {trade['symbol']}*\n"
        f"{trade['direction']}  |  {reason}\n"
        f"Entry {trade['entry_price']}  →  Exit {exit_price}\n"
        f"P&L: {sign}{pnl_pips:.1f} pips  |  {sign}${pnl_dollar:.2f}\n"
        f"Capital: ${capital:,.2f}\n"
        f"`{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC`"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--signals", nargs="+", required=True)
    args = p.parse_args()

    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    # Load signals — skip missing or malformed files gracefully
    signals: dict[str, dict] = {}
    for f in args.signals:
        path = Path(f)
        if not path.exists():
            log(f"Signal file not found: {f} — skipping")
            continue
        try:
            sig = json.loads(path.read_text())
            if not isinstance(sig, dict) or "symbol" not in sig:
                raise ValueError("unexpected format")
            signals[sig["symbol"]] = sig
        except Exception as e:
            log(f"Could not read {f}: {e} — skipping")

    if not signals:
        log("No valid signal files — nothing to do")
        sys.exit(0)

    state   = load_state()
    changed = False

    # ── 1. Check open trades for exit ──────────────────────────────────────
    still_open = []
    for trade in state["open"]:
        symbol = trade["symbol"]
        sig    = signals.get(symbol)

        if sig is None:
            log(f"{symbol}: no signal this bar — keeping position open")
            still_open.append(trade)
            continue

        result = check_exit(trade, sig)
        if result is None:
            still_open.append(trade)
            continue

        reason, exit_price = result
        PS    = pip_size(symbol)
        sign  = 1.0 if trade["direction"] == "LONG" else -1.0
        pnl_pips   = sign * (exit_price - trade["entry_price"]) / PS
        pnl_dollar = pnl_pips * PS * trade["size"]

        state["summary"]["capital"]      += pnl_dollar
        state["summary"]["net_pips"]     += pnl_pips
        state["summary"]["total_trades"] += 1
        if pnl_dollar >= 0:
            state["summary"]["wins"] += 1
        else:
            state["summary"]["losses"] += 1

        trade.update({
            "exit_price":  round(exit_price, 6),
            "exit_time":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "exit_reason": reason,
            "pnl_pips":    round(pnl_pips, 1),
            "pnl_dollar":  round(pnl_dollar, 2),
            "status":      "closed",
        })
        state["closed"].append(trade)
        changed = True
        log(f"Closed {symbol} {reason}  {pnl_pips:+.1f} pips  ${pnl_dollar:+.2f}")
        notify(token, chat_id, fmt_close(
            trade, exit_price, reason, pnl_pips, pnl_dollar, state["summary"]["capital"]
        ))

    state["open"] = still_open

    # ── 2. Open new trades on BREAKOUT ────────────────────────────────────
    open_symbols = {t["symbol"] for t in state["open"]}
    for symbol, sig in signals.items():
        if sig.get("signal") != "BREAKOUT":
            continue
        if symbol in open_symbols:
            log(f"{symbol}: position already open — skipping new entry")
            continue

        entry = sig.get("entry_hint")
        stop  = sig.get("stop_hint")
        tp    = sig.get("tp_hint")
        if not all(isinstance(v, (int, float)) for v in [entry, stop, tp]):
            log(f"{symbol}: invalid entry/stop/tp in signal — skipping")
            continue

        stop_px = abs(entry - stop)
        if stop_px <= 0:
            log(f"{symbol}: stop_px=0 — skipping")
            continue

        capital     = state["summary"]["capital"]
        risk_dollar = capital * RISK_PER_TRADE
        PS          = pip_size(symbol)

        trade = {
            "id":              str(uuid.uuid4())[:8],
            "symbol":          symbol,
            "direction":       sig["direction"],
            "entry_time":      sig.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")),
            "entry_price":     entry,
            "stop":            stop,
            "tp":              tp,
            "stop_pips":       round(stop_px / PS, 1),
            "size":            round(risk_dollar / stop_px, 6),
            "risk_dollar":     round(risk_dollar, 2),
            "capital_at_open": round(capital, 2),
            "status":          "open",
        }
        state["open"].append(trade)
        open_symbols.add(symbol)
        changed = True
        log(f"Opened {symbol} {sig['direction']} @ {entry}  size={trade['size']:.4f}")
        notify(token, chat_id, fmt_open(trade))

    # ── 3. Save & report ──────────────────────────────────────────────────
    if changed:
        save_state(state)
        log("State saved.")
    else:
        log("No changes this bar.")

    print(json.dumps(state["summary"], indent=2))


if __name__ == "__main__":
    main()
