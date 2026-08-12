"""
Paper trading performance report.

Reads paper_trades.json and prints a full breakdown:
  - Overall stats (return, win rate, drawdown, profit factor)
  - Per-symbol breakdown
  - Recent closed trades
  - ASCII equity curve

Also sends a Telegram summary if TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are set.

Usage:
    python scripts/report.py
    python scripts/report.py --telegram-only
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

STATE_FILE = Path(__file__).parent.parent / "paper_trades.json"


def load_state() -> dict:
    return json.loads(STATE_FILE.read_text())


def send_telegram(token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}", file=sys.stderr)
        return False


def equity_curve(closed: list, initial: float) -> list[float]:
    curve = [initial]
    cap = initial
    for t in sorted(closed, key=lambda x: x.get("exit_time", "")):
        cap += t.get("pnl_dollar", 0)
        curve.append(cap)
    return curve


def max_drawdown(curve: list[float]) -> float:
    peak = curve[0]
    max_dd = 0.0
    for v in curve:
        peak = max(peak, v)
        dd = (v - peak) / peak
        max_dd = min(max_dd, dd)
    return max_dd


def ascii_equity(curve: list[float], width: int = 40, height: int = 8) -> str:
    if len(curve) < 2:
        return "  (no closed trades yet)"
    lo, hi = min(curve), max(curve)
    span = hi - lo or 1
    rows = []
    for row in range(height - 1, -1, -1):
        threshold = lo + span * row / (height - 1)
        line = ""
        for val in curve:
            line += "█" if val >= threshold else " "
        label = f"${threshold:>8,.0f} |"
        rows.append(label + line)
    rows.append(" " * 10 + "+" + "─" * len(curve))
    return "\n".join(rows)


def symbol_breakdown(closed: list) -> dict:
    by_sym: dict[str, dict] = {}
    for t in closed:
        s = t["symbol"]
        if s not in by_sym:
            by_sym[s] = {"trades": 0, "wins": 0, "net_pips": 0.0, "net_dollar": 0.0}
        by_sym[s]["trades"] += 1
        pnl = t.get("pnl_pips", 0)
        usd = t.get("pnl_dollar", 0)
        by_sym[s]["net_pips"] += pnl
        by_sym[s]["net_dollar"] += usd
        if pnl > 0:
            by_sym[s]["wins"] += 1
    return by_sym


def build_report(state: dict) -> str:
    s = state["summary"]
    closed = state["closed"]
    open_trades = state["open"]
    initial = s["initial_capital"]
    capital = s["capital"]
    ret = (capital - initial) / initial * 100
    total = s["total_trades"]
    wr = s["wins"] / total * 100 if total else 0

    gross_profit = sum(t["pnl_dollar"] for t in closed if t.get("pnl_dollar", 0) > 0)
    gross_loss = abs(sum(t["pnl_dollar"] for t in closed if t.get("pnl_dollar", 0) < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    curve = equity_curve(closed, initial)
    dd = max_drawdown(curve)

    lines = []
    lines.append("=" * 52)
    lines.append("  PAPER TRADING PERFORMANCE REPORT")
    lines.append("=" * 52)
    lines.append(f"  Initial capital : ${initial:>10,.2f}")
    lines.append(f"  Current capital : ${capital:>10,.2f}  ({ret:+.1f}%)")
    lines.append(f"  Net pips        : {s['net_pips']:>+10.1f}")
    lines.append(f"  Max drawdown    : {dd:>10.1%}")
    lines.append(f"  Total trades    : {total:>10}")
    lines.append(f"  Win rate        : {wr:>9.1f}%  ({s['wins']}W / {s['losses']}L)")
    lines.append(f"  Profit factor   : {pf:>10.2f}")
    lines.append(f"  Open positions  : {len(open_trades):>10}")
    lines.append("")

    # Per-symbol
    if closed:
        lines.append("─" * 52)
        lines.append(f"  {'Symbol':<12} {'Trades':>6} {'Win%':>6} {'Net pips':>10} {'Net $':>10}")
        lines.append("─" * 52)
        for sym, d in sorted(symbol_breakdown(closed).items()):
            sym_wr = d["wins"] / d["trades"] * 100
            lines.append(
                f"  {sym:<12} {d['trades']:>6} {sym_wr:>5.0f}% "
                f"{d['net_pips']:>+10.1f} {d['net_dollar']:>+10.2f}"
            )
        lines.append("")

    # Recent closed trades (last 10)
    if closed:
        lines.append("─" * 52)
        lines.append("  RECENT CLOSED TRADES (last 10)")
        lines.append("─" * 52)
        for t in sorted(closed, key=lambda x: x.get("exit_time", ""))[-10:]:
            icon = "✓" if t.get("pnl_pips", 0) > 0 else "✗"
            lines.append(
                f"  {icon} {t['symbol']:<10} {t['direction']:<6} "
                f"{t.get('exit_reason','?'):<13} "
                f"{t.get('pnl_pips', 0):>+8.1f} pips  "
                f"${t.get('pnl_dollar', 0):>+8.2f}"
            )
        lines.append("")

    # Open positions
    if open_trades:
        lines.append("─" * 52)
        lines.append("  OPEN POSITIONS")
        lines.append("─" * 52)
        for t in open_trades:
            lines.append(
                f"  {t['symbol']:<10} {t['direction']:<6} "
                f"entry={t['entry_price']}  "
                f"stop={t['stop']}  TP={t['tp']}"
            )
        lines.append("")

    # Equity curve
    lines.append("─" * 52)
    lines.append("  EQUITY CURVE")
    lines.append("─" * 52)
    lines.append(ascii_equity(curve))
    lines.append("=" * 52)

    return "\n".join(lines)


def build_telegram_summary(state: dict) -> str:
    s = state["summary"]
    closed = state["closed"]
    open_trades = state["open"]
    initial = s["initial_capital"]
    capital = s["capital"]
    ret = (capital - initial) / initial * 100
    total = s["total_trades"]
    wr = s["wins"] / total * 100 if total else 0

    curve = equity_curve(closed, initial)
    dd = max_drawdown(curve)

    trend = "📈" if ret >= 0 else "📉"
    lines = [
        f"{trend} *Weekly Paper Trading Report*",
        "",
        f"Capital: ${capital:,.2f}  ({ret:+.1f}%)",
        f"Max drawdown: {dd:.1%}",
        f"Trades: {total}  |  Win rate: {wr:.0f}%",
        f"Net pips: {s['net_pips']:+.1f}",
        "",
    ]

    if closed:
        lines.append("*By symbol:*")
        for sym, d in sorted(symbol_breakdown(closed).items()):
            sym_wr = d["wins"] / d["trades"] * 100
            sign = "+" if d["net_dollar"] >= 0 else ""
            lines.append(
                f"  {sym}: {d['trades']} trades  {sym_wr:.0f}% wr  "
                f"{sign}${d['net_dollar']:.0f}"
            )
        lines.append("")

    if open_trades:
        syms = ", ".join(f"{t['symbol']} {t['direction']}" for t in open_trades)
        lines.append(f"*Open:* {syms}")

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--telegram-only", action="store_true", help="Only send Telegram, skip console output")
    args = p.parse_args()

    state = load_state()

    if not args.telegram_only:
        print(build_report(state))

    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        msg = build_telegram_summary(state)
        ok = send_telegram(token, chat_id, msg)
        print(f"Telegram: {'sent ✓' if ok else 'failed ✗'}", file=sys.stderr)


if __name__ == "__main__":
    main()
