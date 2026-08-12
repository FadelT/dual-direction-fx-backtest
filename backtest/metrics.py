from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown_from_pips(pnls: pd.Series) -> float:
    equity = pnls.fillna(0).cumsum()
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min()) if len(dd) else np.nan


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    rows = []
    for strategy, g in trades.groupby("strategy", sort=False):
        pnl = g["net_pips"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())

        rows.append(
            {
                "strategy": strategy,
                "trades": len(g),
                "directional_trades": int((g["direction"] != "NONE").sum()),
                "no_breakout_arms": int((g["direction"] == "NONE").sum()),
                "win_rate": float((pnl > 0).mean()),
                "net_pips": float(pnl.sum()),
                "avg_net_pips": float(pnl.mean()),
                "median_net_pips": float(pnl.median()),
                "avg_win_pips": float(wins.mean()) if len(wins) else np.nan,
                "avg_loss_pips": float(losses.mean()) if len(losses) else np.nan,
                "profit_factor": (
                    float(gross_profit / gross_loss) if gross_loss > 0 else np.inf
                ),
                "expectancy_pips": float(pnl.mean()),
                "avg_R": float(g["r_multiple_net"].mean(skipna=True)),
                "max_drawdown_pips": max_drawdown_from_pips(pnl),
                "total_cost_pips": float(g["cost_pips"].sum()),
                "avg_holding_bars": float(g["holding_bars"].mean()),
            }
        )

    out = pd.DataFrame(rows)

    breakout_net = out.loc[out["strategy"] == "BREAKOUT", "net_pips"]
    if len(breakout_net):
        b = float(breakout_net.iloc[0])
        out["delta_vs_breakout_pips"] = out["net_pips"] - b
    else:
        out["delta_vs_breakout_pips"] = np.nan

    return out


def summarize_by_session(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby(["strategy", "session"], dropna=False)
        .agg(
            trades=("net_pips", "size"),
            net_pips=("net_pips", "sum"),
            avg_net_pips=("net_pips", "mean"),
            win_rate=("net_pips", lambda s: (s > 0).mean()),
            avg_R=("r_multiple_net", "mean"),
        )
        .reset_index()
    )
