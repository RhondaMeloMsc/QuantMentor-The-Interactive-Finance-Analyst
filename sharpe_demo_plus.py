
#!/usr/bin/env python3
"""Sharpe Ratio Demo Plus (Educational)

Enhancements:
- Accepts --window for rolling Sharpe (in trading days)
- Prints a crude t-stat for mean excess return
- Optionally saves a simple PNG chart of rolling Sharpe (no external data calls)

Usage:
    python scripts/sharpe_demo_plus.py examples/sample_returns.csv --window 60 --out rolling_sharpe.png

CSV Requirements:
- Must include columns: date, strategy_return, risk_free_daily
"""
import argparse
import math
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TRADING_DAYS = 252


def compute_basic_stats(excess: pd.Series):
    """Compute basic daily and annualized Sharpe, plus crude t-stat of mean excess.
    Returns a dict with keys: n, mean_excess, std_excess, sharpe_daily, sharpe_annual, t_stat
    """
    # Drop NA just in case
    xs = excess.dropna().astype(float)
    n = xs.shape[0]
    if n < 2:
        return {
            "n": n,
            "mean_excess": np.nan,
            "std_excess": np.nan,
            "sharpe_daily": np.nan,
            "sharpe_annual": np.nan,
            "t_stat": np.nan,
        }

    mean_excess = xs.mean()
    std_excess = xs.std(ddof=1)  # sample std
    sharpe_daily = mean_excess / std_excess if std_excess != 0 else np.nan
    sharpe_annual = sharpe_daily * math.sqrt(TRADING_DAYS) if not np.isnan(sharpe_daily) else np.nan

    # crude t-stat of the mean (assumes iid normal-ish; purely educational)
    se = std_excess / math.sqrt(n) if n > 1 else np.nan
    t_stat = mean_excess / se if (se not in (0, np.nan) and not np.isnan(se)) else np.nan

    return {
        "n": n,
        "mean_excess": mean_excess,
        "std_excess": std_excess,
        "sharpe_daily": sharpe_daily,
        "sharpe_annual": sharpe_annual,
        "t_stat": t_stat,
    }


def rolling_sharpe(excess: pd.Series, window: int) -> pd.Series:
    """Compute rolling annualized Sharpe over a given window (in trading days).
    Sharpe_t = (mean(excess over window) / std(excess over window)) * sqrt(252)
    """
    xs = excess.astype(float)
    roll_mean = xs.rolling(window=window, min_periods=window).mean()
    roll_std = xs.rolling(window=window, min_periods=window).std(ddof=1)
    rs = roll_mean / roll_std
    rs_annual = rs * math.sqrt(TRADING_DAYS)
    return rs_annual


def make_chart(dates: pd.Series, roll_sharpe: pd.Series, out_path: str):
    """Save a simple PNG of rolling Sharpe. One plot, no specified colors or styles."""
    plt.figure(figsize=(10, 4))
    plt.plot(dates, roll_sharpe)
    plt.title("Rolling Annualized Sharpe")
    plt.xlabel("Date")
    plt.ylabel("Sharpe (annualized)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"strategy_return", "risk_free_daily"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must include {required}")
    return df.sort_values("date").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Enhanced Sharpe Ratio Demo (Educational)")
    parser.add_argument("csv_path", help="Path to CSV with columns: date, strategy_return, risk_free_daily")
    parser.add_argument("--window", type=int, default=60, help="Rolling window in trading days (default: 60)")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save rolling Sharpe PNG")
    parser.add_argument("--no-chart", action="store_true", help="Disable chart saving even if --out is provided")
    args = parser.parse_args()

    df = load_csv(args.csv_path)
    excess = df["strategy_return"] - df["risk_free_daily"]

    stats = compute_basic_stats(excess)

    print("Sharpe Ratio Demo Plus (Educational)")
    print("-" * 40)
    print("Plain-English interpretation:")
    print("• We computed daily excess returns = strategy_return - risk_free_daily.")
    print("• Signal = average excess return; Noise = standard deviation of excess returns.")
    print("• Daily Sharpe = Signal / Noise; Annualized Sharpe ≈ Daily Sharpe × √252 (assumes iid).")
    print("• t-stat is a crude signal-strength check for mean excess (assumes iid/normal; educational only).")
    print()
    print(f"Observations (n):        {stats['n']}")
    print(f"Mean excess (daily):     {stats['mean_excess']:.8f}" if not np.isnan(stats['mean_excess']) else "Mean excess (daily):     nan")
    print(f"Std excess  (daily):     {stats['std_excess']:.8f}" if not np.isnan(stats['std_excess']) else "Std excess  (daily):     nan")
    print(f"Sharpe (daily):          {stats['sharpe_daily']:.4f}" if not np.isnan(stats['sharpe_daily']) else "Sharpe (daily):          nan")
    print(f"Sharpe (annualized):     {stats['sharpe_annual']:.4f}" if not np.isnan(stats['sharpe_annual']) else "Sharpe (annualized):     nan")
    print(f"Crude t-stat (mean ex.): {stats['t_stat']:.4f}" if not np.isnan(stats['t_stat']) else "Crude t-stat (mean ex.): nan")
    print()

    # Rolling Sharpe
    window = max(5, int(args.window))  # guard against tiny values
    roll = rolling_sharpe(excess, window)
    available = roll.dropna().shape[0]

    print(f"Rolling Sharpe (annualized) over window={window} trading days:")
    if available == 0:
        print("• Not enough data to compute rolling Sharpe for the chosen window.")
    else:
        print(f"• Available points: {available}")
        print(f"• Last rolling Sharpe: {roll.dropna().iloc[-1]:.4f}")

    # Chart (optional)
    if args.out and not args.no_chart:
        try:
            make_chart(df["date"], roll, args.out)
            print(f"Saved rolling Sharpe chart to: {args.out}")
        except Exception as e:
            print(f"Could not save chart: {e}")

if __name__ == '__main__':
    main()
