
#!/usr/bin/env python3
"""Sharpe Ratio Demo (Educational)

Usage:
    python scripts/sharpe_demo.py examples/sample_returns.csv

The script calculates an approximate Sharpe ratio using daily returns.
It prints plain-English explanations and the numeric result.
"""
import sys
import pandas as pd
import numpy as np

def main(path):
    df = pd.read_csv(path, parse_dates=["date"])
    if not {"strategy_return", "risk_free_daily"}.issubset(df.columns):
        raise ValueError("CSV must include 'strategy_return' and 'risk_free_daily'.")

    # Excess daily returns
    excess = df["strategy_return"] - df["risk_free_daily"]

    mean_excess = excess.mean()
    std_excess = excess.std(ddof=1)  # sample std
    sharpe_daily = mean_excess / std_excess if std_excess != 0 else np.nan

    # Annualization (approx using 252 trading days)
    sharpe_annual = sharpe_daily * np.sqrt(252) if not np.isnan(sharpe_daily) else np.nan

    print("Sharpe Ratio Demo (Educational)")
    print("-" * 32)
    print("Plain-English interpretation:")
    print("• We compared your strategy's daily return to a daily risk-free rate to get 'excess return'.")
    print("• We measured the average excess return and how much it wiggles day to day (its standard deviation).")
    print("• Sharpe ≈ average excess return ÷ its wiggliness (volatility).")
    print("• Higher is better; 0–1 is modest, 1–2 is solid, >2 is strong (context-dependent).")
    print()
    print(f"Mean excess (daily): {mean_excess:.6f}")
    print(f"Std excess  (daily): {std_excess:.6f}")
    print(f"Sharpe (daily):      {sharpe_daily:.3f}")
    print(f"Sharpe (annualized): {sharpe_annual:.3f}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/sharpe_demo.py examples/sample_returns.csv")
        sys.exit(1)
    main(sys.argv[1])
