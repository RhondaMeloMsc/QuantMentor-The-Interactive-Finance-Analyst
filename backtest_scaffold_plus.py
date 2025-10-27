
#!/usr/bin/env python3
"""Backtest Scaffold Plus (Educational)

Adds:
- --fee-bps: trading cost in basis points per flip (entry/exit sides modeled via flips)
- --use-log-returns: use log returns instead of simple returns
- --exec open|close: execution price basis for return calc (default: close)
- Rolling Sharpe (annualized) with --window (default 60)
- Max drawdown calculation
- Mini risk summary: vol (ann), Sharpe (ann), max DD, trade count

CSV Requirements:
- Must include 'date' and:
  - 'close' (always required)
  - 'open' if you choose --exec open
"""
import argparse
import math
import sys
from typing import Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252

def compute_returns_from_price(series: pd.Series, use_log: bool) -> pd.Series:
    series = series.astype(float)
    if use_log:
        return np.log(series).diff()
    else:
        return series.pct_change()

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Toy signal: long today if yesterday's close-to-close return < 0."""
    out = df.copy()
    y_ret = out['close'].pct_change().shift(1)
    out['signal'] = (y_ret < 0).astype(int)  # 1 = long, 0 = flat
    return out

def apply_strategy(df: pd.DataFrame, price_col: str, fee_bps: float, use_log: bool) -> Tuple[pd.DataFrame, int]:
    """Apply position to returns based on chosen execution price column.
    Position for day t is yesterday's signal (no look-ahead). Costs charged per flip.
    Returns modified DataFrame and trade count.
    """
    out = df.copy()
    out['ret'] = compute_returns_from_price(out[price_col], use_log).fillna(0.0)
    pos = out['signal'].shift(1).fillna(0.0)
    out['strategy_ret'] = pos * out['ret']

    # flips: count position changes 0<->1
    flips = out['signal'].diff().abs().fillna(0.0)
    trade_count = int(flips.sum())  # each flip is an order; buy and sell both count

    # fee per flip (per-side bps)
    fee = fee_bps / 10000.0
    out['strategy_ret'] = out['strategy_ret'] - flips * fee

    return out, trade_count

def annualize_vol(std_daily: float) -> float:
    return std_daily * math.sqrt(TRADING_DAYS)

def annualized_sharpe(mean_daily: float, std_daily: float) -> float:
    if std_daily == 0:
        return np.nan
    return (mean_daily / std_daily) * math.sqrt(TRADING_DAYS)

def max_drawdown_from_returns(ret: pd.Series) -> float:
    """Compute max drawdown from a return series (fraction, negative number)."""
    eq = (1.0 + ret.fillna(0.0)).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min()) if len(dd) else np.nan

def rolling_sharpe_annualized(excess: pd.Series, window: int) -> pd.Series:
    m = excess.rolling(window, min_periods=window).mean()
    s = excess.rolling(window, min_periods=window).std(ddof=1)
    rs = m / s
    return rs * math.sqrt(TRADING_DAYS)

def main():
    parser = argparse.ArgumentParser(description="Backtest Scaffold Plus (Educational)")
    parser.add_argument("csv_path", help="Path to CSV with date, close (and open if --exec open)")
    parser.add_argument("--fee-bps", type=float, default=1.0, help="Trading cost per flip in bps (default: 1)")
    parser.add_argument("--use-log-returns", action="store_true", help="Use log returns instead of simple returns")
    parser.add_argument("--exec", dest="exec_price", choices=["open", "close"], default="close",
                        help="Execution price basis for returns (default: close)")
    parser.add_argument("--window", type=int, default=60, help="Rolling Sharpe window in trading days (default: 60)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    required = {"date", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must include columns: {required}")
    if args.exec_price == "open" and "open" not in df.columns:
        raise ValueError("CSV missing 'open' column required for --exec open")

    df = generate_signals(df)

    price_col = args.exec_price
    df, trade_count = apply_strategy(df, price_col, args.fee_bps, args.use_log_returns)

    # Excess return vs. a zero baseline (risk-free omitted to keep scaffold compact)
    excess = df['strategy_ret']  # educational scaffold

    mean_d = excess.mean()
    std_d = excess.std(ddof=1)
    vol_ann = annualize_vol(std_d)
    sharpe_ann = annualized_sharpe(mean_d, std_d)
    mdd = max_drawdown_from_returns(excess)

    # Rolling Sharpe
    window = max(5, int(args.window))
    roll_sharpe = rolling_sharpe_annualized(excess, window)
    last_roll = roll_sharpe.dropna().iloc[-1] if roll_sharpe.dropna().shape[0] else np.nan

    # Cumulative return
    cum_ret = (1.0 + excess).cumprod().iloc[-1] - 1.0 if len(excess) else np.nan

    print("Backtest Scaffold Plus (Educational)")
    print("-" * 40)
    print(f"Rows: {len(df)} | Exec basis: {price_col} | Returns: {'log' if args.use_log_returns else 'simple'}")
    print(f"Fee (per flip): {args.fee_bps:.2f} bps | Trade count (flips): {trade_count}")
    print()
    print("Mini Risk Summary")
    print("-----------------")
    print(f"Mean daily excess:    {mean_d:.8f}" if not np.isnan(mean_d) else "Mean daily excess:    nan")
    print(f"Std daily excess:     {std_d:.8f}" if not np.isnan(std_d) else "Std daily excess:     nan")
    print(f"Vol (annualized):     {vol_ann:.4f}" if not np.isnan(vol_ann) else "Vol (annualized):     nan")
    print(f"Sharpe (annualized):  {sharpe_ann:.4f}" if not np.isnan(sharpe_ann) else "Sharpe (annualized):  nan")
    print(f"Max Drawdown:         {mdd:.2%}" if not np.isnan(mdd) else "Max Drawdown:         nan")
    print(f"Cumulative return:    {cum_ret:.2%}" if not np.isnan(cum_ret) else "Cumulative return:    nan")
    print()
    print(f"Rolling Sharpe (ann.) window={window} → last: {last_roll:.4f}" if not np.isnan(last_roll) else f"Rolling Sharpe (ann.) window={window} → insufficient data")

if __name__ == "__main__":
    main()
