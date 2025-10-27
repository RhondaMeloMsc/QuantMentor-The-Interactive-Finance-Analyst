
#!/usr/bin/env python3
"""Backtest Scaffold Plus (Educational)

Adds:
- --fee-bps: trading cost in basis points per flip (entry/exit sides modeled via flips)
- --use-log-returns: use log returns instead of simple returns
- --exec open|close: execution price basis for return calc (default: close)
- --window: rolling Sharpe window (default 60)
- --riskfree-col: subtract a daily risk-free return column to form excess returns
- Risk summary: vol (ann), Sharpe (ann), Sortino (ann), max DD, hit rate, trade count, cumulative return
- Tiny CSV tearsheet with summary metrics via --tearsheet path.csv

CSV Requirements:
- Must include 'date' and:
  - 'close' (always required)
  - 'open' if you choose --exec open
- If using --riskfree-col NAME, CSV must include that column containing a daily return series.

Notes:
- Risk-free series is assumed to be a daily simple return rate. If using log returns for price, we still subtract the risk-free *simple* rate from the strategy's daily return for the excess calculation (educational simplification).
"""
import argparse
import math
import sys
from typing import Tuple, Dict

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

def annualized_sortino(mean_daily: float, downside_std_daily: float) -> float:
    if downside_std_daily == 0:
        return np.nan
    return (mean_daily / downside_std_daily) * math.sqrt(TRADING_DAYS)

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

def compute_summary_metrics(excess: pd.Series) -> Dict[str, float]:
    xs = excess.dropna().astype(float)
    n = xs.shape[0]
    mean_d = xs.mean()
    std_d = xs.std(ddof=1)
    # downside deviation uses only negative excess values relative to 0
    downside = xs[xs < 0.0]
    down_std_d = downside.std(ddof=1) if downside.shape[0] > 1 else np.nan
    vol_ann = annualize_vol(std_d) if not np.isnan(std_d) else np.nan
    sharpe_ann = annualized_sharpe(mean_d, std_d) if not np.isnan(std_d) else np.nan
    sortino_ann = annualized_sortino(mean_d, down_std_d) if not np.isnan(down_std_d) else np.nan
    mdd = max_drawdown_from_returns(xs)
    cum_ret = (1.0 + xs).cumprod().iloc[-1] - 1.0 if n else np.nan
    hit_rate = float((xs > 0).mean()) if n else np.nan
    return {
        "n": n,
        "mean_daily_excess": mean_d,
        "std_daily_excess": std_d,
        "downside_std_daily": down_std_d,
        "vol_annualized": vol_ann,
        "sharpe_annualized": sharpe_ann,
        "sortino_annualized": sortino_ann,
        "max_drawdown": mdd,
        "hit_rate": hit_rate,
        "cumulative_return": cum_ret,
    }

def main():
    parser = argparse.ArgumentParser(description="Backtest Scaffold Plus (Educational)")
    parser.add_argument("csv_path", help="Path to CSV with date, close (and open if --exec open)")
    parser.add_argument("--fee-bps", type=float, default=1.0, help="Trading cost per flip in bps (default: 1)")
    parser.add_argument("--use-log-returns", action="store_true", help="Use log returns instead of simple returns")
    parser.add_argument("--exec", dest="exec_price", choices=["open", "close"], default="close",
                        help="Execution price basis for returns (default: close)")
    parser.add_argument("--window", type=int, default=60, help="Rolling Sharpe window in trading days (default: 60)")
    parser.add_argument("--riskfree-col", type=str, default=None, help="Name of daily risk-free return column to subtract")
    parser.add_argument("--tearsheet", type=str, default=None, help="Optional path to save a tiny CSV tearsheet of summary metrics")
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

    # Build excess return series
    strategy = df['strategy_ret']
    if args.riskfree_col:
        if args.riskfree_col not in df.columns:
            raise ValueError(f"CSV missing risk-free column: '{args.riskfree_col}'")
        excess = strategy - df[args.riskfree_col].astype(float).fillna(0.0)
    else:
        excess = strategy

    # Summary metrics
    metrics = compute_summary_metrics(excess)

    # Rolling Sharpe
    window = max(5, int(args.window))
    roll_sharpe = rolling_sharpe_annualized(excess, window)
    last_roll = roll_sharpe.dropna().iloc[-1] if roll_sharpe.dropna().shape[0] else np.nan

    print("Backtest Scaffold Plus (Educational)")
    print("-" * 40)
    print(f"Rows: {len(df)} | Exec basis: {price_col} | Returns: {'log' if args.use_log_returns else 'simple'}")
    print(f"Fee (per flip): {args.fee_bps:.2f} bps | Trade count (flips): {trade_count}")
    if args.riskfree_col:
        print(f"Risk-free column: {args.riskfree_col}")
    print()
    print("Mini Risk Summary")
    print("-----------------")
    print(f"Observations (n):      {metrics['n']}")
    print(f"Mean daily excess:     {metrics['mean_daily_excess']:.8f}" if not np.isnan(metrics['mean_daily_excess']) else "Mean daily excess:     nan")
    print(f"Std daily excess:      {metrics['std_daily_excess']:.8f}" if not np.isnan(metrics['std_daily_excess']) else "Std daily excess:      nan")
    print(f"Downside std (daily):  {metrics['downside_std_daily']:.8f}" if not np.isnan(metrics['downside_std_daily']) else "Downside std (daily):  nan")
    print(f"Vol (annualized):      {metrics['vol_annualized']:.4f}" if not np.isnan(metrics['vol_annualized']) else "Vol (annualized):      nan")
    print(f"Sharpe (annualized):   {metrics['sharpe_annualized']:.4f}" if not np.isnan(metrics['sharpe_annualized']) else "Sharpe (annualized):   nan")
    print(f"Sortino (annualized):  {metrics['sortino_annualized']:.4f}" if not np.isnan(metrics['sortino_annualized']) else "Sortino (annualized):  nan")
    print(f"Max Drawdown:          {metrics['max_drawdown']:.2%}" if not np.isnan(metrics['max_drawdown']) else "Max Drawdown:          nan")
    print(f"Hit rate:              {metrics['hit_rate']:.2%}" if not np.isnan(metrics['hit_rate']) else "Hit rate:              nan")
    print(f"Cumulative return:     {metrics['cumulative_return']:.2%}" if not np.isnan(metrics['cumulative_return']) else "Cumulative return:     nan")
    print()
    print(f"Rolling Sharpe (ann.) window={window} → last: {last_roll:.4f}" if not np.isnan(last_roll) else f"Rolling Sharpe (ann.) window={window} → insufficient data")

    # Optional tearsheet CSV
    if args.tearsheet:
        tiny = pd.DataFrame([{k: metrics[k] for k in [
            "n","mean_daily_excess","std_daily_excess","downside_std_daily",
            "vol_annualized","sharpe_annualized","sortino_annualized",
            "max_drawdown","hit_rate","cumulative_return"
        ]}])
        tiny.to_csv(args.tearsheet, index=False)
        print(f"Tearsheet saved to: {args.tearsheet}")

if __name__ == "__main__":
    main()
