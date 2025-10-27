
#!/usr/bin/env python3
"""Backtest Scaffold (Educational)

This is a minimal, readable scaffold for a simple strategy backtest.
Fill in signal logic where indicated.
"""
import sys, pandas as pd

def generate_signals(df):
    # TODO: Replace with actual signal logic (e.g., mean reversion or momentum)
    # Example: buy when return is negative yesterday (toy logic)
    df = df.copy()
    df["signal"] = (df["close"].pct_change().shift(1) < 0).astype(int)  # 1 = long, 0 = flat
    return df

def apply_strategy(df, fee_bps=1):
    df = df.copy()
    df["ret"] = df["close"].pct_change().fillna(0)
    df["strategy_ret"] = df["signal"].shift(1).fillna(0) * df["ret"]
    # approximate trading cost:
    trades = df["signal"].diff().abs().fillna(0)
    df["strategy_ret"] -= trades * (fee_bps / 10000.0)
    return df

def main(path):
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    if not {"date", "close"}.issubset(df.columns):
        raise ValueError("CSV must include 'date' and 'close' columns.")
    df = generate_signals(df)
    df = apply_strategy(df)
    cum = (1 + df["strategy_ret"]).cumprod().iloc[-1] - 1
    print("Backtest Scaffold (Educational)")
    print("-" * 30)
    print(f"Cumulative return (toy logic): {cum:.2%}")
    print("Next steps: replace signal logic, add risk metrics, validate over multiple periods.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/backtest_scaffold.py path/to/ohlc.csv")
        sys.exit(1)
    main(sys.argv[1])
