
#!/usr/bin/env python3
"""Risk Attribution Skeleton (Educational)

Provide a CSV with columns: date, ticker, return
The script estimates simple variance contributions by ticker.
"""
import sys, pandas as pd

def main(path):
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "ticker", "return"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must include {required}")
    by_ticker = df.groupby("ticker")["return"].var().sort_values(ascending=False)
    print("Variance by ticker (higher = more contribution to total volatility):")
    print(by_ticker.to_string())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/risk_attribution.py path/to/returns_by_ticker.csv")
        sys.exit(1)
    main(sys.argv[1])
