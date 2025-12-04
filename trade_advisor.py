# trade_advisor.py
"""
Simple advisor:
- Fetches recent hourly candles
- Computes average return by hour-of-day: shows best hours to trade
- If trades_log.csv exists, prints top profitable trades and when they happened
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LIMIT = 1000
TRADE_LOG = "trades_log.csv"

def fetch_klines(limit=LIMIT):
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": limit}
    r = requests.get(BINANCE_KLINES, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame([{
        "time": pd.to_datetime(c[0], unit='ms'),
        "open": float(c[1]),
        "close": float(c[4])
    } for c in data])
    df.set_index("time", inplace=True)
    return df

def best_hours(df):
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["hour"] = df.index.hour
    grouped = df.groupby("hour")["ret"].mean().sort_values(ascending=False)
    return grouped

def top_trades(log=TRADE_LOG, top_n=10):
    if not os.path.exists(log):
        return None
    df = pd.read_csv(log, parse_dates=["entry_time", "exit_time"])
    df["pnl_pct"] = df["pnl_pct"].astype(float)
    df_sorted = df.sort_values("pnl_pct", ascending=False).head(top_n)
    return df_sorted

if __name__ == "__main__":
    print("Fetching historical hourly returns...")
    try:
        df = fetch_klines()
        bh = best_hours(df)
        print("\nAverage hourly returns (top 8):")
        print(bh.head(8).round(6).to_string())
    except Exception as e:
        print("Error fetching klines:", e)

    print("\nTop historical trades (from trades_log.csv):")
    tt = top_trades()
    if tt is None:
        print("No trades_log.csv found. Run the strategy to generate trades.")
    else:
        print(tt[["entry_time","exit_time","dir","entry","exit","pnl_pct","status"]].to_string(index=False))
