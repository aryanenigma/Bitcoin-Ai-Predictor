import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

INTERVAL = "15m"
END_DATE = dt.datetime.now()
START_DATE = END_DATE - dt.timedelta(days=59)
CAPITAL = 10000
TP_PCT = 12
SL_PCT = 5
FEE_PCT = 0.004
EMA_TOLERANCE = 0.005
print(f"Fetching {INTERVAL} BTC data from {START_DATE.date()} → {END_DATE.date()} …")

df = yf.download(
    "BTC-USD",
    start=START_DATE,
    end=END_DATE,
    interval=INTERVAL,
    progress=False,
    auto_adjust=True,
)
def ema(series, span=10):
    """Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, length=14):
    """Relative Strength Index"""
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    r = 100 - (100 / (1 + rs))
    return r.fillna(50)

# ---------- Download BTC Data ----------
print("Fetching 15m BTC data …")

df = yf.download("BTC-USD", start=START_DATE, end=END_DATE,
                 interval=INTERVAL, progress=False, auto_adjust=True)
df.to_csv("btc_15m_data.csv")

# --- Normalize multi-index columns from yfinance ---
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)  # flatten if multi-index

# Lowercase and clean column names
df.columns = [str(c).lower() for c in df.columns]

# --- Validate essential columns ---
required_cols = {"open", "high", "low", "close", "volume"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"❌ Missing columns in data: {missing_cols}")

# Add indicators
df["ema10"] = ema(df["close"], 10)
df["rsi14"] = rsi(df["close"], 14)
df.dropna(inplace=True)

# ---------- Utility ----------
def day_str(ts):
    return ts.strftime("%Y-%m-%d")
# ---------- STRATEGY SIMULATION ----------
balance = CAPITAL
open_trade = None
trades = []
trades_per_day = {}

for i in range(2, len(df)):
    row, prev = df.iloc[i], df.iloc[i - 1]
    timestamp = row.name
    price = float(row["close"])
    ds = day_str(timestamp)
    trades_today = trades_per_day.get(ds, 0)

    # ---------- EXIT ----------
    if open_trade:
        direction = open_trade["dir"]
        if direction == "LONG":
            if price <= open_trade["sl"]:  # stop
                pnl = (price - open_trade["entry"]) / open_trade["entry"]
            elif price >= open_trade["tp"]:  # target
                pnl = (price - open_trade["entry"]) / open_trade["entry"]
            else:
                pnl = None
        else:  # SHORT
            if price >= open_trade["sl"]:
                pnl = (open_trade["entry"] - price) / open_trade["entry"]
            elif price <= open_trade["tp"]:
                pnl = (open_trade["entry"] - price) / open_trade["entry"]
            else:
                pnl = None

        if pnl is not None:
            balance *= (1 + pnl - FEE_PCT)
            trades.append({
                "entry_time": open_trade["time"],
                "exit_time": timestamp,
                "dir": direction,
                "entry": open_trade["entry"],
                "exit": price,
                "pnl_pct": pnl * 100,
                "status": "WIN" if pnl > 0 else "LOSS"
            })
            open_trade = None
            continue

    # ---------- ENTRY ----------
    if open_trade or trades_today >= 2:
        continue

    prev_open  = float(prev["open"])
    prev_close = float(prev["close"])
    prev_high  = float(prev["high"])
    prev_low   = float(prev["low"])
    curr_open  = float(row["open"])
    curr_close = float(row["close"])
    ema10 = float(row["ema10"])
    rsi14 = float(row["rsi14"])

    is_prev_bear = prev_close < prev_open
    is_prev_bull = prev_close > prev_open
    is_curr_bull = curr_close > curr_open
    is_curr_bear = curr_close < curr_open

    direction = None
    entry_price = None

    # --- LONG breakout ---
    if is_prev_bear and is_curr_bull and (curr_close > prev_high or curr_close > prev_close * 1.0003):
        direction = "LONG"
        entry_price = curr_close

    # --- SHORT breakout ---
    elif is_prev_bull and is_curr_bear and (curr_close < prev_low or curr_close < prev_close * 0.9997):
        direction = "SHORT"
        entry_price = curr_close

    if direction and entry_price:
        price_vs_ema_ok = abs(entry_price - ema10) / ema10 <= EMA_TOLERANCE
        if direction == "LONG":
            if (entry_price < ema10 and not price_vs_ema_ok) or (rsi14 >= 78):
                continue
        else:  # SHORT
            if (entry_price > ema10 and not price_vs_ema_ok) or (rsi14 <= 22):
                continue

        # Create trade
        if direction == "LONG":
            sl = entry_price * (1 - SL_PCT / 100)
            tp = entry_price * (1 + TP_PCT / 100)
        else:
            sl = entry_price * (1 + SL_PCT / 100)
            tp = entry_price * (1 - TP_PCT / 100)

        open_trade = {
            "dir": direction,
            "entry": entry_price,
            "sl": sl,
            "tp": tp,
            "time": timestamp
        }
        trades_per_day[ds] = trades_today + 1

# ---------- CLOSE LAST OPEN TRADE ----------
last_price = float(df["close"].iloc[-1])
if open_trade:
    direction = open_trade["dir"]
    pnl = (last_price - open_trade["entry"]) / open_trade["entry"] if direction == "LONG" else (open_trade["entry"] - last_price) / open_trade["entry"]
    balance *= (1 + pnl - FEE_PCT)
    trades.append({
        "entry_time": open_trade["time"],
        "exit_time": df.index[-1],
        "dir": direction,
        "entry": open_trade["entry"],
        "exit": last_price,
        "pnl_pct": pnl * 100,
        "status": "CLOSED"
    })
# ---------- RESULTS ----------
if not trades:
    print("No trades executed.")
    exit()

df_trades = pd.DataFrame(trades)
df_trades["equity"] = CAPITAL * (1 + (df_trades["pnl_pct"] / 100 - FEE_PCT)).cumprod()

gross_profit = balance - CAPITAL
net_after_tax = balance  # no tax now
win_rate = (df_trades["pnl_pct"] > 0).mean() * 100
wins = sum(df_trades["status"] == "WIN")
losses = sum(df_trades["status"] == "LOSS")

print("\n===== SMART BTC BREAKOUT — 1Y BACKTEST =====")
print(f"Period: {START_DATE} → {END_DATE} ({INTERVAL})")
print(f"Total trades: {len(trades)} | Win-rate: {win_rate:.2f}%")
print(f"Final balance: ${balance:,.2f}")
print(f"Net return: {(balance / CAPITAL - 1) * 100:.2f}%")

# ---------- EXPORT CSV ----------
# --- IMPORTANT: REMOVE ALL OLD plt.show() CODE ---
# Start the script from the 'EXPORT CSV' section onwards.

# ---------- EXPORT CSV ----------
df_trades.to_csv("trades_log.csv", index=False)
print("Saved detailed trade log → trades_log.csv")

import os
import matplotlib.pyplot as plt

OUTPUT_DIR = "analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)   # <- FIXED

# 1️⃣ EQUITY CURVE
plt.figure(figsize=(12, 6))
plt.plot(df_trades["exit_time"], df_trades["equity"], linewidth=2, color="#38bdf8")
plt.title("Smart BTC Breakout — Equity Curve", color="#e2e8f0")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "equity_curve_small.png"), dpi=100)
print("Saved Equity Curve → analysis/equity_curve_small.png")

# 2️⃣ PIE CHART
plt.figure(figsize=(5, 5))
wins = sum(df_trades["status"] == "WIN")
losses = sum(df_trades["status"] == "LOSS")
plt.pie([wins, losses], labels=["Wins", "Losses"], autopct="%1.1f%%", startangle=90)
plt.title("Trade Outcome Distribution", color="#e2e8f0")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "winloss_pie_small.png"), dpi=100)
print("Saved Win/Loss Pie → analysis/winloss_pie_small.png")
