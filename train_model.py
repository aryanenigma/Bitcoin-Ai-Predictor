# train_model.py
"""
Continuous retrainer for BTC classification model.

- Fetches latest 1h klines from Binance periodically
- Builds features, trains a classifier, saves model+scaler to disk
- Keeps a small log file 'train_log.txt' with retrain timestamps
"""

import time
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os
import traceback

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LIMIT = 1000            # number of candles to fetch each retrain
RETRAIN_EVERY = 60 * 10 # seconds (retrain every 10 minutes)
MODEL_PATH = "btc_model.joblib"
SCALER_PATH = "btc_scaler.joblib"
LOG_PATH = "train_log.txt"

def fetch_klines(limit=LIMIT):
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": limit}
    r = requests.get(BINANCE_KLINES, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame([{
        "time": pd.to_datetime(c[0], unit='ms'),
        "open": float(c[1]),
        "high": float(c[2]),
        "low": float(c[3]),
        "close": float(c[4]),
        "volume": float(c[5])
    } for c in data])
    df.set_index("time", inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def prepare_features(df):
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(24).std()  # 24h volatility
    df["rsi"] = compute_rsi(df["close"], 14)
    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_diff"] = (df["ema10"] - df["ema50"]) / df["ema50"]
    df["vol_change"] = df["volume"].pct_change()
    df["hour"] = df.index.hour
    # target: next close higher than current
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna().copy()
    return df

def train_and_save(df):
    features = ["return", "volatility", "rsi", "ema_diff", "vol_change", "hour"]
    X = df[features]
    y = df["target"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    model.fit(Xs, y)
    dump(model, MODEL_PATH)
    dump(scaler, SCALER_PATH)
    with open(LOG_PATH, "a") as f:
        f.write(f"{pd.Timestamp.utcnow()}: trained on {len(df)} samples\n")

def main_loop():
    print("🔁 Starting continuous trainer — will retrain every", RETRAIN_EVERY, "seconds")
    while True:
        try:
            df = fetch_klines()
            dff = prepare_features(df)
            if len(dff) < 200:
                print("Not enough data to train, sleeping...")
            else:
                train_and_save(dff)
                print(f"✅ Model trained & saved ({len(dff)} samples). Next retrain in {RETRAIN_EVERY}s")
        except Exception as e:
            print("⚠️ Error during train loop:", e)
            traceback.print_exc()
        time.sleep(RETRAIN_EVERY)

if __name__ == "__main__":
    main_loop()
