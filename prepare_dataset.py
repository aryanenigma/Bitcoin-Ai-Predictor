# prepare_dataset.py
import requests
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timezone

# --- API endpoints ---
CANDLES_URL = "http://127.0.0.1:8000/api/candles?interval=15m&timezone=UTC&limit=1000"
NEWS_URL = "http://127.0.0.1:8000/api/btc_news?count=30"

def compute_sentiment(text):
    """Convert text into a polarity score between -1 and +1."""
    return TextBlob(text).sentiment.polarity

print("📡 Fetching candle data...")
resp_c = requests.get(CANDLES_URL)
resp_c.raise_for_status()
candles = resp_c.json()
print(f"✅ Received {len(candles)} candles")

# Convert to DataFrame
df = pd.DataFrame(candles)
df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
df = df.sort_values("time").reset_index(drop=True)

# --- Fetch and process news ---
print("📰 Fetching BTC news...")
resp_n = requests.get(NEWS_URL)
resp_n.raise_for_status()
news_items = resp_n.json()

news_df = pd.DataFrame(news_items)
if not news_df.empty:
    news_df["pubDate"] = pd.to_datetime(news_df["pubDate"], errors="coerce", utc=True)
    news_df["sentiment_score"] = news_df["title"].apply(compute_sentiment)
    news_df = news_df.dropna(subset=["pubDate", "sentiment_score"])

    # Merge: assign each candle the avg sentiment of news within last 6h
    sentiment_series = []
    for t in df["time"]:
        mask = (news_df["pubDate"] >= (t - pd.Timedelta(hours=6))) & (news_df["pubDate"] <= t)
        subset = news_df.loc[mask, "sentiment_score"]
        avg = subset.mean() if not subset.empty else 0.0
        sentiment_series.append(avg)
    df["sentiment"] = sentiment_series
else:
    print("⚠️ No news fetched; filling sentiment with 0.")
    df["sentiment"] = 0.0

# --- Save combined dataset ---
df.to_csv("btc_candles.csv", index=False)
print("💾 Saved dataset with sentiment as btc_candles.csv")
print(df[["time", "open", "close", "sentiment"]].head())
