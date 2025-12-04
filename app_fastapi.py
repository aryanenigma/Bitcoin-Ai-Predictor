# app_fastapi.py — BTC AI Dashboard (with strategy_custom) — Hardened & fixed
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import subprocess, os, sys, threading, requests, numpy as np, pandas as pd, datetime as dt, math, traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import subprocess, os, sys, threading, requests, numpy as np, pandas as pd, datetime as dt, math, traceback

# ------------------- Create FastAPI app -------------------
app = FastAPI(title="BTC AI Dashboard API", version="2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ✅ FRONTEND SETUP (serves HTML, JS, CSS directly from folder)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = BASE_DIR  # all frontend files are here
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ----------------------------------------------------------------
# Serve HTML pages
# ----------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    page = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(page):
        return FileResponse(page)
    return HTMLResponse("<h3>⚠️ index.html not found in project folder.</h3>")

@app.get("/analytics", response_class=HTMLResponse)
async def serve_analytics():
    page = os.path.join(FRONTEND_DIR, "analytics.html")
    if os.path.exists(page):
        return FileResponse(page)
    return HTMLResponse("<h3>⚠️ analytics.html not found.</h3>")

@app.get("/strategy", response_class=HTMLResponse)
async def serve_strategy():
    page = os.path.join(FRONTEND_DIR, "strategy.html")
    if os.path.exists(page):
        return FileResponse(page)
    return HTMLResponse("<h3>⚠️ strategy.html not found.</h3>")

# ----------------------------------------------------------------
# Serve JS and CSS directly from root folder
# ----------------------------------------------------------------
@app.get("/{filename}", response_class=HTMLResponse)
async def serve_assets(filename: str):
    """
    Serves script.js, style.css, and any other file in main folder.
    """
    file_path = os.path.join(FRONTEND_DIR, filename)
    if os.path.exists(file_path):
        # detect content type
        if filename.endswith(".js"):
            return FileResponse(file_path, media_type="application/javascript")
        elif filename.endswith(".css"):
            return FileResponse(file_path, media_type="text/css")
        elif filename.endswith(".html"):
            return FileResponse(file_path, media_type="text/html")
        else:
            return FileResponse(file_path)
    return HTMLResponse(f"<h3>⚠️ {filename} not found in folder.</h3>")

@app.get("/stats")
def get_stats():
    return {
        "balance": 10000,
        "winrate": 55,
        "best": 12.5,
        "top_trades": []
    }

# ---------- helper: valid Binance intervals + fallback ----------
VALID_BINANCE_INTERVALS = {
    "1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"
}

def normalize_interval(interval: str) -> str:
    """
    Accepts user-provided interval (e.g., "10m") and returns a Binance-supported interval.
    If unsupported, falls back to nearest valid interval (15m) and logs a warning.
    """
    if not interval:
        return "15m"
    interval = interval.strip()
    if interval in VALID_BINANCE_INTERVALS:
        return interval
    # Common user inputs mapping (support 10m -> 15m)
    mapping = {
        "10m": "15m",
        "60m": "1h",
        "60": "1h",
        "15": "15m"
    }
    if interval in mapping:
        print(f"[WARN] interval '{interval}' replaced with '{mapping[interval]}' for Binance API compatibility.")
        return mapping[interval]
    # try small normalization: "10" -> "10m" but 10m isn't valid, so fallback
    if interval.endswith("m") and interval[:-1].isdigit():
        # fallback to 15m for any unsupported minute interval
        print(f"[WARN] Unsupported minute interval '{interval}' -> falling back to '15m'.")
        return "15m"
    # default fallback
    print(f"[WARN] Unknown interval '{interval}' -> falling back to '15m'.")
    return "15m"

# ---------- Utility indicator helpers ----------
def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14):
    # safe RSI that avoids division by zero
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    # avoid zero division
    ma_down_safe = ma_down.replace(0, np.nan)
    rs = ma_up / ma_down_safe
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral default where calculation isn't possible

# ---------- /api/combined ----------
@app.get("/api/combined")
def api_combined(interval: str = "15m", limit: int = 500):
    try:
        interval = normalize_interval(interval)
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit={limit}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        # validate structure
        if not isinstance(data, list) or len(data) == 0:
            return JSONResponse({"error": "Empty candle data from Binance"}, status_code=500)

        candles = []
        for k in data:
            # each k is array - validate length
            if len(k) < 6:
                continue
            candles.append({
                "time": int(k[0]) // 1000,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5])
            })
        news_agg = {"score": 0.12, "label": "positive"}
        news = [
            {"title": "Bitcoin gaining momentum", "link": "#", "sentiment": "positive"},
            {"title": "Crypto market stabilizing", "link": "#", "sentiment": "neutral"}
        ]
        return JSONResponse({"candles": candles, "news_agg": news_agg, "news": news})
    except requests.exceptions.RequestException as re:
        tb = traceback.format_exc()
        print("[ERROR] Binance request failed:", tb)
        return JSONResponse({"error": f"Binance request error: {str(re)}"}, status_code=500)
    except Exception as e:
        tb = traceback.format_exc()
        print("[ERROR] api_combined crashed:", tb)
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------- Strategy endpoint ----------
@app.get("/api/strategy_custom")
def strategy_custom(
    interval: str = "15m",
    limit: int = 500,
    profit_target_percent: float = 4.0,
    stop_loss_percent: float = 10.0
):
    """
    Returns simulated trades and data for 'Smart BTC Breakout' strategy.
    Implements:
      - interval normalization (auto-fallback for unsupported intervals like '10m')
      - RSI + ema10 indicators
      - avoid near SR, avoid repeating pattern, max 2 trades/day
      - relaxed entry filters to produce actionable trades
    """
    try:
        # normalize interval to something Binance accepts
        safe_interval = normalize_interval(interval)
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={safe_interval}&limit={limit}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        raw = r.json()
        if not isinstance(raw, list) or len(raw) == 0:
            return JSONResponse({"error": "No candle data returned from Binance"}, status_code=500)

        # build DataFrame (ignore invalid rows)
        rows = []
        for k in raw:
            if len(k) < 6:
                continue
            rows.append({
                "time": int(k[0]) // 1000,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5])
            })
        df = pd.DataFrame(rows)
        if df.empty or len(df) < 10:
            return JSONResponse({"error": "Insufficient candle data to run strategy (need at least 10 rows)."}, status_code=500)

        # indicators (safe)
        df["ema10"] = ema(df["close"], span=21)
        df["rsi14"] = rsi(df["close"], length=14)

        trades = []
        open_trade = None
        initial_balance = 10000.0
        balance = initial_balance
        trades_per_day = {}

        def day_str(ts):
            return dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

        # relaxed S/R check: larger tolerance so we don't reject too many signals
        def near_recent_sr(idx, price, lookback=5, pct_threshold=0.01):
            start = max(0, idx - lookback)
            highs = df["high"].iloc[start:idx]
            lows = df["low"].iloc[start:idx]
            for h in highs:
                if h == 0: continue
                if abs(price - h) / h <= pct_threshold:
                    return True
            for l in lows:
                if l == 0: continue
                if abs(price - l) / l <= pct_threshold:
                    return True
            return False

        # iterate candles
        for i in range(2, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            timestamp = int(row["time"])
            ds = day_str(timestamp)
            trades_today = trades_per_day.get(ds, 0)

            # check exits first if we have an open trade
            if open_trade:
                current_price = row["close"]
                direction = open_trade["direction"]

                if direction == "LONG":
                    # stop loss
                    if current_price <= open_trade["stop"]:
                        pnl = (current_price - open_trade["entry"]) / open_trade["entry"]
                        trades.append({
                            "entry_time": open_trade["entry_time"],
                            "exit_time": timestamp,
                            "entry_index": open_trade.get("entry_index"),
                            "exit_index": i,
                            "entry": round(open_trade["entry"], 2),
                            "exit": round(current_price, 2),
                            "pnl_percent": round(pnl * 100, 2),
                            "status": "LOSS",
                            "direction": "LONG"
                        })
                        balance *= (1 + pnl)
                        open_trade = None
                        continue
                    # take profit
                    if current_price >= open_trade["target"]:
                        pnl = (current_price - open_trade["entry"]) / open_trade["entry"]
                        trades.append({
                            "entry_time": open_trade["entry_time"],
                            "exit_time": timestamp,
                            "entry_index": open_trade.get("entry_index"),
                            "exit_index": i,
                            "entry": round(open_trade["entry"], 2),
                            "exit": round(current_price, 2),
                            "pnl_percent": round(pnl * 100, 2),
                            "status": "WIN",
                            "direction": "LONG"
                        })
                        balance *= (1 + pnl)
                        open_trade = None
                        continue

                else:  # SHORT
                    # stop loss for short (price moves above stop)
                    if current_price >= open_trade["stop"]:
                        pnl = (open_trade["entry"] - current_price) / open_trade["entry"]
                        trades.append({
                            "entry_time": open_trade["entry_time"],
                            "exit_time": timestamp,
                            "entry_index": open_trade.get("entry_index"),
                            "exit_index": i,
                            "entry": round(open_trade["entry"], 2),
                            "exit": round(current_price, 2),
                            "pnl_percent": round(pnl * 100, 2),
                            "status": "LOSS",
                            "direction": "SHORT"
                        })
                        balance *= (1 + pnl)
                        open_trade = None
                        continue
                    # take profit for short (price reaches target lower)
                    if current_price <= open_trade["target"]:
                        pnl = (open_trade["entry"] - current_price) / open_trade["entry"]
                        trades.append({
                            "entry_time": open_trade["entry_time"],
                            "exit_time": timestamp,
                            "entry_index": open_trade.get("entry_index"),
                            "exit_index": i,
                            "entry": round(open_trade["entry"], 2),
                            "exit": round(current_price, 2),
                            "pnl_percent": round(pnl * 100, 2),
                            "status": "WIN",
                            "direction": "SHORT"
                        })
                        balance *= (1 + pnl)
                        open_trade = None
                        continue

            # evaluate new entry if none open and daily limit not reached
            if (not open_trade) and (trades_today < 2):
                is_prev_bear = prev["close"] < prev["open"]
                is_prev_bull = prev["close"] > prev["open"]
                is_curr_bull = row["close"] > row["open"]
                is_curr_bear = row["close"] < row["open"]

                direction = None
                breakout = False
                entry_price = None

                # More permissive reversal/breakout: either break previous high/low OR close beyond previous close
                if is_prev_bear and is_curr_bull and (row["close"] > prev["high"] or row["close"] > prev["close"] * 1.0003):
                    direction = "LONG"
                    breakout = True
                    entry_price = row["close"]
                elif is_prev_bull and is_curr_bear and (row["close"] < prev["low"] or row["close"] < prev["close"] * 0.9997):
                    direction = "SHORT"
                    breakout = True
                    entry_price = row["close"]

                if breakout and entry_price:
                    # avoid near recent S/R (less strict now)
                    if near_recent_sr(i, entry_price, lookback=5, pct_threshold=0.01):
                        continue

                    # indicator confirmation (relaxed)
                    ema10 = df["ema10"].iloc[i]
                    rsi14 = df["rsi14"].iloc[i]
                    if pd.isna(ema10) or pd.isna(rsi14):
                        continue

                    # allow entry if price is around EMA (±0.5%) and RSI not extreme
                    ema_tolerance = 0.005
                    price_vs_ema_ok = abs(entry_price - ema10) / ema10 <= ema_tolerance
                    if direction == "LONG":
                        if (entry_price < ema10 and not price_vs_ema_ok) or (rsi14 >= 78):
                            continue
                    else:  # SHORT
                        if (entry_price > ema10 and not price_vs_ema_ok) or (rsi14 <= 22):
                            continue

                    # create trade with SL and target
                    if direction == "LONG":
                        sl = entry_price * (1 - stop_loss_percent / 100.0)
                        target = entry_price * (1 + profit_target_percent / 100.0)
                    else:
                        sl = entry_price * (1 + stop_loss_percent / 100.0)
                        target = entry_price * (1 - profit_target_percent / 100.0)

                    open_trade = {
                        "entry": float(entry_price),
                        "entry_time": timestamp,
                        "entry_index": i,
                        "stop": float(sl),
                        "target": float(target),
                        "direction": direction
                    }

                    # record daily count when trade opened (use local day)
                    trades_per_day[ds] = trades_today + 1

        # If there's an open trade still at the end of the loop, close it at the last candle price
        if open_trade:
            last_idx = len(df) - 1
            last_row = df.iloc[last_idx]
            last_price = last_row["close"]
            direction = open_trade["direction"]
            if direction == "LONG":
                pnl = (last_price - open_trade["entry"]) / open_trade["entry"]
            else:
                pnl = (open_trade["entry"] - last_price) / open_trade["entry"]
            trades.append({
                "entry_time": open_trade["entry_time"],
                "exit_time": int(last_row["time"]),
                "entry_index": open_trade.get("entry_index"),
                "exit_index": last_idx,
                "entry": round(open_trade["entry"], 2),
                "exit": round(last_price, 2),
                "pnl_percent": round(pnl * 100, 2),
                "status": "CLOSED",
                "direction": direction
            })
            balance *= (1 + pnl)
            open_trade = None

        # compute equity curve from trades
        cur_balance = initial_balance
        equity_curve = []
        for t in trades:
            pnl = t["pnl_percent"] / 100.0
            cur_balance *= (1 + pnl)
            equity_curve.append({"time": t["exit_time"], "value": round(cur_balance, 2)})

        win_trades = [t for t in trades if str(t.get("status", "")).upper() == "WIN"]
        win_rate = (len(win_trades) / len(trades) * 100) if trades else 0.0

        candles_out = df[["time", "open", "high", "low", "close", "volume"]].to_dict(orient="records")
        return JSONResponse({
            "candles": candles_out,
            "trades": trades,
            "initial_balance": initial_balance,
            "final_balance": round(balance, 2),
            "total_return_percent": round((balance / initial_balance - 1) * 100, 2),
            "n_trades": len(trades),
            "win_rate_percent": round(win_rate, 2),
            "equity_curve": equity_curve,
            "meta": {
                "requested_interval": interval,
                "used_interval": safe_interval,
                "profit_target_percent": profit_target_percent,
                "stop_loss_percent": stop_loss_percent
            }
        })
    except requests.exceptions.RequestException as re:
        tb = traceback.format_exc()
        print("[ERROR] Binance request failed in strategy_custom:", tb)
        return JSONResponse({"error": f"Binance request error: {str(re)}"}, status_code=500)
    except Exception as e:
        tb = traceback.format_exc()
        print("[ERROR] strategy_custom crashed:", tb)
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)
@app.get("/api/strategy_projection")
def strategy_projection(interval: str = "15m", limit: int = 500):
    """
    Simulate each potential trade from your rule and show realistic profit/loss results.
    """
    try:
        safe_interval = normalize_interval(interval)
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={safe_interval}&limit={limit}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        raw = r.json()
        df = pd.DataFrame([{
            "time": int(k[0]) // 1000,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
        } for k in raw])

        if df.empty:
            return JSONResponse({"error": "No candles"}, status_code=500)

        df["ema10"] = ema(df["close"], span=21)
        df["rsi14"] = rsi(df["close"], length=14)

        signals = []
        wins = losses = 0
        total_pnl = 0.0

        for i in range(2, len(df) - 1):
            row = df.iloc[i]
            nxt = df.iloc[i + 1]
            prev = df.iloc[i - 1]

            ema10 = df["ema10"].iloc[i]
            rsi14 = df["rsi14"].iloc[i]
            if pd.isna(ema10) or pd.isna(rsi14):
                continue

            # your rule — buy when opposite candle breaks previous bearish one
            is_prev_bear = prev["close"] < prev["open"]
            is_curr_bull = row["close"] > row["open"]

            if is_prev_bear and is_curr_bull and row["close"] > prev["high"] and row["close"] > ema10 and rsi14 < 70:
                entry = row["close"]
                stop = entry * 0.90   # 10 % stop-loss
                target = entry * 1.04 # 4 % profit
                # look-ahead to next candle to see which level hit first
                hit_target = nxt["high"] >= target
                hit_stop = nxt["low"] <= stop
                if hit_target and not hit_stop:
                    pnl = (target - entry) / entry * 100
                    wins += 1
                    result = "WIN"
                elif hit_stop and not hit_target:
                    pnl = (stop - entry) / entry * 100
                    losses += 1
                    result = "LOSS"
                else:
                    # if both or none hit, close at next close
                    pnl = (nxt["close"] - entry) / entry * 100
                    result = "NEUTRAL"

                total_pnl += pnl
                signals.append({
                    "time": int(row["time"]),
                    "entry": round(entry, 2),
                    "target": round(target, 2),
                    "stop": round(stop, 2),
                    "exit": round(nxt["close"], 2),
                    "pnl_percent": round(pnl, 2),
                    "result": result,
                })

        total_signals = len(signals)
        win_rate = (wins / total_signals * 100) if total_signals else 0
        avg_pnl = (total_pnl / total_signals) if total_signals else 0
        cum_pnl = sum([s["pnl_percent"] for s in signals])

        return JSONResponse({
            "signals": signals,
            "summary": {
                "total_signals": total_signals,
                "win_rate": round(win_rate, 2),
                "avg_pnl_per_trade": round(avg_pnl, 2),
                "total_pnl_percent": round(cum_pnl, 2),
            }
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =========================================================
# /api/analysis_list — list generated analysis images
# =========================================================
@app.get("/api/analysis_list")
def analysis_list():
    try:
        files = []
        for fname in os.listdir(ANALYSIS_DIR):
            fpath = os.path.join(ANALYSIS_DIR, fname)
            if os.path.isfile(fpath):
                files.append(fname)
        return JSONResponse({"analysis_dir": ANALYSIS_DIR, "files": files})
    except Exception as e:
        tb = traceback.format_exc()
        print("[ERROR] analysis_list crashed:", tb)
        return JSONResponse({"error": str(e)}, status_code=500)

# =========================================================
# /analysis/{filename} — serve generated charts
# =========================================================
@app.get("/analysis/{filename}")
async def get_analysis_image(filename: str):
    path = os.path.join(ANALYSIS_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": f"{filename} not found in analysis directory", "analysis_dir": ANALYSIS_DIR}, status_code=404)
    return FileResponse(path)

# =========================================================
# /api/retrain — retrain model + generate charts
# =========================================================
@app.post("/api/retrain")
async def retrain_model():
    def run_training():
        try:
            print("[INFO] Starting retraining...")
            subprocess.run([sys.executable, os.path.join(BASE_DIR, "train_model.py")], check=True)
            subprocess.run([sys.executable, os.path.join(BASE_DIR, "analyze_model.py")], check=True)
            print("[INFO] ✅ Retraining completed successfully!")
        except Exception as e:
            print(f"[ERROR] Retraining failed: {e}")
            print(traceback.format_exc())

    threading.Thread(target=run_training, daemon=True).start()
    return JSONResponse({
        "status": "success",
        "message": "Retraining started in background. Check console for progress."
    })

# =========================================================
# /api/refresh_visuals — regenerate analysis charts
# =========================================================
@app.post("/api/refresh_visuals")
async def refresh_visuals():
    try:
        # call analyze_model generate function if present
        try:
            from analyze_model import generate_analysis_charts
            generate_analysis_charts()
            # after generation, list files
            files = os.listdir(ANALYSIS_DIR)
            return JSONResponse({"status": "success", "message": "✅ Charts refreshed.", "files": files})
        except Exception as e:
            # fallback: if analyze_model missing, still respond with error detail
            tb = traceback.format_exc()
            print("[ERROR] refresh_visuals failed:", tb)
            return JSONResponse({"status": "error", "message": str(e), "traceback": tb}, status_code=500)
    except Exception as e:
        tb = traceback.format_exc()
        print("[ERROR] refresh_visuals crashed:", tb)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_fastapi:app", host="127.0.0.1", port=8000, reload=True)
