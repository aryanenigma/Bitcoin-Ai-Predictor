# utils.py
import pandas as pd
import numpy as np

def ema(df, period, col="close"):
    return df[col].ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df, n=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def apply_indicators(df):
    # expects df with columns: open, high, low, close, volume
    df = df.copy()
    df['ema_20'] = ema(df, 20)
    df['ema_50'] = ema(df, 50)
    df['rsi'] = rsi(df['close'], period=14)
    macd_line, macd_signal, macd_hist = macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    df['atr'] = atr(df, n=14)
    return df

def detect_breakout(curr, prev, min_close_move_pct=0.01):
    """
    Return "BUY"/"SELL"/None.
    Condition:
      - previous candle color is opposite
      - current candle breaks prev high (BUY) / prev low (SELL)
      - current candle close moves at least min_close_move_pct above/below breakout level
    """
    # prev_color: True if green (close > open)
    prev_green = prev['close'] > prev['open']
    curr_green = curr['close'] > curr['open']

    # require opposite color
    if prev_green and not curr_green:
        # potential SELL breakout (current red)
        if curr['close'] < prev['low']:
            move = (prev['low'] - curr['close']) / prev['low']
            if move >= min_close_move_pct:
                return "SELL"
    elif (not prev_green) and curr_green:
        # potential BUY breakout (current green)
        if curr['close'] > prev['high']:
            move = (curr['close'] - prev['high']) / prev['high']
            if move >= min_close_move_pct:
                return "BUY"
    return None

def has_nearby_s_r(df, level, lookback=12, thresh_pct=0.005):
    """
    Check whether a level is near any recent highs/lows.
    thresh_pct = percentage distance to consider 'near' (e.g. 0.005 = 0.5%)
    """
    look = df.tail(lookback)
    highs = look['high']
    lows = look['low']
    thresh = thresh_pct * level
    near_high = ((highs - level).abs() <= thresh).any()
    near_low = ((lows - level).abs() <= thresh).any()
    return near_high or near_low

def repeating_pattern(df, recent=6):
    """
    Return True if the last `recent` candles are mostly the same direction (no fresh structure).
    We consider repeating if >80% same color.
    """
    if len(df) < recent + 1:
        return False
    tail = df.tail(recent)
    greens = (tail['close'] > tail['open']).sum()
    reds = (tail['close'] < tail['open']).sum()
    total = len(tail)
    if greens / total >= 0.8 or reds / total >= 0.8:
        return True
    return False
