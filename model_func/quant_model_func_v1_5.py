# === File: htf_structure.py ===

# --- Imports ---
import pandas as pd
import numpy as np
import yfinance as yf

# --- Download Function ---
def download_crypto_data(symbol: str, 
                         ltf_interval: str = '15m', ltf_period: str = '60d',
                         htf_interval: str = '1h', htf_period: str = '6mo') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download crypto data for LTF and HTF using yfinance."""
    print(f"\U0001F4E5 Downloading {symbol} data...")
    
    ltf = yf.download(symbol, interval=ltf_interval, period=ltf_period, auto_adjust=False)
    htf = yf.download(symbol, interval=htf_interval, period=htf_period, auto_adjust=False)

    # Normalize column names
    ltf.columns = [col[0].title() if isinstance(col, tuple) else col.title() for col in ltf.columns]
    htf.columns = [col[0].title() if isinstance(col, tuple) else col.title() for col in htf.columns]

    return ltf, htf

# --- Structure Confidence Scoring ---
def calculate_structure_confidence(df, i, atr_window=14, vol_window=20) -> float:
    """Calculate structure confidence score at index i."""
    row = df.iloc[i]
    prev_row = df.iloc[i - 1]
    close = row["Close"]
    volume = row.get("Volume", 1)

    # ATR
    df["H-L"] = df["High"] - df["Low"]
    atr = df["H-L"].rolling(atr_window).mean().iloc[i]

    # Volume spike check
    avg_vol = df["Volume"].rolling(vol_window).mean().iloc[i]
    vol_score = 1 if volume > avg_vol else 0

    # Breakout size relative to ATR
    range_broken = abs(close - prev_row["Close"])
    break_score = min(range_broken / atr, 2) if atr else 0

    # Candle body score
    body = abs(row["Close"] - row["Open"])
    candle_range = row["High"] - row["Low"]
    body_score = body / candle_range if candle_range else 0

    # Weighted sum
    confidence = (0.4 * break_score) + (0.2 * body_score) + (0.2 * vol_score)
    return round(min(confidence, 1.0), 2)


def detect_htf_structure(df, left=2, right=2):
    """
    Detect HTF market structure (BOS and CHoCH) using swing highs/lows
    and assign structure confidence scores.

    Parameters:
        df (pd.DataFrame): HTF price data with 'High', 'Low', 'Close', 'Open', and 'Volume'.
        left (int): Bars to the left of swing point.
        right (int): Bars to the right of swing point.

    Returns:
        pd.DataFrame: DataFrame with added columns:
                      ['swing_high', 'swing_low', 'market_structure', 'structure_confidence']
    """
    df = df.copy()
    df["H-L"] = df["High"] - df["Low"]  # Needed for ATR
    df["structure_confidence"] = None
    df["market_structure"] = None

    # --- Detect swing points ---
    df["swing_high"] = df["High"][
        (df["High"].shift(left) < df["High"]) & 
        (df["High"].shift(-right) < df["High"])
    ]
    df["swing_low"] = df["Low"][
        (df["Low"].shift(left) > df["Low"]) & 
        (df["Low"].shift(-right) > df["Low"])
    ]

    # --- Internal scoring function ---
    def _calculate_confidence(i, atr_window=14, vol_window=20):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        atr = df["H-L"].rolling(atr_window).mean().iloc[i]
        avg_vol = df["Volume"].rolling(vol_window).mean().iloc[i]
        volume = row.get("Volume", 1)
        vol_score = 1 if volume > avg_vol else 0

        range_broken = abs(row["Close"] - prev["Close"])
        break_score = min(range_broken / atr, 2) if atr else 0

        body = abs(row["Close"] - row["Open"])
        candle_range = row["High"] - row["Low"]
        body_score = body / candle_range if candle_range else 0

        score = (0.4 * break_score) + (0.2 * body_score) + (0.2 * vol_score)
        return round(min(score, 1.0), 2)

    # --- Structure detection logic ---
    last_trend = None
    last_high = None
    last_low = None

    for i in range(len(df)):
        row = df.iloc[i]

        # Update most recent swing high/low
        if not pd.isna(row["swing_high"]):
            last_high = row["swing_high"]
        if not pd.isna(row["swing_low"]):
            last_low = row["swing_low"]

        # Skip early rows
        if i < max(left, right):
            continue

        close = row["Close"]

        if last_trend is None:
            if last_high and close > last_high:
                df.at[df.index[i], "market_structure"] = "BOS"
                df.at[df.index[i], "structure_confidence"] = _calculate_confidence(i)
                last_trend = "bullish"
            elif last_low and close < last_low:
                df.at[df.index[i], "market_structure"] = "BOS"
                df.at[df.index[i], "structure_confidence"] = _calculate_confidence(i)
                last_trend = "bearish"

        elif last_trend == "bullish":
            if last_low and close < last_low:
                df.at[df.index[i], "market_structure"] = "CHoCH"
                df.at[df.index[i], "structure_confidence"] = _calculate_confidence(i)
                last_trend = "bearish"
            elif last_high and close > last_high:
                df.at[df.index[i], "market_structure"] = "BOS"
                df.at[df.index[i], "structure_confidence"] = _calculate_confidence(i)

        elif last_trend == "bearish":
            if last_high and close > last_high:
                df.at[df.index[i], "market_structure"] = "CHoCH"
                df.at[df.index[i], "structure_confidence"] = _calculate_confidence(i)
                last_trend = "bullish"
            elif last_low and close < last_low:
                df.at[df.index[i], "market_structure"] = "BOS"
                df.at[df.index[i], "structure_confidence"] = _calculate_confidence(i)

    return df

def classify_trend_bias(df, fast=50, slow=200):
    """
    Add trend bias ('bullish' or 'bearish') based on EMA crossover.

    Parameters:
        df (pd.DataFrame): Must contain a 'Close' column.
        fast (int): Fast EMA span.
        slow (int): Slow EMA span.

    Returns:
        pd.DataFrame: With added columns ['ema_fast', 'ema_slow', 'trend_bias']
    """
    df = df.copy()
    df["ema_fast"] = df["Close"].ewm(span=fast).mean()
    df["ema_slow"] = df["Close"].ewm(span=slow).mean()
    df["trend_bias"] = None

    df.loc[df["ema_fast"] > df["ema_slow"], "trend_bias"] = "bullish"
    df.loc[df["ema_fast"] < df["ema_slow"], "trend_bias"] = "bearish"

    return df

def detect_order_blocks(df, direction="bullish", window=5):
    """
    Detect valid order blocks near BOS in HTF based on trend direction.

    Parameters:
        df (pd.DataFrame): Must include ['Open', 'Close', 'High', 'Low', 'market_structure', 'trend_bias'].
        direction (str): 'bullish' or 'bearish'
        window (int): Number of candles to look back for valid OB

    Returns:
        pd.DataFrame: With added columns ['ob_high', 'ob_low', 'ob_time', 'ob_direction']
    """
    df = df.copy()
    df["ob_high"] = None
    df["ob_low"] = None
    df["ob_time"] = None
    df["ob_direction"] = None

    for i in range(window, len(df)):
        row = df.iloc[i]

        if row["market_structure"] != "BOS":
            continue
        if direction == "bullish" and row.get("trend_bias") != "bullish":
            continue
        if direction == "bearish" and row.get("trend_bias") != "bearish":
            continue

        window_df = df.iloc[i - window:i]

        if direction == "bullish":
            candidates = window_df[window_df["Close"] < window_df["Open"]]  # bearish candles
        else:
            candidates = window_df[window_df["Close"] > window_df["Open"]]  # bullish candles

        if candidates.empty:
            continue

        last_ob = candidates.iloc[-1]
        df.at[df.index[i], "ob_high"] = last_ob["High"]
        df.at[df.index[i], "ob_low"] = last_ob["Low"]
        df.at[df.index[i], "ob_time"] = last_ob.name
        df.at[df.index[i], "ob_direction"] = direction

    return df


# Combine detect_order_blocks and calculate_ote_from_bos into a unified pipeline that retains OB/OTE zone boundaries.

def detect_order_blocks(df, direction="bullish", window=5):
    """
    Detect valid order blocks near BOS in HTF based on trend direction.
    Adds columns: ['ob_high', 'ob_low', 'ob_time', 'ob_direction']
    """
    df = df.copy()
    df["ob_high"] = None
    df["ob_low"] = None
    df["ob_time"] = None
    df["ob_direction"] = None

    for i in range(window, len(df)):
        row = df.iloc[i]

        if row["market_structure"] != "BOS":
            continue
        if direction == "bullish" and row.get("trend_bias") != "bullish":
            continue
        if direction == "bearish" and row.get("trend_bias") != "bearish":
            continue

        window_df = df.iloc[i - window:i]

        if direction == "bullish":
            candidates = window_df[window_df["Close"] < window_df["Open"]]  # bearish candles
        else:
            candidates = window_df[window_df["Close"] > window_df["Open"]]  # bullish candles

        if candidates.empty:
            continue

        last_ob = candidates.iloc[-1]
        df.at[df.index[i], "ob_high"] = last_ob["High"]
        df.at[df.index[i], "ob_low"] = last_ob["Low"]
        df.at[df.index[i], "ob_time"] = last_ob.name
        df.at[df.index[i], "ob_direction"] = direction

    return df

def calculate_ote_from_bos(df, lookback=10):
    """
    Calculate OTE (Optimal Trade Entry) zone from BOS using recent swing range.
    Adds columns: ['ote_start', 'ote_best', 'ote_end', 'ote_dir']
    """
    df = df.copy()
    df["ote_start"] = None
    df["ote_best"] = None
    df["ote_end"] = None
    df["ote_dir"] = None
    df["ote_expire_time"] = None

    for i in range(lookback, len(df)):
        row = df.iloc[i]
        if row.get("market_structure") != "BOS":
            continue

        bias = row.get("trend_bias")
        if pd.isna(bias) or bias not in ["bullish", "bearish"]:
            continue

        bos_time = df.index[i]
        swing_window = df.iloc[i - lookback:i]

        if bias == "bullish":
            swing_low = swing_window["Low"].min()
            range_size = row["High"] - swing_low
            ote_start = row["High"] - 0.62 * range_size
            ote_best = row["High"] - 0.705 * range_size
            ote_end = row["High"] - 0.79 * range_size

        elif bias == "bearish":
            swing_high = swing_window["High"].max()
            range_size = swing_high - row["Low"]
            ote_start = row["Low"] + 0.62 * range_size
            ote_best = row["Low"] + 0.705 * range_size
            ote_end = row["Low"] + 0.79 * range_size

        else:
            continue

        df.at[bos_time, "ote_start"] = round(ote_start, 2)
        df.at[bos_time, "ote_best"] = round(ote_best, 2)
        df.at[bos_time, "ote_end"] = round(ote_end, 2)
        df.at[bos_time, "ote_dir"] = bias

    return df


def detect_ltf_structure(df, left=2, right=2):
    """
    Detect microstructure shifts on LTF using swing highs/lows.

    Parameters:
        df (pd.DataFrame): Price data with 'High', 'Low', 'Close'.
        left/right (int): Bars to check for swing pivots.

    Returns:
        pd.DataFrame: With columns ['lt_swing_high', 'lt_swing_low', 'market_structure']
    """
    df = df.copy()

    # Identify swing highs/lows
    df["lt_swing_high"] = df["High"][
        (df["High"].shift(left) < df["High"]) & 
        (df["High"].shift(-right) < df["High"])
    ]
    df["lt_swing_low"] = df["Low"][
        (df["Low"].shift(left) > df["Low"]) & 
        (df["Low"].shift(-right) > df["Low"])
    ]

    df["market_structure"] = None
    last_trend, last_high, last_low = None, None, None

    for i in range(len(df)):
        row = df.iloc[i]

        # Track most recent swings
        if not pd.isna(row["lt_swing_high"]):
            last_high = row["lt_swing_high"]
        if not pd.isna(row["lt_swing_low"]):
            last_low = row["lt_swing_low"]

        # Skip early rows
        if i < max(left, right):
            continue

        close = row["Close"]

        # Structure detection logic
        if last_trend is None:
            if last_high and close > last_high:
                df.at[df.index[i], "market_structure"] = "BOS"
                last_trend = "bullish"
            elif last_low and close < last_low:
                df.at[df.index[i], "market_structure"] = "BOS"
                last_trend = "bearish"
        elif last_trend == "bullish":
            if last_low and close < last_low:
                df.at[df.index[i], "market_structure"] = "CHoCH"
                last_trend = "bearish"
            elif last_high and close > last_high:
                df.at[df.index[i], "market_structure"] = "BOS"
        elif last_trend == "bearish":
            if last_high and close > last_high:
                df.at[df.index[i], "market_structure"] = "CHoCH"
                last_trend = "bullish"
            elif last_low and close < last_low:
                df.at[df.index[i], "market_structure"] = "BOS"

    return df



def mark_ltf_confirmation(df, max_lag=10):
    """
    Identify LTF confirmation structure: CHoCH followed by BOS within lag window.

    Parameters:
        df (pd.DataFrame): DataFrame with 'lt_structure' (structure label on LTF).
        max_lag (int): Number of bars to look back for CHoCH -> BOS sequence.

    Returns:
        pd.DataFrame: With confirmation flags ['choch_before', 'bos_after_choch']
    """
    df = df.copy()
    df["choch_before"] = False
    df["bos_after_choch"] = False

    for i in range(max_lag, len(df)):
        recent = df.iloc[i - max_lag:i]
        choch_idx = recent[recent["lt_structure"] == "CHoCH"].index
        bos_idx = recent[recent["lt_structure"] == "BOS"].index

        if not choch_idx.empty and not bos_idx.empty:
            if choch_idx[-1] < bos_idx[-1]:
                df.at[df.index[i], "choch_before"] = True
                df.at[df.index[i], "bos_after_choch"] = True

    return df


def detect_liquidity_grab(df, lookback=5):
    """
    Detect wick-only liquidity grabs that fail to close beyond prior swing zones.

    Parameters:
        df (pd.DataFrame): Must include 'High', 'Low', and 'Close'.
        lookback (int): Lookback window for prior highs/lows.

    Returns:
        pd.DataFrame: With 'liquidity_grab' flag (bool)
    """
    df = df.copy()
    df["liquidity_grab"] = False

    for i in range(lookback, len(df)):
        row = df.iloc[i]
        prev_highs = df.iloc[i - lookback:i]["High"]
        prev_lows = df.iloc[i - lookback:i]["Low"]

        # Wick above highs but failed to close above
        if row["High"] > prev_highs.max() and row["Close"] < prev_highs.max():
            df.at[df.index[i], "liquidity_grab"] = True
        # Wick below lows but failed to close below
        elif row["Low"] < prev_lows.min() and row["Close"] > prev_lows.min():
            df.at[df.index[i], "liquidity_grab"] = True

    return df


def detect_bullish_engulfing(df):
    """
    Detect classic bullish engulfing patterns.

    Conditions:
        - Previous candle is bearish (Close < Open)
        - Current candle is bullish (Close > Open)
        - Current body fully engulfs previous body

    Returns:
        pd.DataFrame: With boolean column 'bullish_engulfing'
    """
    df = df.copy()
    df["bullish_engulfing"] = False

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        if prev["Close"] < prev["Open"] and curr["Close"] > curr["Open"]:
            if curr["Close"] > prev["Open"] and curr["Open"] < prev["Close"]:
                df.at[df.index[i], "bullish_engulfing"] = True

    return df


def mark_inside_zones(df, price_col="entry_price"):
    df = df.copy()
    df["inside_ote_zone"] = True
    df["inside_ob_zone"] = True

    for i in range(len(df)):
        row = df.iloc[i]
        price = row.get(price_col, row["entry_price"])

        if pd.notna(row.get("ote_start")) and pd.notna(row.get("ote_end")):
            if row["ote_start"] <= price <= row["ote_end"]:
                df.at[df.index[i], "inside_ote_zone"] = 1

        if pd.notna(row.get("ob_low")) and pd.notna(row.get("ob_high")):
            if row["ob_low"] <= price <= row["ob_high"]:
                df.at[df.index[i], "inside_ob_zone"] = 1

    return df



def filter_valid_entries_2(df, confidence_threshold=0.6):
    df = df.copy()
    df["is_valid_entry"] = False

    for i in range(len(df)):
        row = df.iloc[i]

        if not row.get("inside_ote_zone") and not row.get("inside_ob_zone"):
            continue
        if not row.get("choch_before"):
            continue
        if not row.get("bos_after_choch"):
            continue
        if not row.get("liquidity_grab"):
            continue
        if not row.get("structure_confidence") or row["structure_confidence"] < confidence_threshold:
            continue

        # Optional: bullish engulfing confirmation
        if not row.get("bullish_engulfing"):
            continue

        df.at[df.index[i], "is_valid_entry"] = True

    return df


def set_entry_sl_tp(df, risk_multiple_1=1.5, risk_multiple_2=2.5, atr_buffer=0.0):
    df = df.copy()
    for col in ["entry_price", "stop_loss", "take_profit_1", "take_profit_2", "rr_1", "rr_2"]:
        df[col] = None

    for i, row in df[df["is_valid_entry"] == True].iterrows():
        direction = row.get("ote_dir") if pd.notna(row.get("ote_dir")) else row.get("ob_direction")
        if direction not in ["bullish", "bearish"]:
            print(f"⚠️ Index {i}: Invalid direction → {direction}")
            continue

        ob_low, ob_high = row.get("ob_low"), row.get("ob_high")
        ote_start, ote_end = row.get("ote_start"), row.get("ote_end")

        if pd.notna(ote_start) and pd.notna(ote_end) and ote_start == ote_end:
            print(f"⚠️ Index {i}: Flat OTE zone → start == end == {ote_start}")
            continue
        if pd.notna(ob_low) and pd.notna(ob_high) and ob_low == ob_high:
            print(f"⚠️ Index {i}: Flat OB zone → low == high == {ob_low}")
            continue

        entry, sl = None, None
        if direction == "bullish":
            if pd.notna(ob_low) and not np.isclose(ob_low, ob_low - atr_buffer):
                entry = ob_low
                sl = ob_low - atr_buffer
            elif pd.notna(ote_start) and pd.notna(ote_end) and not np.isclose(ote_start, ote_end - atr_buffer):
                entry = ote_start
                sl = ote_end - atr_buffer
        elif direction == "bearish":
            if pd.notna(ob_high) and not np.isclose(ob_high, ob_high + atr_buffer):
                entry = ob_high
                sl = ob_high + atr_buffer
            elif pd.notna(ote_start) and pd.notna(ote_end) and not np.isclose(ote_start, ote_end + atr_buffer):
                entry = ote_start
                sl = ote_end + atr_buffer

        if entry is None or sl is None or np.isclose(entry, sl):
            print(f"⚠️ Index {i}: Invalid risk → entry = {entry}, SL = {sl}")
            print(f"  ob_low: {ob_low}, ob_high: {ob_high}, ote_start: {ote_start}, ote_end: {ote_end}")
            continue

        risk = abs(entry - sl)
        tp1 = entry + risk_multiple_1 * risk if direction == "bullish" else entry - risk_multiple_1 * risk
        tp2 = entry + risk_multiple_2 * risk if direction == "bullish" else entry - risk_multiple_2 * risk

        df.at[i, "entry_price"] = round(entry, 2)
        df.at[i, "stop_loss"] = round(sl, 2)
        df.at[i, "take_profit_1"] = round(tp1, 2)
        df.at[i, "take_profit_2"] = round(tp2, 2)
        df.at[i, "rr_1"] = round(risk_multiple_1, 2)
        df.at[i, "rr_2"] = round(risk_multiple_2, 2)

    return df

def simulate_trade_outcomes(df, future_window=20):
    df = df.copy()
    df["exit_price"] = None
    df["exit_reason"] = None
    df["r_multiple"] = None
    df["equity_curve"] = None

    equity = 0
    base_risk = 100

    for i in range(len(df)):
        row = df.iloc[i]
        if not row.get("is_valid_entry"):
            continue

        entry_price = row.get("entry_price")
        sl = row.get("stop_loss")
        tp1 = row.get("take_profit_1")
        tp2 = row.get("take_profit_2")
        direction = row.get("ote_dir") or row.get("ob_direction")

        if None in [entry_price, sl, tp1, tp2, direction]:
            print(f"\n❌ Skipping index {i} due to:")
            print(f"  entry_price: {entry_price}")
            print(f"  stop_loss: {sl}")
            print(f"  tp1: {tp1}")
            print(f"  tp2: {tp2}")
            print(f"  direction: {direction}")
            continue

        if direction == "bullish":
            future_prices = df["High"].iloc[i:i+future_window]
            stop_prices = df["Low"].iloc[i:i+future_window]
        else:
            future_prices = df["Low"].iloc[i:i+future_window]
            stop_prices = df["High"].iloc[i:i+future_window]

        hit_tp1 = any(p >= tp1 if direction == "bullish" else p <= tp1 for p in future_prices if pd.notna(p))
        hit_tp2 = any(p >= tp2 if direction == "bullish" else p <= tp2 for p in future_prices if pd.notna(p))
        hit_sl = any(p <= sl if direction == "bullish" else p >= sl for p in stop_prices if pd.notna(p))

        if hit_sl and not hit_tp1:
            r_result = -1
            exit_reason = "SL"
            exit_price = sl
        elif hit_tp1 and not hit_tp2:
            r_result = 0.75
            exit_reason = "TP1"
            exit_price = tp1
        elif hit_tp2:
            r_result = 2.0
            exit_reason = "TP2"
            exit_price = tp2
        else:
            print(f"❌ Entry at index {i} is valid but no TP/SL hit in window.")
            continue

        df.at[df.index[i], "exit_price"] = round(exit_price, 2)
        df.at[df.index[i], "exit_reason"] = exit_reason
        df.at[df.index[i], "r_multiple"] = r_result
        equity += base_risk * r_result
        df.at[df.index[i], "equity_curve"] = equity

    return df


def filter_valid_entries(df, confidence_threshold=0.6):
    df = df.copy()
    df["is_valid_entry"] = False
    df["confirmation_count"] = 0
    df["entry_zone_type"] = None  # Track whether OB or OTE was used
    df["setup_type"] = None      # Track setup type (OB or OTE)

    for i in range(len(df)):
        row = df.iloc[i]
        confirmations = 0

        def check_conditions():
            count = 0
            if row.get("choch_before"):
                count += 1
            if row.get("bos_after_choch"):
                count += 1
            if row.get("liquidity_grab"):
                count += 1
            if row.get("bullish_engulfing"):
                count += 1
            return count

        # First try OB zone
        if row.get("inside_ob_zone"):
            confirmations = check_conditions()
            df.at[df.index[i], "confirmation_count"] = confirmations
            if confirmations >= 3 and row.get("structure_confidence", 0) >= confidence_threshold:
                df.at[df.index[i], "is_valid_entry"] = True
                df.at[df.index[i], "entry_zone_type"] = "OB"
                df.at[df.index[i], "setup_type"] = "OB"
                continue  # OB is prioritized, so skip checking OTE

        # Then try OTE zone
        if row.get("inside_ote_zone"):
            confirmations = check_conditions()
            df.at[df.index[i], "confirmation_count"] = confirmations
            if confirmations >= 3 and row.get("structure_confidence", 0) >= confidence_threshold:
                df.at[df.index[i], "is_valid_entry"] = True
                df.at[df.index[i], "entry_zone_type"] = "OTE"
                df.at[df.index[i], "setup_type"] = "OTE"

    return df


def assign_entry_price(df, method="Low"):
    """
    Assign the entry price used for zone tagging and backtesting.

    Parameters:
        method (str): Which price to use — "Open", "Close", "Low", or custom logic.

    Returns:
        pd.DataFrame with 'entry_price' column
    """
    df = df.copy()

    if method == "Open":
        df["entry_price"] = df["Open"]
    elif method == "Close":
        df["entry_price"] = df["Close"]
    elif method == "Low":
        df["entry_price"] = df["Low"]
    elif method == "High":
        df["entry_price"] = df["High"]
    else:
        # fallback logic or custom rule
        df["entry_price"] = df["Close"]

    return df
