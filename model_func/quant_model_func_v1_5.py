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
    """Enhanced structure confidence score calculation at index i."""
    row = df.iloc[i]
    prev_row = df.iloc[i - 1]
    close = row["Close"]
    volume = row.get("Volume", 1)

    # === ATR Component ===
    df["H-L"] = df["High"] - df["Low"]
    atr = df["H-L"].rolling(atr_window).mean().iloc[i]
    range_broken = abs(close - prev_row["Close"])
    break_score = min(range_broken / atr, 2) if atr else 0  # scale up to 2

    # === Candle Body Component ===
    body = abs(row["Close"] - row["Open"])
    candle_range = row["High"] - row["Low"]
    body_score = body / candle_range if candle_range else 0  # full-bodied = 1

    # === Volume Component (continuous) ===
    avg_vol = df["Volume"].rolling(vol_window).mean().iloc[i]
    vol_score = min(volume / avg_vol, 2.0) if avg_vol else 0  # cap at 2.0, normalize

    # === Structure Type Boost ===
    structure_boost = 0.0
    if "structure" in row:
        if row["structure"] == "BOS":
            structure_boost = 0.2
        elif row["structure"] == "CHoCH":
            structure_boost = 0.1

    # === Final Score ===
    confidence = (
        0.4 * break_score +
        0.2 * body_score +
        0.2 * vol_score +
        structure_boost
    )
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
    Detect LTF microstructure shifts (BOS/CHoCH) using swing highs and lows.

    Parameters:
        df (pd.DataFrame): Must contain 'High', 'Low', 'Close' columns.
        left/right (int): Number of bars to the left/right to consider a swing.

    Returns:
        pd.DataFrame: Adds ['lt_swing_high', 'lt_swing_low', 'lt_structure']
    """
    df = df.copy()
    df["lt_swing_high"] = df["High"][
        (df["High"].shift(left) > df["High"]) & 
        (df["High"].shift(-right) > df["High"])
    ]
    df["lt_swing_low"] = df["Low"][
        (df["Low"].shift(left) < df["Low"]) & 
        (df["Low"].shift(-right) < df["Low"])
    ]
    df["lt_structure"] = None

    last_trend = None
    last_high = None
    last_low = None

    for i in range(max(left, right), len(df)):
        row = df.iloc[i]
        close = row["Close"]

        # Update recent swings
        if not pd.isna(row["lt_swing_high"]):
            last_high = row["lt_swing_high"]
        if not pd.isna(row["lt_swing_low"]):
            last_low = row["lt_swing_low"]

        if last_trend is None:
            if last_high is not None and close > last_high:
                df.at[df.index[i], "lt_structure"] = "BOS"
                last_trend = "bullish"
            elif last_low is not None and close < last_low:
                df.at[df.index[i], "lt_structure"] = "BOS"
                last_trend = "bearish"
        elif last_trend == "bullish":
            if last_low is not None and close < last_low:
                df.at[df.index[i], "lt_structure"] = "CHoCH"
                last_trend = "bearish"
            elif last_high is not None and close > last_high:
                df.at[df.index[i], "lt_structure"] = "BOS"
        elif last_trend == "bearish":
            if last_high is not None and close > last_high:
                df.at[df.index[i], "lt_structure"] = "CHoCH"
                last_trend = "bullish"
            elif last_low is not None and close < last_low:
                df.at[df.index[i], "lt_structure"] = "BOS"

    return df


def mark_ltf_confirmation(df, max_lag=10):
    """
    Mark LTF confirmation (CHoCH → BOS) only when price is inside OB or OTE zone.

    Parameters:
        df (pd.DataFrame): DataFrame with 'lt_structure', 'inside_ob_zone', 'inside_ote_zone'
        max_lag (int): How many previous bars to look for CHoCH → BOS pattern

    Returns:
        pd.DataFrame: Adds 'choch_before' and 'bos_after_choch' flags
    """
    df = df.copy()
    df["choch_before"] = False
    df["bos_after_choch"] = False

    for i in range(max_lag, len(df)):
        row = df.iloc[i]

        # ✅ Only look for confirmation if price is inside OB or OTE zone
        if not (row.get("inside_ob_zone", False) or row.get("inside_ote_zone", False)):
            continue

        # Lookback window for CHoCH → BOS
        recent = df.iloc[i - max_lag:i]
        choch_positions = recent[recent["lt_structure"] == "CHoCH"].index
        bos_positions = recent[recent["lt_structure"] == "BOS"].index

        # Confirmation: CHoCH appears before BOS
        if any(c < b for c in choch_positions for b in bos_positions):
            df.at[df.index[i], "choch_before"] = True
            df.at[df.index[i], "bos_after_choch"] = True

    return df


def detect_liquidity_grab(df, lookback=5):
    """
    Detect directional wick-based liquidity grabs with confirmation filters.

    Parameters:
        df (pd.DataFrame): Must include ['High', 'Low', 'Open', 'Close', 
                                         'inside_ob_zone', 'inside_ote_zone'].
        lookback (int): Number of prior candles to compare for swing highs/lows.

    Returns:
        pd.DataFrame: Adds ['liquidity_grab'] (bool) and ['grab_direction'] ('up', 'down', or None).
    """
    df = df.copy()
    df["liquidity_grab"] = False
    df["grab_direction"] = None

    for i in range(lookback, len(df)):
        row = df.iloc[i]
        high_slice = df.iloc[i - lookback:i]["High"]
        low_slice = df.iloc[i - lookback:i]["Low"]
        prev_high = high_slice.max()
        prev_low = low_slice.min()

        open_price = row["Open"]
        close_price = row["Close"]
        high = row["High"]
        low = row["Low"]

        body = abs(close_price - open_price)
        upper_wick = high - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low

        # Only check for grabs if inside OB/OTE zone
        if not row.get("inside_ob_zone", False) and not row.get("inside_ote_zone", False):
            continue

        # Grab above previous highs (fake breakout up)
        if high > prev_high and close_price < prev_high and upper_wick > body:
            df.at[df.index[i], "liquidity_grab"] = True
            df.at[df.index[i], "grab_direction"] = "up"

        # Grab below previous lows (fake breakdown down)
        elif low < prev_low and close_price > prev_low and lower_wick > body:
            df.at[df.index[i], "liquidity_grab"] = True
            df.at[df.index[i], "grab_direction"] = "down"

    return df


def mark_zone_session(df):
    """
    Marks persistent session where price is inside OB or OTE zone.
    Adds 'zone_session_id' column to track sessions.
    """
    df = df.copy()
    in_session = False
    session_ids = []
    session_id = 0

    for i, row in df.iterrows():
        inside = row.get("inside_ob_zone", False) or row.get("inside_ote_zone", False)
        if inside:
            if not in_session:
                session_id += 1
                in_session = True
            session_ids.append(session_id)
        else:
            in_session = False
            session_ids.append(None)

    df["zone_session_id"] = session_ids
    return df

def accumulate_zone_confirmations(df):
    """
    Accumulates confirmations during each OB/OTE zone session.
    Returns updated DataFrame with persistent confirmation flags.
    """
    df = mark_zone_session(df)
    df["sticky_choch"] = False
    df["sticky_bos_after_choch"] = False
    df["sticky_engulfing"] = False
    df["sticky_liquidity_grab"] = False

    for session_id in df["zone_session_id"].dropna().unique():
        session_mask = df["zone_session_id"] == session_id
        session_df = df[session_mask]

        choch_seen = False
        bos_seen = False
        engulf_seen = False
        grab_seen = False

        for idx in session_df.index:
            row = df.loc[idx]

            if row.get("lt_structure") == "CHoCH":
                choch_seen = True
            if row.get("lt_structure") == "BOS" and choch_seen:
                bos_seen = True
            if row.get("bullish_engulfing"):
                engulf_seen = True
            if row.get("liquidity_grab"):
                grab_seen = True

            df.at[idx, "sticky_choch"] = choch_seen
            df.at[idx, "sticky_bos_after_choch"] = bos_seen
            df.at[idx, "sticky_engulfing"] = engulf_seen
            df.at[idx, "sticky_liquidity_grab"] = grab_seen

    return df

def mark_final_entry(df, confidence_threshold=0.6, require_liquidity=False, require_bos=False):
    """
    Final entry logic using accumulated confirmations.
    Returns df with 'final_entry' boolean column.
    """
    df = accumulate_zone_confirmations(df)
    df["final_entry"] = (
        (df["structure_confidence"] >= confidence_threshold)
        & df["sticky_choch"]
        & (df["sticky_bos_after_choch"] if require_bos else True)
        & (df["sticky_liquidity_grab"] if require_liquidity else True)
        & df["sticky_engulfing"]
    )
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
                df.at[df.index[i], "inside_ote_zone"] = True

        if pd.notna(row.get("ob_low")) and pd.notna(row.get("ob_high")):
            if row["ob_low"] <= price <= row["ob_high"]:
                df.at[df.index[i], "inside_ob_zone"] = True

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


def filter_valid_entries_1(df, confidence_threshold=0.55):
    df = df.copy()
    df["is_valid_entry"] = False
    df["confirmation_count"] = 0
    df["entry_zone_type"] = None
    df["setup_type"] = None

    for i in range(len(df)):
        row = df.iloc[i]

        def check_conditions():
            count = 0
            if row.get("choch_before"):
                count += 1
            if row.get("bullish_engulfing"):
                count += 1
            if row.get("bos_after_choch"):   # optional bonus
                count += 1
            if row.get("liquidity_grab"):    # optional bonus
                count += 1
            return count

        confirmations = check_conditions()
        df.at[df.index[i], "confirmation_count"] = confirmations

        # OB prioritized
        if row.get("inside_ob_zone") and confirmations >= 2 and row.get("structure_confidence", 0) >= confidence_threshold:
            df.at[df.index[i], "is_valid_entry"] = True
            df.at[df.index[i], "entry_zone_type"] = "OB"
            df.at[df.index[i], "setup_type"] = "OB"
            continue

        # OTE fallback
        if row.get("inside_ote_zone") and confirmations >= 2 and row.get("structure_confidence", 0) >= confidence_threshold:
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
