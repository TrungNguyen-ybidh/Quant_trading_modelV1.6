import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import numpy as np
from joblib import load
from ml_utils_combined import apply_ml_models
import sys
import alpaca_trade_api as tradeapi
import pytz

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
et_tz = pytz.timezone("America/New_York")

from model_func.quant_model_func_v1_5 import (
    detect_htf_structure, calculate_structure_confidence, detect_order_blocks,
    calculate_ote_from_bos, assign_entry_price, mark_inside_zones,
    detect_bullish_engulfing, detect_ltf_structure, mark_zone_session,
    accumulate_zone_confirmations, mark_final_entry, set_entry_sl_tp,
    classify_trend_bias, mark_ltf_confirmation, detect_liquidity_grab
)

def print_trade_summary(symbol, row, qty):
    print("\n\U0001F4CA Trade Summary:")
    print(f"\u2022 Symbol: {symbol}")
    print(f"\u2022 Time: {row.name.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\u2022 Entry Price: {row['entry_price']:.2f}")
    print(f"\u2022 Stop Loss: {row['stop_loss']:.2f}")
    print(f"\u2022 Position Size: {qty}")
    print(f"\u2022 ML Confidence: {row['ml_prob']:.2f}")
    print(f"\u2022 ML R Prediction: {row['ml_r_pred']:.2f}")
    print(f"\u2022 Structure Confidence: {row['structure_confidence']:.2f}")


def run_live_trading_model(
    alpaca_client,
    symbol,
    ltf_interval="15Min",
    lookback_minutes=60,
    confidence_threshold=0.6,
    ml_prob_threshold=0.5,
    ml_r_threshold=0.8,
    atr_buffer=0.0,
    use_ml=True,
    verbose=True,
    execute_trade=False,
    trade_qty=1,
    account_balance=10000,
    max_spread_pct=0.002,
    min_volume=10
):
    def to_rfc3339(dt):
        if dt.tzinfo is None:
            # If naive, assume UTC
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def is_crypto(symbol): return "-USD" in symbol
    def convert_symbol(symbol):
        if is_crypto(symbol): return symbol.replace("-USD", "/USD")
        if symbol.endswith("=X"): return symbol.replace("=X", "") + "/USD"
        return symbol

    def calculate_position_size(entry_price, stop_loss, account_balance, max_risk_pct, prob, r_pred):
        risk_perc = max_risk_pct * min(1.0, (prob * r_pred))
        dollar_risk = account_balance * risk_perc
        stop_size = abs(entry_price - stop_loss)
        if stop_size == 0:
            return 0
        qty = dollar_risk / stop_size
        return max(1, int(qty))

    def is_spread_acceptable(df):
        if "AskPrice" in df.columns and "BidPrice" in df.columns:
            df["Spread"] = df["AskPrice"] - df["BidPrice"]
            df["MidPrice"] = (df["AskPrice"] + df["BidPrice"]) / 2
            df["SpreadPct"] = df["Spread"] / df["MidPrice"]
            return df["SpreadPct"].iloc[-1] <= max_spread_pct
        return True

    now = datetime.now(et_tz)
    start_time = now - timedelta(minutes=lookback_minutes)
    start_str = to_rfc3339(start_time)
    conv_symbol = convert_symbol(symbol)

    if is_crypto(symbol):
        min_volume = 10
    else:
        min_volume = 100

    try:
        is_c = is_crypto(symbol)
        if is_c:
            df = alpaca_client.get_crypto_bars(conv_symbol, ltf_interval, start=start_str).df
            #df = df[df["exchange"] == "CBSE"] if "exchange" in df.columns else df
        else:
            df = alpaca_client.get_bars(conv_symbol, ltf_interval, start=start_str).df
        df = df.rename(columns=str.title)
        df.index = df.index.tz_convert(et_tz)
    except Exception as e:
        print(f"\n❌ Error fetching live data: {e}")
        return None

    if df.empty:
        print("\n⚠️ No data received.")
        return None

    if not is_crypto(symbol) and df["Volume"].tail(3).mean() < min_volume:
        print(f"\n⚠️ Avg volume too low: {df['Volume'].tail(3).mean():.2f}")
        return None

    if not is_spread_acceptable(df):
        print("\n⚠️ Spread too wide, skipping trade.")
        return None

    ltf_df = df.copy()

    # === Get HTF (1H) data separately ===
    htf_lookback_minutes = 7 * 24 * 60  
    htf_start_time = now - timedelta(minutes=htf_lookback_minutes)
    htf_start_str = to_rfc3339(htf_start_time)

    try:
        if is_c:
            htf_df = alpaca_client.get_crypto_bars(conv_symbol, "1H", start=htf_start_str).df
        else:
            htf_df = alpaca_client.get_bars(conv_symbol, "1H", start=htf_start_str).df
        htf_df = htf_df.rename(columns=str.title)
        htf_df.index = htf_df.index.tz_convert(et_tz)
    except Exception as e:
        print(f"\n❌ Error fetching HTF data: {e}")
        return None

    if htf_df.empty:
        print("\n⚠️ No HTF data received.")
        return None

    htf_df = detect_htf_structure(htf_df)
    htf_df = classify_trend_bias(htf_df)
    htf_df = calculate_ote_from_bos(htf_df)
    htf_df = detect_order_blocks(htf_df)
    htf_df = calculate_structure_confidence(htf_df)
    htf_df = mark_inside_zones(htf_df)

    # === HTF Summary Counts ===
    htf_bos_count = (htf_df["market_structure"] == "BOS").sum()
    htf_choch_count = (htf_df["market_structure"] == "CHoCH").sum()

    latest_ote_zone = htf_df[["ote_start", "ote_end", "ote_dir"]].dropna().iloc[-1:]
    latest_ob_zone = htf_df[["ob_high", "ob_low", "ob_direction"]].dropna().iloc[-1:]
    
    entered_ote_zone = False
    entered_ob_zone = False
    latest_close = df["Close"].iloc[-1]

    if not latest_ote_zone.empty:
        if latest_ote_zone["ote_dir"].values[0] == "bullish" and latest_ote_zone["ote_start"].values[0] <= latest_close <= latest_ote_zone["ote_end"].values[0]:
            entered_ote_zone = True
        elif latest_ote_zone["ote_dir"].values[0] == "bearish" and latest_ote_zone["ote_end"].values[0] <= latest_close <= latest_ote_zone["ote_start"].values[0]:
            entered_ote_zone = True

    if not latest_ob_zone.empty:
        if latest_ob_zone["ob_direction"].values[0] == "bullish" and latest_ob_zone["ob_low"].values[0] <= latest_close <= latest_ob_zone["ob_high"].values[0]:
            entered_ob_zone = True
        elif latest_ob_zone["ob_direction"].values[0] == "bearish" and latest_ob_zone["ob_high"].values[0] >= latest_close >= latest_ob_zone["ob_low"].values[0]:
            entered_ob_zone = True

    print(f"\n📈 HTF Summary for {symbol}:")
    print(f"  • BOS count: {htf_bos_count}")
    print(f"  • CHoCH count: {htf_choch_count}")
    print(f"  • Price inside OTE zone: {'✅' if entered_ote_zone else '❌'}")
    print(f"  • Price inside OB zone: {'✅' if entered_ob_zone else '❌'}")

    latest = htf_df[[
        "ote_start", "ote_end", "ote_dir",
        "ob_high", "ob_low", "ob_direction",
        "structure_confidence", "trend_bias"
    ]].ffill()

    latest = latest.infer_objects(copy=False).iloc[-1:]

    latest = latest.infer_objects(copy=False).iloc[-1:]
    for col in latest.columns:
        ltf_df[col] = latest[col].values[0]

        # === Invalidate OB/OTE zones if already broken ===
        latest_close = ltf_df["Close"].iloc[-1]

        # Invalidate OB
        if (
            ltf_df["ob_direction"].iloc[-1] == "bearish"
            and latest_close > ltf_df["ob_high"].iloc[-1]
        ):
            ltf_df["ob_direction"] = "invalid"

        if (
            ltf_df["ob_direction"].iloc[-1] == "bullish"
            and latest_close < ltf_df["ob_low"].iloc[-1]
        ):
            ltf_df["ob_direction"] = "invalid"

        # Invalidate OTE
        if (
            ltf_df["ote_dir"].iloc[-1] == "bearish"
            and latest_close > ltf_df["ote_end"].iloc[-1]
        ):
            ltf_df["ote_dir"] = "invalid"

        if (
            ltf_df["ote_dir"].iloc[-1] == "bullish"
            and latest_close < ltf_df["ote_start"].iloc[-1]
        ):
            ltf_df["ote_dir"] = "invalid"

        ltf_df = detect_ltf_structure(ltf_df)
        ltf_df = mark_ltf_confirmation(ltf_df, max_lag=30)
        ltf_df = detect_liquidity_grab(ltf_df)
        ltf_df = detect_bullish_engulfing(ltf_df)
        ltf_df = assign_entry_price(ltf_df, method="Low")
        ltf_df = mark_zone_session(ltf_df)
        ltf_df = calculate_structure_confidence(ltf_df)
        ltf_df = accumulate_zone_confirmations(ltf_df)
        ltf_df = mark_final_entry(ltf_df)
        ltf_df = set_entry_sl_tp(ltf_df, atr_buffer=atr_buffer)

        # Optional: print summary of final signals
        print(f"\n📊 Post structure detection for {symbol} (last few rows):")
        print(ltf_df[["final_entry", "structure_confidence", "inside_ob_zone", "bullish_engulfing", "liquidity_grab", "entry_price"]].tail(5))

        print(f"\n🔍 Entry Confirmation Breakdown for {symbol}:")
        recent = ltf_df.tail(5)  
        for idx, row in recent.iterrows():  
            print(f"🕓 {idx}")
            print(f"  • structure_confidence: {row.get('structure_confidence')}")
            print(f"  • inside_ob_zone: {row.get('inside_ob_zone')}")
            print(f"  • inside_ote_zone: {row.get('inside_ote_zone')}")
            print(f"  • choch_before: {row.get('choch_before')}")
            print(f"  • bos_after_choch: {row.get('bos_after_choch')}")
            print(f"  • liquidity_grab: {row.get('liquidity_grab')}")
            print(f"  • bullish_engulfing: {row.get('bullish_engulfing')}")
            print(f"  • confirmation_count: {row.get('confirmation_count')}")
            print(f"  • final_entry: {row.get('final_entry')}")
            print(f"  • entry_zone_type: {row.get('entry_zone_type')}")
            print(f"  • setup_type: {row.get('setup_type')}")
            print("-" * 40)
        ltf_df = set_entry_sl_tp(ltf_df, atr_buffer=atr_buffer)

        print(f"\n📊 Post structure detection for {symbol} (last few rows):")
        print(ltf_df[["final_entry", "structure_confidence", "inside_ob_zone", "bullish_engulfing", "liquidity_grab", "entry_price"]].tail(5))

    if use_ml:
        try:
            base_path = r"/Users/tnguyen287/Documents/quant_model_v1.6/machine_learning"
            clf_model = load(os.path.join(base_path, "ml_model_clf.pkl"))
            clf_scaler = load(os.path.join(base_path, "ml_model_clf_scaler.pkl"))
            reg_model = load(os.path.join(base_path, "ml_model_reg.pkl"))
            reg_scaler = load(os.path.join(base_path, "ml_model_reg_scaler.pkl"))

            ml_candidates = ltf_df[ltf_df["final_entry"] == 1].copy()
            if not ml_candidates.empty:
                ml_scored = apply_ml_models(ml_candidates, clf_model, clf_scaler, reg_model, reg_scaler)
                ltf_df.loc[ml_scored.index, ["ml_valid_entry", "ml_prob", "ml_r_pred"]] = ml_scored[["ml_valid_entry", "ml_prob", "ml_r_pred"]]

            for col, default in [("ml_valid_entry", 0), ("ml_prob", 0.0), ("ml_r_pred", 0.0)]:
                if col not in ltf_df.columns:
                    ltf_df[col] = default
                else:
                    ltf_df[col] = ltf_df[col].fillna(default)
            ltf_df["ml_prob"] = ltf_df.get("ml_prob", 0.0).fillna(0.0)
            ltf_df["ml_r_pred"] = ltf_df.get("ml_r_pred", 0.0).fillna(0.0)
        except Exception as e:
            print(f"\n⚠️ ML model error: {e}")
            ltf_df["ml_valid_entry"] = 0
            ltf_df["ml_prob"] = 0.0
            ltf_df["ml_r_pred"] = 0.0

    ltf_df["is_closed_bar"] = ltf_df.index < now - timedelta(minutes=1)
    ltf_df["trend_align"] = (
        ((ltf_df["ote_dir"] == "bullish") & (ltf_df["trend_bias"] == "bullish")) |
        ((ltf_df["ote_dir"] == "bearish") & (ltf_df["trend_bias"] == "bearish"))
    )

    # === Debug: Check what entries passed before final filtering ===
    print(f"\n🧪 Candidates before ML filtering for {symbol}:")
    print(ltf_df[ltf_df["final_entry"] == 1][[
        "ml_prob", "ml_r_pred", "ml_valid_entry", "ote_dir", "trend_bias", "is_closed_bar"
    ]].tail(5))  # tail(5) gives you a few more rows

    ltf_df["ml_entry_approve"] = (
        (ltf_df["final_entry"] == 1) &
        (ltf_df["ml_valid_entry"] == 1) &
        (ltf_df["ml_prob"] >= ml_prob_threshold) &
        (ltf_df["ml_r_pred"] >= ml_r_threshold) &
        (ltf_df["is_closed_bar"]) &
        (ltf_df["trend_align"])
    )

    signal_df = ltf_df[ltf_df["ml_entry_approve"] == 1].copy().tail(1)

    if not signal_df.empty:
        print("\n✅ LIVE SIGNAL FOUND:")
        print(signal_df[["timestamp", "entry_price", "stop_loss", "ml_prob", "ml_r_pred", "structure_confidence"]])

        if execute_trade:
            try:
                row = signal_df.iloc[0]
                qty = calculate_position_size(
                    row["entry_price"], row["stop_loss"],
                    account_balance, 0.02,
                    row["ml_prob"], row["ml_r_pred"]
                )
                limit_price = round(row["entry_price"] * (1 + 0.001), 2)

                alpaca_client.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="buy",
                    type="limit",
                    limit_price=limit_price,
                    time_in_force="gtc"
                )
                print(f"\n🚀 Limit order placed for {symbol} at {limit_price} with quantity {qty}.")
                print_trade_summary(symbol, row, qty)

                # Log to open_trades.csv
                open_log = "open_trades.csv"
                trade_row = {
                    "timestamp": row.name,
                    "symbol": symbol,
                    "entry_price": row["entry_price"],
                    "stop_loss": row["stop_loss"],
                    "take_profit": row.get("take_profit", np.nan),
                    "qty": qty,
                    "ml_prob": row["ml_prob"],
                    "ml_r_pred": row["ml_r_pred"],
                    "structure_confidence": row["structure_confidence"],
                    "status": "open"
                }
                header = not os.path.exists(open_log)
                pd.DataFrame([trade_row]).to_csv(open_log, mode="a", header=header, index=False)
                print(f"📁 Trade recorded to {open_log} (awaiting close)")

            except Exception as e:
                print(f"\n❌ Trade execution failed: {e}")
    else:
        print("\n🚫 No valid trade signal found.")

    return signal_df

def get_alpaca_client(paper=True):
    if paper:
        base_url = 'https://paper-api.alpaca.markets/v2'
        key_id = 'PK5W4A1DDSKJ41IN45I4'
        secret_key = 'vfrNwYa5Zcp7OAfwnRYr7bKo6CKIo2LVKKiKGZPH'
    else:
        base_url = 'https://paper-api.alpaca.markets/v2'
        key_id = 'PK5W4A1DDSKJ41IN45I4'
        secret_key = 'vfrNwYa5Zcp7OAfwnRYr7bKo6CKIo2LVKKiKGZPH' 

    return tradeapi.REST(key_id, secret_key, base_url, api_version="v2")

def check_and_close_trades(alpaca_client, log_path="executed_trades_log.csv"):
    open_path = "open_trades.csv"
    if not os.path.exists(open_path):
        return

    df = pd.read_csv(open_path, parse_dates=["timestamp"])
    remaining = []
    closed = []

    for _, row in df.iterrows():
        symbol = row["symbol"]
        qty = row["qty"]
        entry = row["entry_price"]
        sl = row["stop_loss"]
        tp = row.get("take_profit", None)

        try:
            last_price = alpaca_client.get_last_trade(symbol).price
        except Exception as e:
            print(f"⚠️ Could not get last price for {symbol}: {e}")
            remaining.append(row)
            continue

        exit_reason = None
        if last_price <= sl:
            exit_reason = "SL"
            exit_price = sl
        elif tp is not None and last_price >= tp:
            exit_reason = "TP"
            exit_price = tp

        if exit_reason:
            risk = abs(entry - sl)
            r_multiple = round((exit_price - entry) / risk, 2) if risk != 0 else 0
            closed.append({
                **row,
                "exit_price": exit_price,
                "exit_time": datetime.now(timezone.utc),
                "result": exit_reason,
                "r_multiple": r_multiple,
                "status": "closed"
            })
        else:
            remaining.append(row)

    pd.DataFrame(remaining).to_csv(open_path, index=False)

    if closed:
        header = not os.path.exists(log_path)
        pd.DataFrame(closed).to_csv(log_path, mode="a", header=header, index=False)
        print(f"📁 {len(closed)} trade(s) closed and logged to {log_path}")
