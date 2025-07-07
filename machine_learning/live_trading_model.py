import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from joblib import load
from ml_utils_combined import apply_ml_models
import sys
import alpaca_trade_api as tradeapi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_func.quant_model_func_v1_5 import (
    detect_htf_structure, calculate_ote_from_bos, detect_order_blocks, assign_entry_price,
    detect_ltf_structure, mark_ltf_confirmation, detect_bullish_engulfing,
    detect_liquidity_grab, mark_inside_zones, filter_valid_entries,
    set_entry_sl_tp, classify_trend_bias
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
    ltf_interval="1Min",
    lookback_minutes=60,
    confidence_threshold=0.6,
    ml_prob_threshold=0.7,
    ml_r_threshold=1.0,
    atr_buffer=0.0,
    use_ml=True,
    verbose=True,
    execute_trade=False,
    trade_qty=1,
    account_balance=10000,
    max_spread_pct=0.002,
    min_volume=100
):
    def to_rfc3339(dt): return dt.replace(microsecond=0).isoformat() + "Z"
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

    now = datetime.utcnow()
    start_time = now - timedelta(minutes=lookback_minutes)
    start_str = to_rfc3339(start_time)
    conv_symbol = convert_symbol(symbol)

    try:
        is_c = is_crypto(symbol)
        if is_c:
            df = alpaca_client.get_crypto_bars(conv_symbol, ltf_interval, start=start_str).df
            df = df[df["exchange"] == "CBSE"] if "exchange" in df.columns else df
        else:
            df = alpaca_client.get_bars(conv_symbol, ltf_interval, start=start_str).df
        df = df.rename(columns=str.title)
    except Exception as e:
        print(f"\n❌ Error fetching live data: {e}")
        return None

    if df.empty:
        print("\n⚠️ No data received.")
        return None

    if df["Volume"].iloc[-1] < min_volume:
        print(f"\n⚠️ Volume too low: {df['Volume'].iloc[-1]}")
        return None

    if not is_spread_acceptable(df):
        print("\n⚠️ Spread too wide, skipping trade.")
        return None

    ltf_df = df.copy()

    htf_df = ltf_df.resample("1H").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()

    htf_df = detect_htf_structure(htf_df)
    htf_df = classify_trend_bias(htf_df)
    htf_df = calculate_ote_from_bos(htf_df)
    htf_df = detect_order_blocks(htf_df)

    latest = htf_df[["ote_start", "ote_end", "ote_dir", "ob_high", "ob_low", "ob_direction", "structure_confidence", "trend_bias"]].ffill().iloc[-1:]
    for col in latest.columns:
        ltf_df[col] = latest[col].values[0]

    ltf_df = detect_ltf_structure(ltf_df)
    ltf_df = mark_ltf_confirmation(ltf_df, max_lag=30)
    ltf_df = detect_liquidity_grab(ltf_df)
    ltf_df = detect_bullish_engulfing(ltf_df)
    ltf_df = assign_entry_price(ltf_df, method="Low")
    ltf_df = mark_inside_zones(ltf_df)
    ltf_df = filter_valid_entries(ltf_df, confidence_threshold=confidence_threshold)
    ltf_df = set_entry_sl_tp(ltf_df, atr_buffer=atr_buffer)

    if use_ml:
        try:
            base_path = r"c:\\Users\\YoungBossTrungNguyen\\OneDrive\\Documents\\Quant_trading_modelV1.6\\machine_learning"
            clf_model = load(os.path.join(base_path, "ml_model_clf.pkl"))
            clf_scaler = load(os.path.join(base_path, "ml_model_clf_scaler.pkl"))
            reg_model = load(os.path.join(base_path, "ml_model_reg.pkl"))
            reg_scaler = load(os.path.join(base_path, "ml_model_reg_scaler.pkl"))

            ml_candidates = ltf_df[ltf_df["is_valid_entry"] == 1].copy()
            if not ml_candidates.empty:
                ml_scored = apply_ml_models(ml_candidates, clf_model, clf_scaler, reg_model, reg_scaler)
                ltf_df.loc[ml_scored.index, ["ml_valid_entry", "ml_prob", "ml_r_pred"]] = ml_scored[["ml_valid_entry", "ml_prob", "ml_r_pred"]]

            ltf_df["ml_valid_entry"] = ltf_df.get("ml_valid_entry", 0).fillna(0)
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

    ltf_df["final_entry"] = (
        (ltf_df["is_valid_entry"] == 1) &
        (ltf_df["ml_valid_entry"] == 1) &
        (ltf_df["ml_prob"] >= ml_prob_threshold) &
        (ltf_df["ml_r_pred"] >= ml_r_threshold) &
        (ltf_df["is_closed_bar"]) &
        (ltf_df["trend_align"])
    )

    signal_df = ltf_df[ltf_df["final_entry"] == 1].copy().tail(1)

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
        base_url = "https://paper-api.alpaca.markets"
        key_id = os.getenv("ALPACA_PAPER_KEY_ID")
        secret_key = os.getenv("ALPACA_PAPER_SECRET_KEY")
    else:
        base_url = "https://api.alpaca.markets"
        key_id = os.getenv("ALPACA_LIVE_KEY_ID")
        secret_key = os.getenv("ALPACA_LIVE_SECRET_KEY")

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
                "exit_time": datetime.utcnow(),
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
