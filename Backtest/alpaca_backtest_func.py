# alpaca_backtest_func.py (fixed)

import os
import sys
import pandas as pd
from datetime import datetime
from alpaca_trade_api.rest import REST
from joblib import load

# Add root directory to sys.path for clean relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Import quant model core functions ===
from model_func.quant_model_func_v1_5 import (
    detect_htf_structure, calculate_ote_from_bos, detect_order_blocks, assign_entry_price,
    detect_ltf_structure, mark_ltf_confirmation, detect_bullish_engulfing,
    detect_liquidity_grab, mark_inside_zones, filter_valid_entries,
    set_entry_sl_tp, simulate_trade_outcomes, classify_trend_bias
)

from machine_learning.ml_utils_combined import apply_ml_models

# --- Alpaca API Setup ---
def get_alpaca_client(api_key, secret_key):
    base_url = "https://paper-api.alpaca.markets/v2"
    return REST(api_key, secret_key, base_url=base_url)

# --- Helpers ---
def is_crypto(symbol):
    return "-USD" in symbol

def convert_symbol(symbol):
    if is_crypto(symbol):
        return symbol.replace("-USD", "/USD")
    if symbol.endswith("=X"):
        return symbol.replace("=X", "") + "/USD"
    return symbol

def fetch_data(alpaca, symbol, interval, start, crypto=True):
    if crypto:
        bars = alpaca.get_crypto_bars(symbol, interval, start=start).df
        if bars.empty:
            raise ValueError(f"No crypto data returned for {symbol}")
        return bars if "exchange" not in bars.columns else bars[bars["exchange"] == "CBSE"].drop(columns=["exchange"])
    return alpaca.get_bars(symbol, interval, start=start).df

def convert_to_days(period):
    return str(int(period[:-2]) * 30) + "d" if period.endswith("mo") else period

def to_rfc3339(dt):
    return dt.replace(microsecond=0).isoformat() + "Z"

# --- Data Downloader ---
def download_data(alpaca, symbol, ltf_interval="15Min", ltf_period="60d", htf_interval="1H", htf_period="6mo"):
    now = datetime.utcnow()
    ltf_start = now - pd.Timedelta(convert_to_days(ltf_period))
    htf_start = now - pd.Timedelta(convert_to_days(htf_period))
    ltf_str, htf_str = to_rfc3339(ltf_start), to_rfc3339(htf_start)
    sym_conv = convert_symbol(symbol)
    is_c = is_crypto(symbol)
    ltf = fetch_data(alpaca, sym_conv, ltf_interval, ltf_str, is_c)
    htf = fetch_data(alpaca, sym_conv, htf_interval, htf_str, is_c)
    return ltf.rename(columns=str.title), htf.rename(columns=str.title)

# --- Backtest Logic ---
def calculate_holding_time(df):
    df = df.copy()
    df["holding_bars"] = None
    for i in range(len(df)):
        if pd.notna(df.at[df.index[i], "exit_price"]):
            for j in range(i + 1, min(i + 100, len(df))):
                if pd.notna(df.at[df.index[j], "exit_price"]):
                    df.at[df.index[i], "holding_bars"] = j - i
                    break
    return df

import os
import numpy as np
from joblib import load
from machine_learning.ml_utils_combined import apply_ml_models

import os
import numpy as np
from joblib import load

import os
from joblib import load
from machine_learning.ml_utils_combined import apply_ml_models

def run_backtest_pipeline(
    symbol, alpaca_client,
    ltf_interval="15Min", ltf_period="60d",
    htf_interval="1H", htf_period="6mo",
    confidence_threshold=0.6,
    future_window=20, atr_buffer=0.0,
    verbose=True,
    use_ml=True
):
    if verbose:
        print(f"\n🔁 Running backtest for {symbol}")

    # === Download + Process Data ===
    ltf_df, htf_df = download_data(alpaca_client, symbol, ltf_interval, ltf_period, htf_interval, htf_period)

    htf_df = detect_htf_structure(htf_df)
    htf_df = classify_trend_bias(htf_df)
    htf_df = calculate_ote_from_bos(htf_df)
    htf_df = detect_order_blocks(htf_df)

    latest = htf_df[[
        "ote_start", "ote_end", "ote_dir",
        "ob_high", "ob_low", "ob_direction",
        "structure_confidence"
    ]]
    ltf_df = ltf_df.merge(latest, left_index=True, right_index=True, how="left").ffill()

    ltf_df = detect_ltf_structure(ltf_df)
    ltf_df["lt_structure"] = ltf_df["market_structure"]

    if verbose:
        print("📊 LTF structure counts:")
        print(ltf_df["market_structure"].value_counts(dropna=False))

    # === Signal Detection and Entry Simulation ===
    ltf_df = mark_ltf_confirmation(ltf_df, max_lag=30)
    ltf_df = detect_liquidity_grab(ltf_df)
    ltf_df = detect_bullish_engulfing(ltf_df)
    ltf_df = assign_entry_price(ltf_df, method="Low")
    ltf_df = mark_inside_zones(ltf_df)
    ltf_df = filter_valid_entries(ltf_df, confidence_threshold=confidence_threshold)
    ltf_df = set_entry_sl_tp(ltf_df, atr_buffer=atr_buffer)
    ltf_df = simulate_trade_outcomes(ltf_df, future_window=future_window)
    ltf_df = calculate_holding_time(ltf_df)

    # === Final Column Setup ===
    ltf_df["timestamp"] = ltf_df.index
    ltf_df["setup_type"] = ltf_df.get("setup_type", "unknown")
    ltf_df["trend_bias"] = ltf_df.get("trend_bias", "unknown")
    ltf_df["entry_zone_type"] = ltf_df.get("entry_zone_type", "none")
    ltf_df["is_valid_entry"] = ltf_df.get("is_valid_entry", False).astype(int)
    ltf_df["r_multiple"] = ltf_df["r_multiple"].fillna(0).infer_objects(copy=False)
    ltf_df["is_profitable"] = (ltf_df["r_multiple"] > 0).astype(int)

    # === ML Enhancement ===
    if use_ml:
        try:
            base_path = r"c:\Users\YoungBossTrungNguyen\OneDrive\Documents\Quant_trading_modelV1.6\machine_learning"
            clf_model = load(os.path.join(base_path, "ml_model_clf.pkl"))
            clf_scaler = load(os.path.join(base_path, "ml_model_clf_scaler.pkl"))
            reg_model = load(os.path.join(base_path, "ml_model_reg.pkl"))
            reg_scaler = load(os.path.join(base_path, "ml_model_reg_scaler.pkl"))

            ml_candidates = ltf_df[ltf_df["is_valid_entry"] == 1].copy()

            if not ml_candidates.empty:
                ml_scored = apply_ml_models(ml_candidates, clf_model, clf_scaler, reg_model, reg_scaler)

                # Initialize columns
                ltf_df["ml_valid_entry"] = 0
                ltf_df["ml_prob"] = 0.0
                ltf_df["ml_r_pred"] = 0.0

                # Fill ML scores
                ltf_df.loc[ml_scored.index, ["ml_valid_entry", "ml_prob", "ml_r_pred"]] = ml_scored[
                    ["ml_valid_entry", "ml_prob", "ml_r_pred"]
                ]
            else:
                ltf_df["ml_valid_entry"] = 0
                ltf_df["ml_prob"] = 0.0
                ltf_df["ml_r_pred"] = 0.0

            # Final hybrid filter
            ltf_df["final_entry"] = (ltf_df["is_valid_entry"] == 1) & (ltf_df["ml_valid_entry"] == 1)

        except Exception as e:
            print("⚠️ ML model could not be loaded/applied:", str(e))
            ltf_df["ml_valid_entry"] = 0
            ltf_df["ml_prob"] = 0.0
            ltf_df["ml_r_pred"] = 0.0
            ltf_df["final_entry"] = ltf_df["is_valid_entry"] == 1
    else:
        ltf_df["ml_valid_entry"] = 0
        ltf_df["ml_prob"] = 0.0
        ltf_df["ml_r_pred"] = 0.0
        ltf_df["final_entry"] = ltf_df["is_valid_entry"] == 1

    # === Summary ===
    if verbose:
        total = len(ltf_df)
        valid_trades = ltf_df["is_valid_entry"].sum()
        ml_approved = ltf_df["ml_valid_entry"].sum()
        final_count = ltf_df["final_entry"].sum()
        avg_r = round(ltf_df[ltf_df["final_entry"] == 1]["r_multiple"].mean(), 2)
        print(f"✅ Backtest complete: {valid_trades} rule trades, {ml_approved} ML trades, {final_count} hybrid entries, avg R = {avg_r}")

    return ltf_df



# --- Export + Summary Combined ---
def summarize_and_export(
    df, symbol,
    path="",
    summary_path=""
):
    df = df.copy()
    df["timestamp"] = df.index
    df["symbol"] = symbol
    df["htf_trend"] = df.get("ote_dir", "unknown")

    # Normalize binary flags
    for col in ["choch_before", "bos_after_choch", "liquidity_grab", "bullish_engulfing", "is_valid_entry", "ml_valid_entry", "final_entry"]:
        df[col] = df.get(col, False).fillna(False).astype(int)

    df["confirmation_count"] = df[["choch_before", "bos_after_choch", "liquidity_grab", "bullish_engulfing"]].sum(axis=1)
    df["trend_strength"] = df.get("structure_confidence", 0.0).apply(
        lambda x: "strong" if x >= 0.75 else "moderate" if x >= 0.5 else "weak"
    )
    df["r_multiple"] = df["r_multiple"].fillna(0)
    df["is_profitable"] = (df["r_multiple"] > 0).astype(int)

    # Filter only final entries
    entry_df = df[df["final_entry"] == 1].copy()

    export_cols = [
        "timestamp", "symbol", "setup_type", "is_valid_entry", "structure_confidence",
        "confirmation_count", "r_multiple", "inside_ote_zone", "trend_bias",
        "inside_ob_zone", "is_profitable", "htf_trend", "choch_before", "bos_after_choch",
        "liquidity_grab", "bullish_engulfing", "exit_reason", "trend_strength", "holding_bars",
        "entry_price", "exit_price", "stop_loss", "ob_low", "ob_high", "ote_start",
        "ote_end", "entry_zone_type",
        "ml_valid_entry", "ml_prob", "ml_r_pred", "final_entry"
    ]

    for col in export_cols:
        if col not in entry_df.columns:
            entry_df[col] = None

    header = not os.path.exists(path)
    entry_df.to_csv(path, mode='a', header=header, index=False)
    print(f"✅ Exported {len(entry_df)} entries to {path}")

    # === Summary ===
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "total_trades": len(entry_df),
        "wins": (entry_df["r_multiple"] > 0).sum(),
        "losses": (entry_df["r_multiple"] <= 0).sum(),
        "win_rate (%)": round((entry_df["r_multiple"] > 0).mean() * 100, 2) if len(entry_df) > 0 else 0,
        "net_r": round(entry_df["r_multiple"].sum(), 2),
        "avg_r": round(entry_df["r_multiple"].mean(), 2) if len(entry_df) > 0 else 0,
        "avg_holding_bars": round(entry_df["holding_bars"].mean(), 2) if len(entry_df) > 0 else 0
    }

    summary_header = not os.path.exists(summary_path)
    pd.DataFrame([summary]).to_csv(summary_path, mode="a", header=summary_header, index=False)

    print("\n📊 ML-Enhanced Entry Summary:")
    [print(f"{k:<20}: {v}") for k, v in summary.items()]

    return summary