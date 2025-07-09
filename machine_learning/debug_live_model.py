from live_trading_model import run_live_trading_model, get_alpaca_client
import pandas as pd

# === Setup ===
symbol = "MSFT"
client = get_alpaca_client(paper=True)

# === Run model (skip trade execution) ===
result_df = run_live_trading_model(
    alpaca_client=client,
    symbol=symbol,
    execute_trade=False,
    account_balance=10000,
    use_ml=True,
    confidence_threshold=0.6,
    ml_prob_threshold=0.7,
    ml_r_threshold=1.0,
    atr_buffer=0.0
)

# === Debug Output ===
if result_df is None:
    print("\n🚫 result_df is None.")
elif result_df.empty:
    print("\n⚠️ result_df is empty.")
else:
    print("\n✅ result_df returned.")
    
    # Print all columns
    print("📋 Columns:")
    print(result_df.columns.tolist())

    # Print index type
    print("\n🕓 Index Info:")
    print(result_df.index)

    # Check if inside_ob_zone exists
    print("\n❓ 'inside_ob_zone' exists?", "inside_ob_zone" in result_df.columns)

    # Print sample if it exists
    if "inside_ob_zone" in result_df.columns:
        print("\n📊 inside_ob_zone preview:")
        print(result_df[["inside_ob_zone", "entry_price"]].tail())

