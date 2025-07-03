# main.py

from alpaca_backtest_func import get_alpaca_client, run_backtest_pipeline, summarize_and_export
import time

# --- API Credentials ---
API_KEY = 'PK9GSSRVIH47LIFF31ZM'
API_SECRET = 'gpCWkS7VKNtVaNiYMizTyEgcAJl37YH5oNaCee6R'

# --- Symbols to Backtest ---
symbols = [
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD",
    # Stocks
    "TSLA", "MSFT", "NVDA"
]

# --- Prompt User for Timeframes ---
htf_period = input("Enter HTF period (e.g., 3mo, 6mo, 12mo): ").strip()
ltf_period = input("Enter LTF period (e.g., 30d, 60d, 90d): ").strip()

# --- Output Paths ---
summary_log = "ML_backtest_summary_log.csv"
trade_log = "ML_trade_backtest_master_log.csv"

# --- Initialize Alpaca Client ---
alpaca = get_alpaca_client(API_KEY, API_SECRET)

# --- Run Main Pipeline ---
if __name__ == "__main__":
    for symbol in symbols:
        print(f"\n🚀 Backtest: {symbol} | LTF: {ltf_period}, HTF: {htf_period}")
        start_time = time.time()

        try:
            df = run_backtest_pipeline(
                symbol=symbol,
                alpaca_client=alpaca,
                ltf_period=ltf_period,
                htf_period=htf_period,
                verbose=True
            )

            summarize_and_export(
                df=df,
                symbol=symbol,
                path=trade_log,
                summary_path=summary_log
            )

        except Exception as e:
            print(f"❌ Failed on {symbol} ({htf_period} HTF): {e}")

        finally:
            elapsed = round(time.time() - start_time, 2)
            print(f"⏱️ Elapsed: {elapsed}s")

