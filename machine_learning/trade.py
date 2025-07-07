import os
import time
import pandas as pd
from datetime import datetime
from live_trading_model import run_live_trading_model, get_alpaca_client
from live_trading_model import check_and_close_trades

# === Settings ===
symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "MSFT", "NVDA"]
account_balance = 100000
max_drawdown_pct = -0.04  # -4%
profit_target_pct = 0.05  # +5%
max_consecutive_losses = 2
check_interval_sec = 900
risk_per_trade_pct = 0.02

# === State Tracking ===
closed_trades_file = "executed_trades_log.csv"
total_profit = 0
consecutive_losses = 0

alpaca_client = get_alpaca_client(paper=True)

print("\n📈 Starting live paper trading loop...")

while True:
    for symbol in symbols:
        try:
            result = run_live_trading_model(
                alpaca_client=alpaca_client,
                symbol=symbol,
                execute_trade=True,
                account_balance=account_balance + total_profit,
                confidence_threshold=0.6,
                ml_prob_threshold=0.7,
                ml_r_threshold=1.0,
                atr_buffer=0.0
            )

            if result is not None and result.empty is False:
                print(f"✅ Signal for {symbol} evaluated.")

        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")

    # Step 2: Check and close any open trades
    try:
        check_and_close_trades(alpaca_client)
    except Exception as e:
        print(f"❌ Error checking open trades: {e}")

    # Step 3: Update state
    if not os.path.exists(closed_trades_file):
        time.sleep(check_interval_sec)
        continue

    try:
        df = pd.read_csv(closed_trades_file)
        df = df[df["status"] == "closed"]
        df = df.sort_values("exit_time")

        risk_per_trade = account_balance * risk_per_trade_pct
        df["pnl"] = df["r_multiple"] * risk_per_trade
        total_profit = df["pnl"].sum()

        recent_results = df["result"].tail(2).tolist()
        consecutive_losses = recent_results.count("SL") if len(set(recent_results)) == 1 and recent_results[0] == "SL" else 0

        print(f"\n💰 Net Profit: ${total_profit:.2f} | Drawdown: {total_profit/account_balance:.2%} | Loss streak: {consecutive_losses}")

        if total_profit / account_balance <= max_drawdown_pct:
            print("\n❌ Max drawdown hit. Stopping.")
            break
        if total_profit / account_balance >= profit_target_pct:
            print("\n✅ Profit target hit. Stopping.")
            break
        if consecutive_losses >= max_consecutive_losses:
            print("\n⚠️ Max consecutive losses hit. Stopping.")
            break


    except Exception as e:
        print(f"❌ Error updating state: {e}")

    time.sleep(check_interval_sec)