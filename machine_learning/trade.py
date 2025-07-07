import os
import time
import pandas as pd
from datetime import datetime
from live_trading_model import run_live_trading_model, get_alpaca_client, check_and_close_trades
import pytz
import math

# === Settings ===
symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "MSFT", "NVDA"]
account_balance = 100000
max_drawdown_pct = -0.04   # -4%
profit_target_pct = 0.05   # +5%
max_consecutive_losses = 2
check_interval_sec = 900   # 15 minutes
risk_per_trade_pct = 0.02
closed_trades_file = "executed_trades_log.csv"

def wait_until_next_15_min():
    now = datetime.now().astimezone(pytz.timezone("America/New_York"))
    # Round up to the next multiple of 15
    next_minute = ((now.minute // 15) + 1) * 15
    next_hour = now.hour
    if next_minute == 60:
        next_minute = 0
        next_hour = (now.hour + 1) % 24

    next_tick = now.replace(minute=next_minute, second=0, microsecond=0, hour=next_hour)
    wait_time = (next_tick - now).total_seconds()

    print(f"⏳ Waiting {int(wait_time)} seconds until next 15-minute candle ({next_tick.strftime('%H:%M')})...")
    time.sleep(wait_time)


# === Timezone ===
et_tz = pytz.timezone("America/New_York")
def to_est(ts): return ts.astimezone(et_tz)

# === State Tracking ===
total_profit = 0
consecutive_losses = 0
last_printed_ts_by_symbol = {}

# === Start Client ===
alpaca_client = get_alpaca_client(paper=True)
print("\n📈 Starting live paper trading loop...")

# === Main Loop ===
while True:
    et_now = datetime.now(et_tz)
    print(f"\n🕒 Loop running at {et_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    for symbol in symbols:
        now = datetime.now(et_tz)

        # 🕐 Time-based Filter
        if symbol in ["MSFT", "NVDA"]:
            if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 16:
                print(f"⏳ Skipping {symbol}: outside US stock hours ({now.strftime('%H:%M')})")
                continue
        elif symbol in ["BTC-USD", "ETH-USD", "SOL-USD"]:
            if now.hour not in list(range(2, 5)) + list(range(9, 18)):
                print(f"🌙 Skipping {symbol}: outside active crypto hours ({now.strftime('%H:%M')})")
                continue

        # 📊 Run Trading Logic
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

            if result is not None and not result.empty:
                latest_ts = result.index[-1]
                if last_printed_ts_by_symbol.get(symbol) == latest_ts:
                    print(f"🔁 Skipping duplicate signal for {symbol} at {latest_ts}")
                    continue
                last_printed_ts_by_symbol[symbol] = latest_ts
                print(f"✅ Signal for {symbol} evaluated at {latest_ts.strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")

    # 🔒 Check Open Trades
    try:
        check_and_close_trades(alpaca_client)
    except Exception as e:
        print(f"❌ Error checking open trades: {e}")

    # 💰 Update PnL and Stop Conditions
    if not os.path.exists(closed_trades_file):
        print("⚠️ Closed trades file not found. Skipping PnL update.")
        wait_until_next_15_min()
        continue

    try:
        df = pd.read_csv(closed_trades_file, parse_dates=["exit_time"])
        df = df[df["status"] == "closed"].sort_values("exit_time")

        risk_per_trade = account_balance * risk_per_trade_pct
        df["pnl"] = df["r_multiple"] * risk_per_trade
        total_profit = df["pnl"].sum()

        recent_results = df["result"].tail(2).tolist()
        consecutive_losses = recent_results.count("SL") if recent_results == ["SL", "SL"] else 0

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

    wait_until_next_15_min()
    et_now = datetime.now(et_tz)
    print(f"\n🕒 Loop running at {et_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")



