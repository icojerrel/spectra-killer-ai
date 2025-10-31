#!/usr/bin/env python3
"""
MT5 CONNECTIVITY TEST
Verify account 5041139909 is accessible
"""

import MetaTrader5 as mt5
import sys

print("="*70)
print("🔗 MT5 CONNECTIVITY TEST")
print("="*70)

print("\n1️⃣ Initializing MT5...")
if not mt5.initialize():
    print(f"❌ FAILED: {mt5.last_error()}")
    sys.exit(1)

print("✅ MT5 initialized")

print("\n2️⃣ Getting account info...")
account_info = mt5.account_info()

if account_info is None:
    print(f"❌ FAILED: {mt5.last_error()}")
    mt5.shutdown()
    sys.exit(1)

print(f"✅ Account connected!")
print(f"   Login: {account_info.login}")
print(f"   Balance: ${account_info.balance:.2f}")
print(f"   Equity: ${account_info.equity:.2f}")
print(f"   Margin Free: ${account_info.margin_free:.2f}")
print(f"   Leverage: 1:{account_info.leverage}")
print(f"   Currency: {account_info.currency}")

print("\n3️⃣ Testing XAUUSD symbol...")
symbol_info = mt5.symbol_info("XAUUSD")
if symbol_info is None:
    print(f"❌ Symbol not found: {mt5.last_error()}")
    mt5.shutdown()
    sys.exit(1)

print(f"✅ Symbol accessible!")

print("\n4️⃣ Fetching market data...")
tick = mt5.symbol_info_tick("XAUUSD")
rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 10)

print(f"✅ Market data available!")
print(f"   Ask: ${tick.ask:.2f}")
print(f"   Bid: ${tick.bid:.2f}")
print(f"   Spread: {(tick.ask - tick.bid) * 100:.2f} pips")
print(f"   Last candles: {len(rates)}")

print("\n5️⃣ Checking positions...")
positions = mt5.positions_get()
print(f"✅ Open positions: {len(positions) if positions else 0}")
if positions:
    for pos in positions:
        print(f"   - #{pos.ticket}: {pos.symbol} {pos.type} @ ${pos.price_open:.2f}")

print("\n" + "="*70)
print("✅ MT5 IS FULLY OPERATIONAL")
print("="*70)

mt5.shutdown()
