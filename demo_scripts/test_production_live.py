#!/usr/bin/env python3
"""
PRODUCTION TEST - 1 REAL CYCLE
Test grok4_production.py met echte MT5
"""

import sys
sys.path.insert(0, r'C:\Users\Gebruiker\Desktop\spectra_killer_ai')

from grok4_production import Grok4Orchestrator, RiskConfig

print("="*70)
print("🚀 GROK4 PRODUCTION TEST - 1 REAL CYCLE")
print("="*70)

# Configure risk
risk_config = RiskConfig(
    max_position_size=0.5,  # Conservative
    max_daily_trades=5,
    max_risk_per_trade=0.01,
    min_confidence=0.55
)

# Initialize
api_key = "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"

try:
    print("\n1️⃣  Initializing Grok4 Orchestrator...")
    orchestrator = Grok4Orchestrator(
        account_id=5041139909,
        grok_api_key=api_key,
        risk_config=risk_config
    )
    print("✅ Initialized")
    
    print("\n2️⃣  Running single trading cycle...")
    result = orchestrator.run_cycle()
    
    print("\n3️⃣  CYCLE RESULT:")
    print(f"   Status: {result['status']}")
    print(f"   Timestamp: {result['timestamp']}")
    
    if result['market_data']:
        print(f"\n   Market Data:")
        print(f"      Price: ${result['market_data']['current_price']:.2f}")
        print(f"      MA20: ${result['market_data']['ma20']:.2f}")
    
    if result['decision']:
        print(f"\n   Grok Decision:")
        print(f"      Signal: {result['decision']['decision']}")
        print(f"      Confidence: {result['decision']['confidence']:.1%}")
        print(f"      Reasoning: {result['decision']['reasoning']}")
    
    if result['trade_ticket']:
        print(f"\n   ✅ TRADE EXECUTED:")
        print(f"      Ticket: {result['trade_ticket']}")
    else:
        print(f"\n   ⏸️  No trade executed")
    
    print("\n" + "="*70)
    print("✅ PRODUCTION TEST COMPLETE")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\n🔌 Shutting down...")
