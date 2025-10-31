#!/usr/bin/env python3
"""
Test Enhanced Backtest Engine
Quick validation run
"""

import asyncio
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime, timedelta
import logging

# Test with minimal data first
async def quick_test():
    """Quick test with 1 week of data"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("QUICK TEST - Enhanced Backtest Engine")
    print("=" * 80)
    
    # Import here to check for errors
    try:
        from backtesting.enhanced_engine import EnhancedBacktestEngine
        print("[OK] Enhanced engine imported successfully")
    except Exception as e:
        print(f"[X] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create minimal config
    config = {
        'initial_balance': 10000,
        'use_mock_agents': True,  # Fast mock agents
        'enable_ml': False,  # Disable ML for quick test
        'enable_online_learning': False,
        'enable_adaptive_params': False,
    }
    
    engine = EnhancedBacktestEngine(config)
    
    try:
        # Initialize
        print("\nInitializing...")
        if not await engine.initialize():
            print("[X] Initialization failed")
            return
        
        print("[OK] Initialization successful")
        
        # Test with 1 week
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"\nTesting with {start_date.date()} to {end_date.date()}")
        
        results = await engine.run_backtest(
            symbol="XAUUSD",
            start_date=start_date,
            end_date=end_date,
            timeframe="M5"
        )
        
        if results:
            print("\n[OK] TEST PASSED!")
            print(f"Trades executed: {results.get('trading', {}).get('total_trades', 0)}")
        else:
            print("\n[X] TEST FAILED - No results")
    
    except Exception as e:
        print(f"\n[X] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await engine.cleanup()

if __name__ == "__main__":
    asyncio.run(quick_test())
