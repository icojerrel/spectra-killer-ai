#!/usr/bin/env python3
# Simple MT5 connection test

import sys
print("Testing MetaTrader5 library...")

try:
    import MetaTrader5 as mt5
    print("[OK] MetaTrader5 library imported successfully")
    
    # Test initialization
    print("Testing MT5 initialization...")
    result = mt5.initialize()
    
    if result:
        print("[OK] MT5 initialized successfully")
        
        # Get terminal info
        terminal_info = mt5.terminal_info()
        if terminal_info:
            print(f"[OK] Terminal: {terminal_info.name} v{terminal_info.version}")
            print(f"[OK] Broker: {terminal_info.company}")
            print(f"[OK] Account: {terminal_info.login}")
        else:
            print("[ERROR] Could not get terminal info")
        
        # Check for XAUUSD symbol
        symbol_info = mt5.symbol_info("XAUUSD")
        if symbol_info:
            print("[OK] XAUUSD symbol available")
            print(f"   Digits: {symbol_info.digits}")
            print(f"   Point: {symbol_info.point}")
        else:
            print("[ERROR] XAUUSD symbol not found")
        
        # Shutdown
        mt5.shutdown()
        print("[OK] MT5 shutdown completed")
        
    else:
        print("[ERROR] MT5 initialization failed")
        print("   Please ensure:")
        print("   1. MT5 terminal is installed")
        print("   2. MT5 terminal is running")
        print("   3. Algo Trading is enabled")
        
except ImportError as e:
    print(f"[ERROR] Failed to import MetaTrader5: {e}")
    print("   Please install with: pip install MetaTrader5")
except Exception as e:
    print(f"[ERROR] Error: {e}")

print("MT5 test completed.")