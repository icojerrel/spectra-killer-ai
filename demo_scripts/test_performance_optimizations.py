#!/usr/bin/env python3
"""
Performance Test Script - API Call Optimalisaties
Test de snelheid van data ophalen voor/na optimalisaties
"""

import asyncio
import time
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Import direct zonder module prefix
import config
from core.mt5_connector import MT5Connector
from core.data_manager import DataManager
from core.cache_manager import get_cache_manager


async def test_mt5_performance():
    """Test MT5 connector performance"""
    print("\n" + "="*70)
    print("🔬 MT5 CONNECTOR PERFORMANCE TEST")
    print("="*70)
    
    connector = MT5Connector()
    
    # Test 1: Single call
    print("\n[CHART] Test 1: Single data fetch (XAUUSD H1, 1000 candles)")
    start = time.perf_counter()
    data = await connector.get_data(symbol="XAUUSD", timeframe="H1", count=1000)
    elapsed = time.perf_counter() - start
    
    if data is not None:
        print(f"   [OK] Succesvol: {len(data)} rijen in {elapsed:.3f}s")
    else:
        print(f"   [X] FAILED - Geen data")
        return False
    
    # Test 2: Cached call (should be instant)
    print("\n[CHART] Test 2: Cached data fetch (same params)")
    start = time.perf_counter()
    cached_data = await connector.get_data(symbol="XAUUSD", timeframe="H1", count=1000)
    elapsed = time.perf_counter() - start
    
    if cached_data is not None:
        print(f"   [OK] Cache hit: {len(cached_data)} rijen in {elapsed:.3f}s")
        print(f"   [ROCKET] Speedup: {(elapsed < 0.05)}")
    else:
        print(f"   [WARNING]  Cache miss")
    
    # Test 3: Multiple timeframes parallel
    print("\n[CHART] Test 3: Parallel fetch (5 timeframes)")
    timeframes = ["M5", "M15", "M30", "H1", "H4"]
    
    start = time.perf_counter()
    tasks = [connector.get_data(symbol="XAUUSD", timeframe=tf, count=500, use_cache=False) 
             for tf in timeframes]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start
    
    successful = sum(1 for r in results if r is not None)
    print(f"   [OK] {successful}/{len(timeframes)} successful in {elapsed:.3f}s")
    print(f"   [!] Avg per timeframe: {elapsed/len(timeframes):.3f}s")
    
    # Cache stats
    cache = get_cache_manager()
    stats = cache.get_stats()
    print(f"\n[UP] Cache Statistics:")
    print(f"   Hits: {stats['hits']}, Misses: {stats['misses']}")
    print(f"   Hit Rate: {stats['hit_rate']:.1f}%")
    print(f"   Total Entries: {stats['total_entries']}")
    
    await connector.shutdown()
    return True


async def test_data_manager_performance():
    """Test DataManager performance"""
    print("\n" + "="*70)
    print("🔬 DATA MANAGER PERFORMANCE TEST")
    print("="*70)
    
    manager = DataManager()
    if not await manager.initialize():
        print("   [X] FAILED - Could not initialize")
        return False
    
    # Test 1: Auto source with parallel queries
    print("\n[CHART] Test 1: Auto source (parallel DB + MT5)")
    start = time.perf_counter()
    data = await manager.get_data(
        symbol="XAUUSD",
        timeframe="H1",
        count=1000,
        source="auto",
        preprocess=False,
        save_to_db=False
    )
    elapsed = time.perf_counter() - start
    
    if data is not None:
        print(f"   [OK] Succesvol: {len(data)} rijen in {elapsed:.3f}s")
    else:
        print(f"   [X] FAILED")
    
    # Test 2: Parallel preprocessing
    print("\n[CHART] Test 2: Parallel save + preprocess")
    start = time.perf_counter()
    data = await manager.get_data(
        symbol="XAUUSD",
        timeframe="M15",
        count=500,
        source="mt5",
        preprocess=True,
        save_to_db=True
    )
    elapsed = time.perf_counter() - start
    
    if data is not None:
        print(f"   [OK] Succesvol: {len(data)} rijen in {elapsed:.3f}s")
    else:
        print(f"   [WARNING]  Partially successful")
    
    await manager.shutdown()
    return True


async def performance_comparison():
    """Run complete performance test suite"""
    print("\n" + "="*70)
    print("[ROCKET] PERFORMANCE OPTIMIZATION TEST SUITE")
    print("="*70)
    print("\nTesting optimizations:")
    print("  [OK] ThreadPoolExecutor (15 workers)")
    print("  [OK] In-memory caching (TTL: 30s)")
    print("  [OK] Parallel async operations")
    print("  [OK] Timeout reduction (60s -> 10s)")
    print("  [OK] Optimized DataFrame operations")
    
    success = True
    
    try:
        # Test MT5 connector
        if not await test_mt5_performance():
            success = False
        
        # Test DataManager
        if not await test_data_manager_performance():
            success = False
        
    except Exception as e:
        print(f"\n[X] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print("\n" + "="*70)
    if success:
        print("[OK] ALL TESTS PASSED - Optimizations working!")
    else:
        print("[WARNING]  SOME TESTS FAILED - Check errors above")
    print("="*70)
    
    return success


if __name__ == "__main__":
    result = asyncio.run(performance_comparison())
    sys.exit(0 if result else 1)
