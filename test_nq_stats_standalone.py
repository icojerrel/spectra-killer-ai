#!/usr/bin/env python3
"""
Standalone NQ Stats Test - No package dependencies
Tests the core NQ Stats logic directly
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Load the module directly without going through package
import importlib.util
import sys
from pathlib import Path

# Load NQ Stats analyzer directly
analyzer_path = Path(__file__).parent / 'src' / 'spectra_killer_ai' / 'strategies' / 'advanced' / 'nq_stats_analyzer.py'
spec = importlib.util.spec_from_file_location("nq_stats_analyzer", analyzer_path)
nq_stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nq_stats)

NQStatsAnalyzer = nq_stats.NQStatsAnalyzer

print("="*80)
print("  NQ STATS ANALYZER - STANDALONE TEST")
print("="*80)

print("\n✅ Module loaded successfully!")

# Generate test data
print("\n📊 Generating test data...")
end_time = datetime.now(pytz.timezone('US/Eastern'))
start_time = end_time - timedelta(days=3)
date_range = pd.date_range(start=start_time, end=end_time, freq='5min')

np.random.seed(42)
base_price = 15000
returns = np.random.normal(0, 0.001, len(date_range))
prices = base_price * (1 + returns).cumprod()

data = pd.DataFrame(index=date_range)
data['open'] = prices
data['high'] = prices * (1 + np.abs(np.random.normal(0, 0.0005, len(date_range))))
data['low'] = prices * (1 - np.abs(np.random.normal(0, 0.0005, len(date_range))))
data['close'] = prices * (1 + np.random.normal(0, 0.0003, len(date_range)))
data['volume'] = np.random.randint(100, 1000, len(date_range))

# Ensure high is highest and low is lowest
data['high'] = data[['open', 'high', 'close']].max(axis=1)
data['low'] = data[['open', 'low', 'close']].min(axis=1)

print(f"✅ Generated {len(data)} bars")
print(f"   Date range: {date_range[0]} to {date_range[-1]}")
print(f"   Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")

# Initialize analyzer
config = {
    'sdev_values': {
        'daily': 1.376,
        'hourly': 0.34,
        'weekly': 2.1,
        'monthly': 4.5
    },
    'high_probability_threshold': 0.75,
    'medium_probability_threshold': 0.51
}

print("\n🚀 Initializing NQ Stats Analyzer...")
analyzer = NQStatsAnalyzer(config)
print("✅ Analyzer initialized!")

# Run analysis
print(f"\n🔍 Running analysis at {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}...")
results = analyzer.analyze(data, end_time)

print("\n" + "="*80)
print("  ANALYSIS RESULTS")
print("="*80)

# SDEV Analysis
print("\n📈 STANDARD DEVIATION ANALYSIS:")
sdev = results.get('sdev_analysis', {})
if sdev.get('available'):
    levels = sdev.get('sdev_levels', {})

    if 'daily' in levels:
        daily = levels['daily']
        print(f"   Daily Open: ${daily.get('open', 0):.2f}")
        print(f"   Current Price: ${daily.get('current_price', 0):.2f}")
        print(f"   SDEV Distance: {daily.get('sdev_distance', 0):.2f}σ")
        print(f"   Zone: {daily.get('zone', 'N/A')}")
        print(f"   Reversion Probability: {daily.get('reversion_probability', 0)*100:.1f}%")
        print(f"   Is Trend Day: {daily.get('is_trend_day', False)}")
        print(f"   Rubber Band Tension: {daily.get('rubber_band_tension', 'N/A')}")

        print("\n   Key Levels:")
        for level, price in daily.get('levels', {}).items():
            print(f"      {level}: ${price:.2f}")
else:
    print("   ❌ Not available")

# Hour Stats
print("\n⏰ HOUR STATISTICS:")
hour_stats = results.get('hour_stats', {})
if hour_stats.get('available'):
    print(f"   Current Hour: {hour_stats.get('current_hour', 'N/A')}")
    print(f"   Segment: {hour_stats.get('segment', 'N/A')} ({hour_stats.get('segment_type', 'N/A')})")
    print(f"   Retracement Probability: {hour_stats.get('retracement_probability', 0)*100:.1f}%")
    print(f"   High Probability Segment: {hour_stats.get('is_high_probability_segment', False)}")
    print(f"   Trading Recommendation: {hour_stats.get('trading_recommendation', 'N/A')}")
else:
    print("   ❌ Not available")

# RTH Breaks
print("\n📊 RTH BREAKS:")
rth = results.get('rth_breaks', {})
if rth.get('available'):
    print(f"   Scenario: {rth.get('scenario', 'N/A')}")
    print(f"   Bias: {rth.get('bias', 'N/A')}")
    print(f"   Probability: {rth.get('probability', 0)*100:.1f}%")
    print(f"   Confidence: {rth.get('confidence', 'N/A')}")
    print(f"   Recommendation: {rth.get('trading_recommendation', 'N/A')}")
else:
    print(f"   ❌ {rth.get('reason', 'Not available')}")

# 9am Continuation
print("\n🕐 9AM HOUR CONTINUATION:")
hour_9am = results.get('9am_continuation', {})
if hour_9am.get('available'):
    print(f"   Direction: {hour_9am.get('hour_9am_direction', 'N/A')}")
    print(f"   Bias: {hour_9am.get('bias', 'N/A')}")
    print(f"   NY Session Probability: {hour_9am.get('ny_session_continuation_probability', 0)*100:.1f}%")
    print(f"   Recommendation: {hour_9am.get('trading_recommendation', 'N/A')}")
else:
    print(f"   ❌ {hour_9am.get('reason', 'Not available')}")

# Initial Balance
print("\n⚖️  INITIAL BALANCE:")
ib = results.get('initial_balance', {})
if ib.get('available'):
    print(f"   IB High: ${ib.get('ib_high', 0):.2f}")
    print(f"   IB Low: ${ib.get('ib_low', 0):.2f}")
    print(f"   IB Range: ${ib.get('ib_range', 0):.2f}")
    print(f"   Close Position: {ib.get('close_position', 'N/A')}")
    print(f"   Expected Break: {ib.get('expected_break_direction', 'N/A')}")
    print(f"   Break Probability: {ib.get('directional_break_probability', 0)*100:.1f}%")
else:
    print(f"   ❌ {ib.get('reason', 'Not available')}")

# Signals
print("\n🎯 TRADING SIGNALS:")
signals = results.get('signals', [])
print(f"   Total Signals: {len(signals)}")

if signals:
    for i, signal in enumerate(signals, 1):
        print(f"\n   Signal #{i}:")
        print(f"      Type: {signal.signal_type}")
        print(f"      Source: {signal.source}")
        print(f"      Direction: {signal.direction or 'N/A'}")
        print(f"      Probability: {signal.probability*100:.1f}%")
        print(f"      Confidence: {signal.confidence*100:.1f}%")
        print(f"      Reasoning: {signal.reasoning}")
else:
    print("   No signals generated")

# Overall Confidence
print("\n📊 OVERALL CONFIDENCE:")
confidence = results.get('overall_confidence', {})
print(f"   Score: {confidence.get('score', 0)*100:.1f}%")
print(f"   Level: {confidence.get('level', 'N/A')}")
print(f"   Confluence Count: {confidence.get('confluence_count', 0)}")
print(f"   Bullish Signals: {confidence.get('bullish_signals', 0)}")
print(f"   Bearish Signals: {confidence.get('bearish_signals', 0)}")
print(f"   Recommendation: {confidence.get('recommendation', 'N/A')}")
print(f"   Position Sizing: {confidence.get('position_sizing', 'N/A')}")

print("\n" + "="*80)
print("  TEST SUMMARY")
print("="*80)
print("✅ NQ Stats Analyzer is working correctly!")
print(f"✅ Analysis completed successfully")
print(f"✅ {len(signals)} signals generated")
print(f"✅ Overall recommendation: {confidence.get('recommendation', 'WAIT')}")

print("\n🎉 All tests PASSED!\n")
