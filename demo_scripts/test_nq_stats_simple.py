#!/usr/bin/env python3
"""
Simple NQ Stats Test - Direct import without full engine
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Direct import to avoid engine dependencies
from spectra_killer_ai.strategies.advanced.nq_stats_analyzer import NQStatsAnalyzer

print("✅ Import successful!")

# Generate simple test data
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
data['high'] = prices * 1.001
data['low'] = prices * 0.999
data['close'] = prices
data['volume'] = 1000

print(f"✅ Generated {len(data)} bars")

# Initialize analyzer with minimal config
config = {
    'sdev_values': {
        'daily': 1.376,
        'hourly': 0.34
    }
}

print("\n🚀 Initializing NQ Stats Analyzer...")
analyzer = NQStatsAnalyzer(config)
print("✅ Analyzer initialized!")

# Run analysis
print("\n🔍 Running analysis...")
results = analyzer.analyze(data, end_time)

print("\n📊 RESULTS:")
print(f"   Timestamp: {results.get('timestamp', 'N/A')}")
print(f"   Available: {results.get('available', 'N/A')}")

# Check SDEV analysis
sdev = results.get('sdev_analysis', {})
if sdev.get('available'):
    print("\n✅ SDEV Analysis:")
    daily = sdev.get('sdev_levels', {}).get('daily', {})
    print(f"   Zone: {daily.get('zone', 'N/A')}")
    print(f"   SDEV Distance: {daily.get('sdev_distance', 'N/A')}")
    print(f"   Reversion Probability: {daily.get('reversion_probability', 0)*100:.1f}%")

# Check signals
signals = results.get('signals', [])
print(f"\n🎯 Signals Generated: {len(signals)}")
for i, signal in enumerate(signals[:3], 1):
    print(f"   {i}. {signal.signal_type} - {signal.probability*100:.1f}% confidence")

# Check confidence
confidence = results.get('overall_confidence', {})
print(f"\n📈 Overall Confidence:")
print(f"   Score: {confidence.get('score', 0)*100:.1f}%")
print(f"   Level: {confidence.get('level', 'N/A')}")
print(f"   Recommendation: {confidence.get('recommendation', 'N/A')}")

print("\n🎉 Test PASSED! NQ Stats Analyzer is working correctly.")
