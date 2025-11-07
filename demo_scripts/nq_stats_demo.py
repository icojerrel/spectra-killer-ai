#!/usr/bin/env python3
"""
NQ Stats Analyzer Demo Script

Demonstrates how to use the NQStatsAnalyzer for probability-based trading analysis.
Based on 10-20 years of NQ historical data (2004-2025).

Usage:
    python demo_scripts/nq_stats_demo.py
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import yaml

from spectra_killer_ai.strategies.advanced import NQStatsAnalyzer


def load_config():
    """Load NQ Stats configuration"""
    config_path = Path(__file__).parent.parent / 'config' / 'nq_stats_config.yaml'

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config
    else:
        # Default config
        return {
            'sdev_values': {
                'daily': 1.376,
                'hourly': 0.34
            },
            'high_probability_threshold': 0.75,
            'medium_probability_threshold': 0.51
        }


def generate_sample_data(days=5, freq='5min'):
    """
    Generate sample OHLCV data for demonstration

    In production, this would be replaced with real market data from MT5 or another source.
    """
    print("📊 Generating sample market data...")

    # Generate date range
    end_time = datetime.now(pytz.timezone('US/Eastern'))
    start_time = end_time - timedelta(days=days)

    # Create datetime index
    date_range = pd.date_range(start=start_time, end=end_time, freq=freq)

    # Generate realistic price data (simulating NQ futures around 15000)
    np.random.seed(42)
    base_price = 15000

    # Generate returns with realistic volatility
    returns = np.random.normal(0, 0.001, len(date_range))
    prices = base_price * (1 + returns).cumprod()

    # Create OHLCV data
    data = pd.DataFrame(index=date_range)
    data['open'] = prices
    data['high'] = prices * (1 + np.abs(np.random.normal(0, 0.0005, len(date_range))))
    data['low'] = prices * (1 - np.abs(np.random.normal(0, 0.0005, len(date_range))))
    data['close'] = prices * (1 + np.random.normal(0, 0.0003, len(date_range)))
    data['volume'] = np.random.randint(100, 1000, len(date_range))

    # Ensure high is highest and low is lowest
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    print(f"✅ Generated {len(data)} bars of sample data")
    print(f"📅 Date range: {date_range[0]} to {date_range[-1]}")
    print(f"💰 Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
    print()

    return data


def print_analysis_section(title: str, data: dict):
    """Pretty print an analysis section"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    if not data or not data.get('available', True):
        print("❌ Not available")
        return

    for key, value in data.items():
        if key in ['available', 'timestamp', 'current_time_est']:
            continue

        if isinstance(value, dict):
            print(f"\n📋 {key.replace('_', ' ').title()}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    if 'probability' in sub_key.lower():
                        print(f"   {sub_key}: {sub_value*100:.1f}%")
                    else:
                        print(f"   {sub_key}: {sub_value:.4f}")
                else:
                    print(f"   {sub_key}: {sub_value}")
        elif isinstance(value, list):
            print(f"\n📋 {key.replace('_', ' ').title()}:")
            for item in value:
                print(f"   - {item}")
        elif isinstance(value, float):
            if 'probability' in key.lower() or 'confidence' in key.lower():
                print(f"📊 {key}: {value*100:.1f}%")
            else:
                print(f"📊 {key}: {value:.4f}")
        else:
            print(f"📊 {key}: {value}")


def print_signals(signals):
    """Pretty print trading signals"""
    print(f"\n{'='*80}")
    print(f"  🎯 TRADING SIGNALS ({len(signals)} signals)")
    print(f"{'='*80}")

    if not signals:
        print("No signals generated")
        return

    for i, signal in enumerate(signals, 1):
        print(f"\n🔔 Signal #{i}")
        print(f"   Type: {signal.signal_type}")
        print(f"   Source: {signal.source}")
        print(f"   Direction: {signal.direction or 'N/A'}")
        print(f"   Probability: {signal.probability*100:.1f}%")
        print(f"   Confidence: {signal.confidence*100:.1f}%")
        print(f"   Reasoning: {signal.reasoning}")

        if signal.supporting_stats:
            print(f"   Supporting Stats:")
            for key, value in signal.supporting_stats.items():
                print(f"      - {key}: {value}")


def main():
    """Main demo function"""
    print("\n" + "="*80)
    print("  NQ STATS ANALYZER - DEMO")
    print("  Probability-Based Trading Analysis")
    print("  Based on 10-20 years of NQ historical data (2004-2025)")
    print("="*80 + "\n")

    # Load configuration
    print("⚙️  Loading NQ Stats configuration...")
    config = load_config()
    print("✅ Configuration loaded")
    print(f"   Daily SDEV: {config.get('sdev_values', {}).get('daily', 1.376)}%")
    print(f"   Hourly SDEV: {config.get('sdev_values', {}).get('hourly', 0.34)}%")
    print()

    # Generate sample data
    data = generate_sample_data(days=5, freq='5min')

    # Initialize analyzer
    print("🚀 Initializing NQ Stats Analyzer...")
    analyzer = NQStatsAnalyzer(config)
    print("✅ Analyzer initialized\n")

    # Perform analysis
    current_time = datetime.now(pytz.timezone('US/Eastern'))
    print(f"🕐 Analysis Time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()

    print("🔍 Running comprehensive NQ Stats analysis...")
    results = analyzer.analyze(data, current_time)
    print("✅ Analysis complete!\n")

    # Display results
    print_analysis_section("📈 STANDARD DEVIATION ANALYSIS", results.get('sdev_analysis', {}))
    print_analysis_section("⏰ HOUR STATS", results.get('hour_stats', {}))
    print_analysis_section("📊 RTH BREAKS", results.get('rth_breaks', {}))
    print_analysis_section("🕐 9AM HOUR CONTINUATION", results.get('9am_continuation', {}))
    print_analysis_section("🌅 MORNING JUDAS", results.get('morning_judas', {}))
    print_analysis_section("⚖️  INITIAL BALANCE", results.get('initial_balance', {}))
    print_analysis_section("🌓 NOON CURVE", results.get('noon_curve', {}))
    print_analysis_section("🌍 SESSION PATTERN (ALN)", results.get('session_pattern', {}))

    # Display signals
    signals = results.get('signals', [])
    print_signals(signals)

    # Display overall confidence
    confidence = results.get('overall_confidence', {})
    print(f"\n{'='*80}")
    print(f"  📊 OVERALL TRADING CONFIDENCE")
    print(f"{'='*80}")
    print(f"   Score: {confidence.get('score', 0)*100:.1f}%")
    print(f"   Level: {confidence.get('level', 'UNKNOWN')}")
    print(f"   Confluence Count: {confidence.get('confluence_count', 0)} signals")
    print(f"   Bullish Signals: {confidence.get('bullish_signals', 0)}")
    print(f"   Bearish Signals: {confidence.get('bearish_signals', 0)}")
    print(f"   Recommendation: {confidence.get('recommendation', 'WAIT')}")
    print(f"   Position Sizing: {confidence.get('position_sizing', 'UNKNOWN')}")

    # Summary
    print(f"\n{'='*80}")
    print(f"  ✨ SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Analysis completed successfully")
    print(f"📊 {len(signals)} trading signals generated")
    print(f"🎯 Overall confidence: {confidence.get('level', 'NEUTRAL')}")
    print(f"💡 Recommendation: {confidence.get('recommendation', 'WAIT')}")

    print(f"\n{'='*80}")
    print(f"  📚 NEXT STEPS")
    print(f"{'='*80}")
    print("1. Integrate NQ Stats analyzer into your trading strategy")
    print("2. Use signals for entry/exit decisions")
    print("3. Combine with existing technical analysis")
    print("4. Backtest on historical data to validate probabilities")
    print("5. Adjust position sizing based on confluence score")
    print("6. Review docs/NQ_STATS_COMPLETE_GUIDE.md for detailed methodology")
    print()

    print("🎉 Demo complete! Happy trading!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
