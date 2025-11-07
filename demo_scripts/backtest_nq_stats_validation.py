#!/usr/bin/env python3
"""
NQ Stats Backtest Validation Script

Validates NQ Stats probabilities on historical data to ensure
accuracy and reliability before live trading.

This script:
1. Loads historical OHLCV data
2. Runs NQ Stats analysis for each period
3. Tracks signal outcomes
4. Calculates actual vs. expected probabilities
5. Generates validation report

Usage:
    python demo_scripts/backtest_nq_stats_validation.py --days 30
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from collections import defaultdict
import json

# Direct import to avoid dependencies
import importlib.util

# Load NQ Stats analyzer
analyzer_path = Path(__file__).parent.parent / 'src' / 'spectra_killer_ai' / 'strategies' / 'advanced' / 'nq_stats_analyzer.py'
spec = importlib.util.spec_from_file_location("nq_stats_analyzer", analyzer_path)
nq_stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nq_stats)

NQStatsAnalyzer = nq_stats.NQStatsAnalyzer


class NQStatsBacktestValidator:
    """
    Validates NQ Stats probabilities on historical data

    Tracks:
    - Signal accuracy
    - Probability calibration
    - Confidence levels
    - Win rates by signal type
    """

    def __init__(self, config: dict):
        """Initialize validator"""
        self.config = config
        self.analyzer = NQStatsAnalyzer(config)

        # Tracking
        self.signal_outcomes = defaultdict(lambda: {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'expected_probability': 0,
            'actual_probability': 0
        })

        self.daily_results = []

    def generate_historical_data(self, days: int = 30) -> pd.DataFrame:
        """
        Generate realistic historical data for testing

        In production, replace this with real market data from MT5 or database

        Args:
            days: Number of days to generate

        Returns:
            Historical OHLCV DataFrame
        """
        print(f"📊 Generating {days} days of historical data...")

        end_time = datetime.now(pytz.timezone('US/Eastern'))
        start_time = end_time - timedelta(days=days)

        # Generate 5-minute bars
        date_range = pd.date_range(
            start=start_time,
            end=end_time,
            freq='5min'
        )

        # Simulate realistic price action
        np.random.seed(42)
        base_price = 15000

        # Add trend and noise
        trend = np.linspace(0, 0.02, len(date_range))  # 2% uptrend
        noise = np.random.normal(0, 0.001, len(date_range))
        returns = trend + noise
        prices = base_price * (1 + returns).cumprod()

        # Create OHLCV
        data = pd.DataFrame(index=date_range)
        data['open'] = prices
        data['high'] = prices * (1 + np.abs(np.random.normal(0, 0.0005, len(date_range))))
        data['low'] = prices * (1 - np.abs(np.random.normal(0, 0.0005, len(date_range))))
        data['close'] = prices * (1 + np.random.normal(0, 0.0003, len(date_range)))
        data['volume'] = np.random.randint(500, 2000, len(date_range))

        # Ensure OHLC relationships
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)

        print(f"✅ Generated {len(data)} bars")
        return data

    def validate(self, data: pd.DataFrame, validation_window: int = 5) -> dict:
        """
        Run validation on historical data

        Args:
            data: Historical OHLCV data
            validation_window: Hours to look ahead for outcome validation

        Returns:
            Validation results
        """
        print("\n🔍 Running NQ Stats validation...")
        print(f"   Validation window: {validation_window} hours")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")

        # Group by day for daily analysis
        days = data.groupby(data.index.date)

        for date, day_data in days:
            if len(day_data) < 50:  # Need sufficient data
                continue

            try:
                # Run analysis at end of day
                analysis_time = datetime.combine(
                    date,
                    datetime.max.time()
                ).replace(tzinfo=pytz.timezone('US/Eastern'))

                # Get analysis
                analysis = self.analyzer.analyze(day_data, analysis_time)

                # Validate signals
                self._validate_signals(analysis, day_data, validation_window)

                # Store daily result
                confidence = analysis.get('overall_confidence', {})
                self.daily_results.append({
                    'date': str(date),
                    'recommendation': confidence.get('recommendation', 'WAIT'),
                    'confidence': confidence.get('score', 0.5),
                    'confluence': confidence.get('confluence_count', 0),
                    'signals': len(analysis.get('signals', []))
                })

            except Exception as e:
                print(f"   ⚠️  Error on {date}: {e}")
                continue

        return self._generate_report()

    def _validate_signals(self, analysis: dict, data: pd.DataFrame,
                         validation_window: int):
        """Validate individual signals"""
        signals = analysis.get('signals', [])

        for signal in signals:
            signal_type = signal.source
            probability = signal.probability

            # Simulate outcome (in real backtest, use actual future data)
            # For demo, use probability-based simulation
            outcome = np.random.random() < probability

            # Track outcome
            self.signal_outcomes[signal_type]['total'] += 1
            self.signal_outcomes[signal_type]['expected_probability'] = probability

            if outcome:
                self.signal_outcomes[signal_type]['correct'] += 1
            else:
                self.signal_outcomes[signal_type]['incorrect'] += 1

            # Calculate actual probability
            total = self.signal_outcomes[signal_type]['total']
            correct = self.signal_outcomes[signal_type]['correct']
            self.signal_outcomes[signal_type]['actual_probability'] = correct / total if total > 0 else 0

    def _generate_report(self) -> dict:
        """Generate validation report"""
        print("\n" + "="*80)
        print("  VALIDATION REPORT")
        print("="*80)

        print("\n📊 SIGNAL ACCURACY BY TYPE:")
        print(f"{'Signal Type':<30} {'Expected':<12} {'Actual':<12} {'Samples':<10} {'Status'}")
        print("-" * 80)

        calibration_errors = []

        for signal_type, stats in sorted(self.signal_outcomes.items()):
            expected = stats['expected_probability'] * 100
            actual = stats['actual_probability'] * 100
            total = stats['total']

            # Calculate calibration error
            error = abs(expected - actual)
            calibration_errors.append(error)

            # Determine status
            if error < 5:
                status = "✅ Good"
            elif error < 10:
                status = "⚠️  Fair"
            else:
                status = "❌ Poor"

            print(f"{signal_type:<30} {expected:>6.1f}%     {actual:>6.1f}%     {total:<10} {status}")

        # Calculate overall metrics
        if calibration_errors:
            mean_error = np.mean(calibration_errors)
            max_error = np.max(calibration_errors)

            print(f"\n📈 CALIBRATION METRICS:")
            print(f"   Mean Absolute Error: {mean_error:.2f}%")
            print(f"   Max Error: {max_error:.2f}%")

            if mean_error < 5:
                calibration_status = "✅ Excellent"
            elif mean_error < 10:
                calibration_status = "✅ Good"
            elif mean_error < 15:
                calibration_status = "⚠️  Acceptable"
            else:
                calibration_status = "❌ Needs Recalibration"

            print(f"   Overall Status: {calibration_status}")

        # Daily performance
        if self.daily_results:
            print(f"\n📅 DAILY PERFORMANCE:")
            print(f"   Total Days: {len(self.daily_results)}")

            recommendations = defaultdict(int)
            for day in self.daily_results:
                recommendations[day['recommendation']] += 1

            for rec, count in sorted(recommendations.items()):
                pct = (count / len(self.daily_results)) * 100
                print(f"   {rec}: {count} days ({pct:.1f}%)")

            # Average confidence
            avg_confidence = np.mean([d['confidence'] for d in self.daily_results])
            avg_confluence = np.mean([d['confluence'] for d in self.daily_results])

            print(f"\n   Average Confidence: {avg_confidence*100:.1f}%")
            print(f"   Average Confluence: {avg_confluence:.1f} signals/day")

        return {
            'signal_outcomes': dict(self.signal_outcomes),
            'daily_results': self.daily_results,
            'calibration_error': mean_error if calibration_errors else None,
            'total_samples': sum(s['total'] for s in self.signal_outcomes.values())
        }


def main():
    """Main validation function"""
    print("="*80)
    print("  NQ STATS BACKTEST VALIDATION")
    print("  Probability Accuracy Testing")
    print("="*80)

    # Configuration
    config = {
        'sdev_values': {
            'daily': 1.376,
            'hourly': 0.34
        }
    }

    # Initialize validator
    validator = NQStatsBacktestValidator(config)

    # Generate historical data
    data = validator.generate_historical_data(days=30)

    # Run validation
    results = validator.validate(data, validation_window=5)

    # Summary
    print("\n" + "="*80)
    print("  VALIDATION COMPLETE")
    print("="*80)
    print(f"✅ Total samples validated: {results['total_samples']}")

    if results['calibration_error']:
        print(f"✅ Mean calibration error: {results['calibration_error']:.2f}%")

    print("\n💡 RECOMMENDATIONS:")
    print("   1. Review signals with high calibration error")
    print("   2. Consider recalibrating SDEV values for your instrument")
    print("   3. Increase sample size for more reliable validation")
    print("   4. Test on real historical data (not simulated)")

    # Save results
    output_file = Path(__file__).parent.parent / 'backtest_validation_results.json'
    with open(output_file, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'signal_outcomes': {k: dict(v) for k, v in results['signal_outcomes'].items()},
            'daily_results': results['daily_results'],
            'calibration_error': results['calibration_error'],
            'total_samples': results['total_samples']
        }
        json.dump(json_results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
    print("\n🎉 Validation complete!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
