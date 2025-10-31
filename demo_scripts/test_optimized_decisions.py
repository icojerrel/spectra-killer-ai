#!/usr/bin/env python3
# Spectra Killer AI - Optimized Decision Engine Test
# Test de geoptimaliseerde decision engine zonder MT5 dependency

import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any

from decision_optimizer import DecisionOptimizer
from user_config import PAPER_TRADING_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedDecisionTester:
    """
    Test de geoptimaliseerde decision engine met realistische data.
    """

    def __init__(self):
        """Initialize decision tester."""
        self.config = PAPER_TRADING_CONFIG
        self.optimizer = DecisionOptimizer(self.config)

        logger.info("Optimized Decision Tester initialized")

    def create_realistic_xau_data(self, scenario: str = "normal") -> pd.DataFrame:
        """
        Create realistische XAUUSD data voor testing.

        Args:
            scenario (str): Market scenario type

        Returns:
            pd.DataFrame: OHLCV data met technical indicators
        """
        # Base parameters for XAUUSD
        base_price = 2350.0  # Realistische gold price
        volatility = 0.008    # ~0.8% daily volatility for gold
        trend_strength = 0.0002  # Small trend component

        # Create time series
        dates = pd.date_range(start='2025-10-01', periods=100, freq='h')

        # Generate price series based on scenario
        if scenario == "bullish":
            trend = np.linspace(0, 0.05, 100)  # 5% uptrend
        elif scenario == "bearish":
            trend = np.linspace(0, -0.05, 100)  # 5% downtrend
        elif scenario == "sideways":
            trend = np.random.normal(0, 0.001, 100)  # Random walk
        elif scenario == "volatile":
            volatility = 0.015  # Higher volatility
            trend = np.random.normal(0, 0.005, 100)
        else:  # normal
            trend = np.random.normal(0, 0.002, 100)

        # Generate returns
        returns = np.random.normal(trend, volatility/np.sqrt(24), 100)  # Hourly volatility

        # Calculate prices
        prices = [base_price]
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)

        prices = np.array(prices)

        # Generate OHLC
        high = prices * (1 + np.abs(np.random.normal(0, 0.002, 100)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.002, 100)))
        open_price = np.roll(prices, 1)
        open_price[0] = prices[0]

        # Volume (realistic for gold)
        base_volume = 5000
        volume_spikes = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
        volume = base_volume * (1 + volume_spikes * np.random.uniform(0.5, 2.0, 100))
        tick_volume = volume / 10

        # Create DataFrame
        data = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume.astype(int),
            'tick_volume': tick_volume.astype(int)
        }, index=dates)

        # Add technical indicators
        data = self._add_indicators(data)

        return data

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe."""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Moving Averages
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()

        return df

    def test_scenarios(self) -> Dict[str, Any]:
        """
        Test verschillende market scenarios.

        Returns:
            Dict[str, Any]: Test results
        """
        scenarios = [
            ("Normal Market", "normal"),
            ("Bullish Trend", "bullish"),
            ("Bearish Trend", "bearish"),
            ("Sideways Market", "sideways"),
            ("Volatile Market", "volatile"),
            ("Oversold Condition", "oversold"),
            ("Overbought Condition", "overbought")
        ]

        results = {
            'test_timestamp': datetime.now().isoformat(),
            'scenarios': {},
            'summary': {
                'total_decisions': 0,
                'buy_decisions': 0,
                'sell_decisions': 0,
                'hold_decisions': 0,
                'avg_confidence': 0.0,
                'actionability_rate': 0.0
            }
        }

        logger.info("Testing optimized decision engine across scenarios...")

        for scenario_name, scenario_type in scenarios:
            logger.info(f"\nTesting scenario: {scenario_name}")

            # Create scenario data
            if scenario_type == "oversold":
                data = self.create_realistic_xau_data("normal")
                # Force oversold RSI
                data['rsi'] = data['rsi'] - 15  # Shift RSI down
                data.loc[data['rsi'] < 0, 'rsi'] = data.loc[data['rsi'] < 0, 'rsi'] + 30
            elif scenario_type == "overbought":
                data = self.create_realistic_xau_data("normal")
                # Force overbought RSI
                data['rsi'] = data['rsi'] + 15  # Shift RSI up
                data.loc[data['rsi'] > 100, 'rsi'] = data.loc[data['rsi'] > 100, 'rsi'] - 30
            else:
                data = self.create_realistic_xau_data(scenario_type)

            # Test decisions over time
            scenario_results = {
                'decisions': [],
                'decision_stats': {},
                'final_state': {}
            }

            # Test last 30 data points
            test_data = data.tail(30)

            for i in range(10, len(test_data)):  # Skip first 10 for indicator stability
                current_data = data.iloc[:i+50]  # Use 50 candles for decision
                if len(current_data) < 50:
                    continue

                decision = self.optimizer.analyze_and_decide(current_data, debug=False)
                scenario_results['decisions'].append(decision)

            # Calculate scenario statistics
            if scenario_results['decisions']:
                decision_types = [d['decision'] for d in scenario_results['decisions']]
                confidences = [d['confidence'] for d in scenario_results['decisions']]
                should_trade = [d['should_trade'] for d in scenario_results['decisions']]

                scenario_results['decision_stats'] = {
                    'total_decisions': len(decision_types),
                    'buy_count': decision_types.count('BUY'),
                    'sell_count': decision_types.count('SELL'),
                    'hold_count': decision_types.count('HOLD'),
                    'buy_pct': decision_types.count('BUY') / len(decision_types),
                    'sell_pct': decision_types.count('SELL') / len(decision_types),
                    'hold_pct': decision_types.count('HOLD') / len(decision_types),
                    'avg_confidence': np.mean(confidences),
                    'min_confidence': np.min(confidences),
                    'max_confidence': np.max(confidences),
                    'actionable_trades': sum(should_trade),
                    'actionability_rate': sum(should_trade) / len(should_trade)
                }

                # Get final market state
                latest = test_data.iloc[-1]
                scenario_results['final_state'] = {
                    'price': latest['close'],
                    'rsi': latest['rsi'],
                    'ema_9': latest['ema_9'],
                    'ema_21': latest['ema_21'],
                    'bb_position': (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']),
                    'atr': latest['atr'],
                    'volume': latest['tick_volume']
                }

                # Log results
                stats = scenario_results['decision_stats']
                logger.info(f"  Decisions: BUY={stats['buy_count']}, SELL={stats['sell_count']}, HOLD={stats['hold_count']}")
                logger.info(f"  Actionability: {stats['actionability_rate']:.1%} | Avg Confidence: {stats['avg_confidence']:.2f}")

                # Update summary
                results['summary']['total_decisions'] += stats['total_decisions']
                results['summary']['buy_decisions'] += stats['buy_count']
                results['summary']['sell_decisions'] += stats['sell_count']
                results['summary']['hold_decisions'] += stats['hold_count']

            results['scenarios'][scenario_name] = scenario_results

        # Calculate final summary
        total = results['summary']['total_decisions']
        if total > 0:
            results['summary']['buy_pct'] = results['summary']['buy_decisions'] / total
            results['summary']['sell_pct'] = results['summary']['sell_decisions'] / total
            results['summary']['hold_pct'] = results['summary']['hold_decisions'] / total
            results['summary']['actionability_rate'] = (results['summary']['buy_decisions'] + results['summary']['sell_decisions']) / total

        return results

    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("SPECTRA KILLER AI - OPTIMIZED DECISION ENGINE TEST RESULTS")
        report.append("=" * 80)
        report.append(f"Test Date: {results['test_timestamp']}")
        report.append("")

        # Summary
        summary = results['summary']
        report.append("[GRAFIEK] OVERALL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Total Decisions: {summary['total_decisions']}")
        report.append(f"BUY Decisions: {summary['buy_decisions']} ({summary.get('buy_pct', 0):.1%})")
        report.append(f"SELL Decisions: {summary['sell_decisions']} ({summary.get('sell_pct', 0):.1%})")
        report.append(f"HOLD Decisions: {summary['hold_decisions']} ({summary.get('hold_pct', 0):.1%})")
        report.append(f"Actionability Rate: {summary.get('actionability_rate', 0):.1%}")
        report.append("")

        # Scenario results
        report.append("[DETAILS] SCENARIO RESULTS")
        report.append("-" * 40)

        for scenario_name, scenario_data in results['scenarios'].items():
            if 'decision_stats' in scenario_data:
                stats = scenario_data['decision_stats']
                final_state = scenario_data.get('final_state', {})

                report.append(f"\n{scenario_name}:")
                report.append(f"  Decisions: {stats['total_decisions']}")
                report.append(f"  BUY: {stats['buy_count']} ({stats['buy_pct']:.1%})")
                report.append(f"  SELL: {stats['sell_count']} ({stats['sell_pct']:.1%})")
                report.append(f"  HOLD: {stats['hold_count']} ({stats['hold_pct']:.1%})")
                report.append(f"  Confidence: {stats['avg_confidence']:.2f} (range: {stats['min_confidence']:.2f}-{stats['max_confidence']:.2f})")
                report.append(f"  Actionability: {stats['actionability_rate']:.1%}")

                if final_state:
                    report.append(f"  Final Price: ${final_state['price']:.2f}")
                    report.append(f"  Final RSI: {final_state['rsi']:.1f}")

        # Analysis
        report.append("\n[ANALYSE] DECISION ENGINE PERFORMANCE")
        report.append("-" * 40)

        hold_rate = summary.get('hold_pct', 0)
        actionable_rate = summary.get('actionability_rate', 0)

        if hold_rate < 0.2:
            report.append("[OK] EXCELLENT: Low HOLD rate indicates good decision activity")
        elif hold_rate < 0.5:
            report.append("[OK] GOOD: Moderate HOLD rate")
        else:
            report.append("[WAARSCHUWING] High HOLD rate detected")

        if actionable_rate > 0.7:
            report.append("[OK] EXCELLENT: High actionability rate")
        elif actionable_rate > 0.5:
            report.append("[OK] GOOD: Reasonable actionability rate")
        else:
            report.append("[WAARSCHUWING] Low actionability rate")

        # Bias check
        buy_pct = summary.get('buy_pct', 0)
        sell_pct = summary.get('sell_pct', 0)
        balance_diff = abs(buy_pct - sell_pct)

        if balance_diff < 0.1:
            report.append("[OK] EXCELLENT: Well-balanced BUY/SELL decisions")
        elif balance_diff < 0.2:
            report.append("[OK] GOOD: Reasonably balanced decisions")
        else:
            report.append(f"[WAARSCHUWING] Unbalanced decisions - BUY: {buy_pct:.1%}, SELL: {sell_pct:.1%}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

if __name__ == "__main__":
    print("Spectra Killer AI - Optimized Decision Engine Test")
    print("=" * 60)
    print("Testing the FIXED decision engine (HOLD bias eliminated)")
    print("=" * 60)

    tester = OptimizedDecisionTester()
    results = tester.test_scenarios()

    # Generate and display report
    report = tester.generate_test_report(results)
    print("\n" + report)

    # Save results
    import json
    with open('optimized_decision_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    with open('optimized_decision_test_report.txt', 'w') as f:
        f.write(report)

    print(f"\n[MAP] Results saved to:")
    print(f"  - optimized_decision_test_results.json")
    print(f"  - optimized_decision_test_report.txt")