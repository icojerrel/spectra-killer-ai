"""
NQ Stats Integration Module for Spectra Trading Engine

Provides seamless integration of NQ Stats probability-based analysis
into the existing trading engine without modifying core code.
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime
import pandas as pd

from .nq_stats_analyzer import NQStatsAnalyzer, NQStatsSignal

logger = logging.getLogger(__name__)


class NQStatsIntegration:
    """
    Integration wrapper for NQ Stats in Spectra Trading Engine

    Usage:
        # In your trading engine or strategy
        nq_stats_integration = NQStatsIntegration(config)

        # Analyze market data
        analysis = nq_stats_integration.analyze(data)

        # Get trading recommendations
        recommendation = nq_stats_integration.get_trading_recommendation(analysis)

        # Apply to risk management
        adjusted_position_size = nq_stats_integration.adjust_position_size(
            base_size, analysis
        )
    """

    def __init__(self, config: Dict):
        """
        Initialize NQ Stats integration

        Args:
            config: Configuration dictionary with NQ Stats settings
        """
        self.config = config
        self.analyzer = NQStatsAnalyzer(config.get('nq_stats', {}))
        self.enabled = config.get('nq_stats', {}).get('enabled', True)

        # Integration settings
        self.use_for_filtering = config.get('nq_stats', {}).get('use_for_filtering', True)
        self.use_for_sizing = config.get('nq_stats', {}).get('use_for_sizing', True)
        self.use_for_tp_sl = config.get('nq_stats', {}).get('use_for_tp_sl', True)
        self.min_confluence_for_signal = config.get('nq_stats', {}).get('min_confluence', 2)

        logger.info(f"NQ Stats Integration initialized (enabled={self.enabled})")

    def analyze(self, data: pd.DataFrame, current_time: Optional[datetime] = None) -> Dict:
        """
        Run NQ Stats analysis on market data

        Args:
            data: OHLCV DataFrame
            current_time: Current time (optional)

        Returns:
            Complete NQ Stats analysis results
        """
        if not self.enabled:
            return {'enabled': False}

        try:
            return self.analyzer.analyze(data, current_time)
        except Exception as e:
            logger.error(f"Error in NQ Stats analysis: {e}", exc_info=True)
            return {'error': str(e), 'enabled': True, 'available': False}

    def get_trading_recommendation(self, analysis: Dict) -> Dict:
        """
        Extract trading recommendation from NQ Stats analysis

        Args:
            analysis: NQ Stats analysis results

        Returns:
            Trading recommendation with direction, confidence, and reasoning
        """
        if not analysis.get('enabled', False):
            return {'available': False, 'reason': 'NQ Stats disabled'}

        confidence = analysis.get('overall_confidence', {})

        return {
            'available': True,
            'direction': confidence.get('recommendation', 'WAIT'),
            'confidence': confidence.get('score', 0.5),
            'level': confidence.get('level', 'NEUTRAL'),
            'confluence_count': confidence.get('confluence_count', 0),
            'signals': analysis.get('signals', []),
            'reasoning': self._generate_reasoning(analysis)
        }

    def should_trade(self, analysis: Dict, base_signal: Optional[str] = None) -> bool:
        """
        Determine if trading should proceed based on NQ Stats

        Args:
            analysis: NQ Stats analysis results
            base_signal: Base signal from other strategy (BUY/SELL/WAIT)

        Returns:
            True if trading is recommended, False otherwise
        """
        if not self.use_for_filtering or not analysis.get('enabled', False):
            return True  # Don't filter if disabled

        confidence = analysis.get('overall_confidence', {})
        recommendation = confidence.get('recommendation', 'WAIT')
        confluence = confidence.get('confluence_count', 0)

        # If no base signal, use NQ Stats recommendation
        if not base_signal:
            return recommendation != 'WAIT' and confluence >= self.min_confluence_for_signal

        # If base signal provided, check alignment
        if base_signal == 'BUY' or base_signal == 'LONG':
            # Check if NQ Stats agrees (bullish)
            return recommendation in ['LONG', 'LONG_SMALL'] and confluence >= self.min_confluence_for_signal
        elif base_signal == 'SELL' or base_signal == 'SHORT':
            # Check if NQ Stats agrees (bearish)
            return recommendation in ['SHORT', 'SHORT_SMALL'] and confluence >= self.min_confluence_for_signal

        return recommendation != 'WAIT'

    def adjust_position_size(self, base_position_size: float, analysis: Dict) -> float:
        """
        Adjust position size based on NQ Stats confluence

        Args:
            base_position_size: Base position size
            analysis: NQ Stats analysis results

        Returns:
            Adjusted position size
        """
        if not self.use_for_sizing or not analysis.get('enabled', False):
            return base_position_size

        confidence = analysis.get('overall_confidence', {})
        confluence = confidence.get('confluence_count', 0)
        position_sizing = confidence.get('position_sizing', 'REDUCED')

        # Apply position sizing multiplier
        if position_sizing == 'FULL' and confluence >= 4:
            multiplier = 2.0  # 4+ signals
        elif confluence >= 3:
            multiplier = 1.5  # 3 signals
        elif confluence >= 2:
            multiplier = 1.25  # 2 signals
        elif confluence >= 1:
            multiplier = 1.0  # 1 signal
        else:
            multiplier = 0.5  # No signals - reduce size

        adjusted_size = base_position_size * multiplier

        logger.debug(f"Position size adjusted: {base_position_size} -> {adjusted_size} "
                    f"(confluence={confluence}, multiplier={multiplier})")

        return adjusted_size

    def get_dynamic_stop_loss(self, analysis: Dict, entry_price: float,
                             direction: str) -> Optional[float]:
        """
        Calculate dynamic stop loss using SDEV levels

        Args:
            analysis: NQ Stats analysis results
            entry_price: Entry price
            direction: Trade direction ('LONG' or 'SHORT')

        Returns:
            Stop loss price or None if not available
        """
        if not self.use_for_tp_sl or not analysis.get('enabled', False):
            return None

        sdev_data = analysis.get('sdev_analysis', {})
        if not sdev_data.get('available'):
            return None

        levels = sdev_data.get('sdev_levels', {}).get('daily', {})
        if not levels:
            return None

        # Use -1.5 SDEV for longs, +1.5 SDEV for shorts
        if direction.upper() in ['LONG', 'BUY']:
            stop_loss = levels.get('levels', {}).get('-1.5σ')
        else:  # SHORT/SELL
            stop_loss = levels.get('levels', {}).get('+1.5σ')

        if stop_loss:
            logger.debug(f"Dynamic stop loss: ${stop_loss:.2f} (direction={direction})")

        return stop_loss

    def get_dynamic_take_profit(self, analysis: Dict, entry_price: float,
                                direction: str) -> Optional[float]:
        """
        Calculate dynamic take profit using SDEV levels

        Args:
            analysis: NQ Stats analysis results
            entry_price: Entry price
            direction: Trade direction ('LONG' or 'SHORT')

        Returns:
            Take profit price or None if not available
        """
        if not self.use_for_tp_sl or not analysis.get('enabled', False):
            return None

        sdev_data = analysis.get('sdev_analysis', {})
        if not sdev_data.get('available'):
            return None

        levels = sdev_data.get('sdev_levels', {}).get('daily', {})
        if not levels:
            return None

        # Use +1.0 SDEV for longs, -1.0 SDEV for shorts
        if direction.upper() in ['LONG', 'BUY']:
            take_profit = levels.get('levels', {}).get('+1.0σ')
        else:  # SHORT/SELL
            take_profit = levels.get('levels', {}).get('-1.0σ')

        if take_profit:
            logger.debug(f"Dynamic take profit: ${take_profit:.2f} (direction={direction})")

        return take_profit

    def get_hour_segment_filter(self, analysis: Dict) -> Dict:
        """
        Get hour segment-based trading filter

        Args:
            analysis: NQ Stats analysis results

        Returns:
            Hour segment filter information
        """
        hour_stats = analysis.get('hour_stats', {})

        if not hour_stats.get('available'):
            return {'should_filter': False, 'reason': 'Hour stats not available'}

        segment = hour_stats.get('segment', 2)
        segment_type = hour_stats.get('segment_type', 'expansion')
        is_9am_hour = hour_stats.get('is_9am_hour', False)

        # Segment 1 (first 20 min): High retracement probability - favor mean reversion
        # Segment 2 (middle 20 min): Expansion - favor momentum
        # Segment 3 (last 20 min): Wick formation - be cautious

        if is_9am_hour:
            return {
                'should_filter': True,
                'reason': 'High volatility during 9am hour pre-open',
                'recommended_action': 'WAIT'
            }

        return {
            'should_filter': False,
            'segment': segment,
            'segment_type': segment_type,
            'strategy_preference': 'mean_reversion' if segment == 1 else 'momentum' if segment == 2 else 'cautious',
            'retracement_probability': hour_stats.get('retracement_probability', 0.5)
        }

    def get_session_quality(self, analysis: Dict) -> Dict:
        """
        Get session quality metrics

        Args:
            analysis: NQ Stats analysis results

        Returns:
            Session quality information
        """
        session_pattern = analysis.get('session_pattern', {})

        if not session_pattern.get('available'):
            return {'quality': 'UNKNOWN', 'probability': 0.5}

        probability = session_pattern.get('probability', 0.5)
        pattern_type = session_pattern.get('pattern_type', 'no_clear_pattern')

        if probability >= 0.90:
            quality = 'EXCELLENT'
        elif probability >= 0.75:
            quality = 'HIGH'
        elif probability >= 0.60:
            quality = 'MEDIUM'
        else:
            quality = 'LOW'

        return {
            'quality': quality,
            'probability': probability,
            'pattern_type': pattern_type,
            'expected_behavior': session_pattern.get('expected_behavior', 'Unknown')
        }

    def _generate_reasoning(self, analysis: Dict) -> str:
        """Generate human-readable reasoning from analysis"""
        signals = analysis.get('signals', [])
        confidence = analysis.get('overall_confidence', {})

        if not signals:
            return "No high-probability setups detected"

        reasoning_parts = []

        # Add confluence count
        confluence = confidence.get('confluence_count', 0)
        reasoning_parts.append(f"{confluence} confirming signals")

        # Add key signals
        for signal in signals[:3]:  # Top 3 signals
            reasoning_parts.append(
                f"{signal.source}: {signal.probability*100:.0f}% probability"
            )

        return " | ".join(reasoning_parts)

    def get_summary(self, analysis: Dict) -> str:
        """
        Get human-readable summary of NQ Stats analysis

        Args:
            analysis: NQ Stats analysis results

        Returns:
            Summary string
        """
        if not analysis.get('enabled', False):
            return "NQ Stats: Disabled"

        confidence = analysis.get('overall_confidence', {})
        recommendation = confidence.get('recommendation', 'WAIT')
        score = confidence.get('score', 0.5) * 100
        confluence = confidence.get('confluence_count', 0)

        return (f"NQ Stats: {recommendation} ({score:.0f}% confidence, "
                f"{confluence} signals)")


# Helper function for easy integration
def create_nq_stats_integration(config: Dict) -> NQStatsIntegration:
    """
    Factory function to create NQ Stats integration

    Args:
        config: Configuration dictionary

    Returns:
        NQStatsIntegration instance
    """
    return NQStatsIntegration(config)
