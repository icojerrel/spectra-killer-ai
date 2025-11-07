"""
NQ Stats Hybrid Trading Strategy

Combines technical analysis with NQ Stats probability-based analysis
for enhanced signal quality and risk management.
"""

import logging
from typing import Dict, Optional, Tuple
import pandas as pd
from datetime import datetime

from ..technical.technical_analyzer import TechnicalAnalyzer
from ..advanced.nq_stats_integration import NQStatsIntegration

logger = logging.getLogger(__name__)


class NQStatsHybridStrategy:
    """
    Hybrid trading strategy combining:
    - Technical Analysis (RSI, MACD, Moving Averages, etc.)
    - NQ Stats Probability Analysis (SDEV, Sessions, IB, etc.)

    Signal Generation Logic:
    1. Generate base technical signal
    2. Validate with NQ Stats probabilities
    3. Adjust position size based on confluence
    4. Set dynamic TP/SL using SDEV levels
    """

    def __init__(self, config: Dict):
        """
        Initialize hybrid strategy

        Args:
            config: Strategy configuration
        """
        self.config = config

        # Initialize components
        self.technical_analyzer = TechnicalAnalyzer(
            config.get('technical', {})
        )
        self.nq_stats = NQStatsIntegration(config)

        # Strategy settings
        self.min_technical_confidence = config.get(
            'min_technical_confidence', 0.6
        )
        self.require_nq_stats_agreement = config.get(
            'require_nq_stats_agreement', True
        )
        self.min_confluence_for_trade = config.get(
            'min_confluence', 2
        )

        logger.info("NQ Stats Hybrid Strategy initialized")

    async def analyze(self, data: pd.DataFrame,
                     current_time: Optional[datetime] = None) -> Dict:
        """
        Perform comprehensive hybrid analysis

        Args:
            data: OHLCV DataFrame
            current_time: Current time (optional)

        Returns:
            Complete analysis with signals and recommendations
        """
        try:
            # 1. Technical Analysis
            technical_analysis = self.technical_analyzer.analyze(data)

            # 2. NQ Stats Analysis
            nq_stats_analysis = self.nq_stats.analyze(data, current_time)

            # 3. Combine analyses
            combined_signal = self._combine_signals(
                technical_analysis,
                nq_stats_analysis
            )

            return {
                'timestamp': datetime.now().isoformat(),
                'technical_analysis': technical_analysis,
                'nq_stats_analysis': nq_stats_analysis,
                'combined_signal': combined_signal,
                'metadata': {
                    'strategy': 'NQStatsHybridStrategy',
                    'version': '1.0'
                }
            }

        except Exception as e:
            logger.error(f"Error in hybrid analysis: {e}", exc_info=True)
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _combine_signals(self, technical: Dict, nq_stats: Dict) -> Dict:
        """
        Combine technical and NQ Stats signals

        Args:
            technical: Technical analysis results
            nq_stats: NQ Stats analysis results

        Returns:
            Combined signal with direction, confidence, and parameters
        """
        # Extract technical signal
        tech_signal = technical.get('signal', 'WAIT')
        tech_confidence = technical.get('confidence', 0.5)

        # Extract NQ Stats recommendation
        nq_recommendation = self.nq_stats.get_trading_recommendation(nq_stats)
        nq_direction = nq_recommendation.get('direction', 'WAIT')
        nq_confidence = nq_recommendation.get('confidence', 0.5)
        nq_confluence = nq_recommendation.get('confluence_count', 0)

        # Check if signals agree
        signals_agree = self._check_agreement(tech_signal, nq_direction)

        # Hour segment filter
        hour_filter = self.nq_stats.get_hour_segment_filter(nq_stats)
        if hour_filter.get('should_filter', False):
            return {
                'signal': 'WAIT',
                'confidence': 0.0,
                'reason': hour_filter.get('reason', 'Hour segment filter'),
                'technical_signal': tech_signal,
                'nq_stats_direction': nq_direction,
                'agreement': False
            }

        # If require agreement and they don't agree, no trade
        if self.require_nq_stats_agreement and not signals_agree:
            return {
                'signal': 'WAIT',
                'confidence': 0.0,
                'reason': 'Technical and NQ Stats signals do not agree',
                'technical_signal': tech_signal,
                'nq_stats_direction': nq_direction,
                'agreement': False
            }

        # If confluence too low, wait
        if nq_confluence < self.min_confluence_for_trade:
            return {
                'signal': 'WAIT',
                'confidence': 0.0,
                'reason': f'Insufficient confluence ({nq_confluence} < {self.min_confluence_for_trade})',
                'technical_signal': tech_signal,
                'nq_stats_direction': nq_direction,
                'confluence': nq_confluence
            }

        # Signals agree and confluence sufficient - generate combined signal
        # Combined confidence = weighted average
        combined_confidence = (tech_confidence * 0.4) + (nq_confidence * 0.6)

        # Determine final signal
        if tech_signal in ['BUY', 'LONG'] and signals_agree:
            final_signal = 'LONG'
        elif tech_signal in ['SELL', 'SHORT'] and signals_agree:
            final_signal = 'SHORT'
        else:
            final_signal = 'WAIT'

        # Get position sizing adjustment
        base_position_size = 1.0
        adjusted_position_size = self.nq_stats.adjust_position_size(
            base_position_size, nq_stats
        )

        # Get dynamic TP/SL
        current_price = float(data['close'].iloc[-1])
        stop_loss = self.nq_stats.get_dynamic_stop_loss(
            nq_stats, current_price, final_signal
        )
        take_profit = self.nq_stats.get_dynamic_take_profit(
            nq_stats, current_price, final_signal
        )

        # Session quality
        session_quality = self.nq_stats.get_session_quality(nq_stats)

        return {
            'signal': final_signal,
            'confidence': combined_confidence,
            'agreement': signals_agree,
            'technical_signal': tech_signal,
            'technical_confidence': tech_confidence,
            'nq_stats_direction': nq_direction,
            'nq_stats_confidence': nq_confidence,
            'nq_confluence': nq_confluence,
            'position_size_multiplier': adjusted_position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_price': current_price,
            'session_quality': session_quality,
            'hour_segment': hour_filter.get('segment_type', 'unknown'),
            'reason': nq_recommendation.get('reasoning', 'Combined signal agreement')
        }

    def _check_agreement(self, technical_signal: str, nq_stats_direction: str) -> bool:
        """
        Check if technical and NQ Stats signals agree

        Args:
            technical_signal: Technical analysis signal
            nq_stats_direction: NQ Stats direction

        Returns:
            True if signals agree, False otherwise
        """
        # Normalize signals
        technical_bullish = technical_signal in ['BUY', 'LONG']
        technical_bearish = technical_signal in ['SELL', 'SHORT']

        nq_bullish = nq_stats_direction in ['LONG', 'LONG_SMALL']
        nq_bearish = nq_stats_direction in ['SHORT', 'SHORT_SMALL']

        # Check agreement
        if technical_bullish and nq_bullish:
            return True
        elif technical_bearish and nq_bearish:
            return True
        elif technical_signal == 'WAIT' or nq_stats_direction == 'WAIT':
            return False
        else:
            return False

    def get_risk_parameters(self, signal: Dict) -> Dict:
        """
        Extract risk management parameters from signal

        Args:
            signal: Combined signal from analyze()

        Returns:
            Risk parameters for trade execution
        """
        return {
            'position_size_multiplier': signal.get('position_size_multiplier', 1.0),
            'stop_loss': signal.get('stop_loss'),
            'take_profit': signal.get('take_profit'),
            'max_risk_per_trade': self.config.get('max_risk_per_trade', 0.02),
            'confidence_required': signal.get('confidence', 0.5)
        }

    def format_signal_summary(self, signal: Dict) -> str:
        """
        Format signal into human-readable summary

        Args:
            signal: Combined signal

        Returns:
            Formatted summary string
        """
        direction = signal.get('signal', 'WAIT')
        confidence = signal.get('confidence', 0.0) * 100
        confluence = signal.get('nq_confluence', 0)
        session = signal.get('session_quality', {}).get('quality', 'UNKNOWN')

        if direction == 'WAIT':
            return f"⏸️  WAIT - {signal.get('reason', 'No clear setup')}"

        emoji = "📈" if direction == 'LONG' else "📉"

        return (
            f"{emoji} {direction} Signal\n"
            f"   Confidence: {confidence:.1f}%\n"
            f"   Confluence: {confluence} signals\n"
            f"   Session Quality: {session}\n"
            f"   Position Size: {signal.get('position_size_multiplier', 1.0):.2f}x\n"
            f"   Entry: ${signal.get('entry_price', 0):.2f}\n"
            f"   Stop Loss: ${signal.get('stop_loss', 0):.2f}\n"
            f"   Take Profit: ${signal.get('take_profit', 0):.2f}"
        )
