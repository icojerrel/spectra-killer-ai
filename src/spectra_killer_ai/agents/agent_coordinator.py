"""
NQ Stats Agent Coordinator

Orchestrates 8 specialized NQ Stats trading agents:
- Collects signals from all active agents
- Prioritizes signals based on probability and confluence
- Aggregates multiple signals into portfolio recommendations
- Manages agent lifecycle and monitoring
- Coordinates multi-agent portfolio allocation

Coordination Strategies:
1. Best Signal - Take highest probability signal
2. Confluence Voting - Combine agreeing signals
3. Portfolio Diversification - Trade multiple agents simultaneously
4. Risk-Weighted Allocation - Size based on probability
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from collections import defaultdict

from .nq_stats_agents import (
    BaseNQStatsAgent,
    AgentSignal,
    AgentStatus,
    create_all_agents
)
from ..strategies.advanced.nq_stats_analyzer import NQStatsAnalyzer

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """
    Coordinates multiple specialized NQ Stats trading agents

    Responsibilities:
    - Activate/deactivate agents based on time and conditions
    - Collect signals from all active agents
    - Resolve conflicts when agents disagree
    - Aggregate signals for portfolio management
    - Track agent performance
    - Rebalance agent allocations
    """

    def __init__(self, config: Dict):
        """
        Initialize agent coordinator

        Args:
            config: Coordinator configuration
        """
        self.config = config

        # Initialize NQ Stats analyzer (shared by all agents)
        self.analyzer = NQStatsAnalyzer(config.get('nq_stats', {}))

        # Create all specialized agents
        self.agents = create_all_agents(config.get('agents', {}), self.analyzer)

        # Coordination settings
        self.coordination_mode = config.get('coordination_mode', 'confluence_voting')
        # Modes: 'best_signal', 'confluence_voting', 'portfolio_diversification'

        self.min_confluence_for_trade = config.get('min_confluence', 2)
        self.max_concurrent_positions = config.get('max_concurrent_positions', 3)

        # Performance tracking
        self.signal_history = []
        self.agent_performance = defaultdict(lambda: {
            'signals_generated': 0,
            'signals_taken': 0,
            'wins': 0,
            'losses': 0
        })

        logger.info(f"Agent Coordinator initialized with {len(self.agents)} agents "
                   f"(mode={self.coordination_mode})")

    def analyze_and_coordinate(self, data: pd.DataFrame,
                               current_time: Optional[datetime] = None) -> Dict:
        """
        Main coordination loop:
        1. Run NQ Stats analysis
        2. Update agent statuses
        3. Collect signals from all active agents
        4. Coordinate and prioritize signals
        5. Return aggregated recommendations

        Args:
            data: OHLCV DataFrame
            current_time: Current time

        Returns:
            Coordinated trading recommendations
        """
        if current_time is None:
            current_time = datetime.now()

        # 1. Run comprehensive NQ Stats analysis
        analysis = self.analyzer.analyze(data, current_time)

        # 2. Update all agent statuses
        for agent in self.agents:
            agent.update_status(current_time, analysis)

        # 3. Collect signals from active agents
        active_signals = []
        for agent in self.agents:
            if agent.status == AgentStatus.ACTIVE:
                signal = agent.generate_signal(data, analysis, current_time)
                if signal:
                    active_signals.append(signal)
                    self.agent_performance[agent.name]['signals_generated'] += 1

        logger.info(f"Collected {len(active_signals)} signals from active agents")

        # 4. Coordinate signals based on mode
        coordinated_recommendation = self._coordinate_signals(active_signals, analysis)

        # 5. Return comprehensive results
        return {
            'timestamp': current_time.isoformat(),
            'nq_stats_analysis': analysis,
            'active_agents': [a.name for a in self.agents if a.status == AgentStatus.ACTIVE],
            'agent_signals': active_signals,
            'coordinated_recommendation': coordinated_recommendation,
            'agent_statuses': self._get_all_agent_statuses(),
            'portfolio_allocation': self._calculate_portfolio_allocation(active_signals)
        }

    def _coordinate_signals(self, signals: List[AgentSignal], analysis: Dict) -> Dict:
        """
        Coordinate multiple signals into unified recommendation

        Args:
            signals: List of signals from active agents
            analysis: NQ Stats analysis

        Returns:
            Coordinated recommendation
        """
        if not signals:
            return {
                'action': 'WAIT',
                'reason': 'No active agent signals',
                'confidence': 0.0,
                'signals_count': 0
            }

        if self.coordination_mode == 'best_signal':
            return self._best_signal_mode(signals)
        elif self.coordination_mode == 'confluence_voting':
            return self._confluence_voting_mode(signals)
        elif self.coordination_mode == 'portfolio_diversification':
            return self._portfolio_diversification_mode(signals)
        else:
            return self._best_signal_mode(signals)  # Default

    def _best_signal_mode(self, signals: List[AgentSignal]) -> Dict:
        """
        Mode 1: Best Signal
        Take the highest probability signal

        Simple and conservative approach
        """
        # Sort by probability
        sorted_signals = sorted(signals, key=lambda s: s.probability, reverse=True)
        best = sorted_signals[0]

        return {
            'action': best.signal_type,
            'mode': 'best_signal',
            'primary_agent': best.agent_name,
            'probability': best.probability,
            'confidence': best.confidence,
            'entry_price': best.entry_price,
            'stop_loss': best.stop_loss,
            'take_profit': best.take_profit,
            'position_size': best.position_size,
            'reasoning': f"Best signal from {best.agent_name}: {best.reasoning}",
            'signals_count': len(signals),
            'all_signals': [{'agent': s.agent_name, 'type': s.signal_type,
                           'prob': s.probability} for s in signals]
        }

    def _confluence_voting_mode(self, signals: List[AgentSignal]) -> Dict:
        """
        Mode 2: Confluence Voting
        Combine signals that agree, increase size with confluence

        Balanced approach with risk scaling
        """
        # Count signals by direction
        long_signals = [s for s in signals if s.signal_type in ['LONG', 'BUY']]
        short_signals = [s for s in signals if s.signal_type in ['SHORT', 'SELL']]

        # Determine direction
        if len(long_signals) > len(short_signals):
            direction = 'LONG'
            agreeing_signals = long_signals
        elif len(short_signals) > len(long_signals):
            direction = 'SHORT'
            agreeing_signals = short_signals
        else:
            # Tie - go with higher average probability
            long_avg_prob = sum(s.probability for s in long_signals) / len(long_signals) if long_signals else 0
            short_avg_prob = sum(s.probability for s in short_signals) / len(short_signals) if short_signals else 0

            if long_avg_prob > short_avg_prob:
                direction = 'LONG'
                agreeing_signals = long_signals
            else:
                direction = 'SHORT'
                agreeing_signals = short_signals

        confluence = len(agreeing_signals)

        if confluence < self.min_confluence_for_trade:
            return {
                'action': 'WAIT',
                'mode': 'confluence_voting',
                'reason': f'Insufficient confluence ({confluence} < {self.min_confluence_for_trade})',
                'confidence': 0.0,
                'signals_count': len(signals),
                'confluence': confluence
            }

        # Calculate weighted average probability
        avg_probability = sum(s.probability for s in agreeing_signals) / len(agreeing_signals)

        # Calculate weighted average confidence
        avg_confidence = sum(s.confidence for s in agreeing_signals) / len(agreeing_signals)

        # Aggregate position size (scale with confluence)
        base_size = agreeing_signals[0].position_size
        confluence_multiplier = min(confluence / 2, 2.0)  # Max 2x at 4+ signals
        total_position_size = base_size * confluence_multiplier

        # Use best TP/SL from highest probability signal
        best_signal = max(agreeing_signals, key=lambda s: s.probability)

        return {
            'action': direction,
            'mode': 'confluence_voting',
            'confluence': confluence,
            'probability': avg_probability,
            'confidence': avg_confidence,
            'entry_price': best_signal.entry_price,
            'stop_loss': best_signal.stop_loss,
            'take_profit': best_signal.take_profit,
            'position_size': total_position_size,
            'reasoning': f"{confluence} agents agree on {direction} (avg prob={avg_probability*100:.1f}%)",
            'signals_count': len(signals),
            'agreeing_agents': [s.agent_name for s in agreeing_signals]
        }

    def _portfolio_diversification_mode(self, signals: List[AgentSignal]) -> Dict:
        """
        Mode 3: Portfolio Diversification
        Trade multiple agents simultaneously with risk allocation

        Advanced approach for maximum edge capture
        """
        # Filter high-quality signals
        quality_signals = [s for s in signals if s.probability >= 0.70 and s.confidence >= 0.60]

        if not quality_signals:
            return {
                'action': 'WAIT',
                'mode': 'portfolio_diversification',
                'reason': 'No high-quality signals',
                'confidence': 0.0,
                'signals_count': len(signals)
            }

        # Limit to max concurrent positions
        if len(quality_signals) > self.max_concurrent_positions:
            # Sort by probability and take top N
            quality_signals = sorted(quality_signals, key=lambda s: s.probability, reverse=True)
            quality_signals = quality_signals[:self.max_concurrent_positions]

        # Calculate total portfolio position
        total_position_size = sum(s.position_size for s in quality_signals)

        # Build portfolio
        portfolio_positions = []
        for signal in quality_signals:
            portfolio_positions.append({
                'agent': signal.agent_name,
                'direction': signal.signal_type,
                'probability': signal.probability,
                'entry': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'size': signal.position_size,
                'reasoning': signal.reasoning
            })

        return {
            'action': 'PORTFOLIO',
            'mode': 'portfolio_diversification',
            'positions': portfolio_positions,
            'total_position_size': total_position_size,
            'portfolio_count': len(portfolio_positions),
            'avg_probability': sum(s.probability for s in quality_signals) / len(quality_signals),
            'reasoning': f"Diversified portfolio of {len(portfolio_positions)} high-quality agents",
            'signals_count': len(signals)
        }

    def _get_all_agent_statuses(self) -> List[Dict]:
        """Get status of all agents"""
        return [agent.get_stats() for agent in self.agents]

    def _calculate_portfolio_allocation(self, signals: List[AgentSignal]) -> Dict:
        """Calculate recommended portfolio allocation"""
        if not signals:
            return {'total_allocation': 0.0, 'allocations': []}

        allocations = []
        total = 0.0

        for signal in signals:
            allocations.append({
                'agent': signal.agent_name,
                'allocation': signal.position_size,
                'probability': signal.probability
            })
            total += signal.position_size

        return {
            'total_allocation': total,
            'allocations': allocations,
            'agents_count': len(signals)
        }

    def get_performance_report(self) -> Dict:
        """Get performance report for all agents"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_agents': len(self.agents),
            'agent_performance': {}
        }

        for agent in self.agents:
            stats = agent.get_stats()
            perf = self.agent_performance[agent.name]

            report['agent_performance'][agent.name] = {
                'status': stats['status'],
                'total_trades': stats['total_trades'],
                'win_rate': stats['win_rate'],
                'pnl': stats['pnl'],
                'signals_generated': perf['signals_generated'],
                'signals_taken': perf['signals_taken']
            }

        return report

    def update_agent_result(self, agent_name: str, outcome: str, pnl: float):
        """Update agent with trade result"""
        for agent in self.agents:
            if agent.name == agent_name:
                agent.record_trade(outcome, pnl)
                self.agent_performance[agent_name][outcome + 's'] += 1
                break
