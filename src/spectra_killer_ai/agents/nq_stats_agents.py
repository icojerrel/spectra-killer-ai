"""
Specialized NQ Stats Trading Agents

Each agent specializes in ONE specific NQ Stats methodology, trading
only when their edge is active and probability is high.

Architecture:
- 8 Specialized Agents (one per methodology)
- Agent Coordinator (orchestrates all agents)
- Multi-Agent Portfolio Manager (allocates capital)
- Signal Aggregator (combines signals)

Benefits:
- Time-based specialization (each agent active at optimal times)
- Independent optimization (each agent tunes own parameters)
- Portfolio diversification (8 different edges)
- Risk distribution (not dependent on single strategy)
- Higher overall win rate (specialists > generalists)
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
from enum import Enum
from dataclasses import dataclass
import pandas as pd

from .nq_stats_analyzer import NQStatsAnalyzer

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent operational status"""
    ACTIVE = "active"           # Currently trading
    WAITING = "waiting"         # Waiting for setup
    MONITORING = "monitoring"   # Watching but not trading
    DISABLED = "disabled"       # Turned off


@dataclass
class AgentSignal:
    """Trading signal from specialized agent"""
    agent_name: str
    signal_type: str           # LONG, SHORT, WAIT, CLOSE
    probability: float         # Expected win probability
    confidence: float          # Agent's confidence (0-1)
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: float       # Recommended size
    reasoning: str
    timestamp: datetime
    metadata: Dict


class BaseNQStatsAgent:
    """
    Base class for specialized NQ Stats trading agents

    Each agent focuses on ONE specific edge and trades ONLY when
    that edge is active with high probability.
    """

    def __init__(self, name: str, config: Dict, analyzer: NQStatsAnalyzer):
        """
        Initialize base agent

        Args:
            name: Agent identifier
            config: Agent-specific configuration
            analyzer: Shared NQ Stats analyzer
        """
        self.name = name
        self.config = config
        self.analyzer = analyzer

        # Agent state
        self.status = AgentStatus.WAITING
        self.current_position = None
        self.pnl = 0.0
        self.trades_count = 0
        self.wins = 0
        self.losses = 0

        # Settings
        self.min_probability = config.get('min_probability', 0.70)
        self.position_size_pct = config.get('position_size_pct', 0.10)  # 10% of capital
        self.enabled = config.get('enabled', True)

        logger.info(f"{self.name} agent initialized (min_prob={self.min_probability})")

    def should_be_active(self, current_time: datetime, analysis: Dict) -> bool:
        """
        Check if agent should be active based on time and conditions

        To be overridden by each specialized agent
        """
        raise NotImplementedError("Subclass must implement should_be_active()")

    def generate_signal(self, data: pd.DataFrame, analysis: Dict,
                       current_time: datetime) -> Optional[AgentSignal]:
        """
        Generate trading signal if edge is present

        To be overridden by each specialized agent
        """
        raise NotImplementedError("Subclass must implement generate_signal()")

    def update_status(self, current_time: datetime, analysis: Dict):
        """Update agent status based on current conditions"""
        if not self.enabled:
            self.status = AgentStatus.DISABLED
        elif self.should_be_active(current_time, analysis):
            self.status = AgentStatus.ACTIVE
        else:
            self.status = AgentStatus.MONITORING

    def record_trade(self, outcome: str, pnl: float):
        """Record trade outcome"""
        self.trades_count += 1
        self.pnl += pnl

        if outcome == 'win':
            self.wins += 1
        else:
            self.losses += 1

    def get_win_rate(self) -> float:
        """Calculate win rate"""
        if self.trades_count == 0:
            return 0.0
        return (self.wins / self.trades_count) * 100

    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            'name': self.name,
            'status': self.status.value,
            'total_trades': self.trades_count,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.get_win_rate(),
            'pnl': self.pnl,
            'current_position': self.current_position is not None
        }


# ============================================================================
# SPECIALIZED AGENTS - One per NQ Stats Methodology
# ============================================================================

class SDEVMeanReversionAgent(BaseNQStatsAgent):
    """
    Agent 1: SDEV Mean Reversion Specialist

    Edge: Price at ±1.5σ or beyond → Mean reversion (85-95% probability)

    Active: Always (24/7)
    Strategy: Fade extremes, trade back to mean
    Win Probability: 70-95% (depending on distance)
    """

    def __init__(self, config: Dict, analyzer: NQStatsAnalyzer):
        super().__init__("SDEV_MeanReversion", config, analyzer)
        self.min_sdev_distance = config.get('min_sdev_distance', 1.5)

    def should_be_active(self, current_time: datetime, analysis: Dict) -> bool:
        """Active when price at extremes"""
        sdev = analysis.get('sdev_analysis', {})
        if not sdev.get('available'):
            return False

        daily = sdev.get('sdev_levels', {}).get('daily', {})
        distance = abs(daily.get('sdev_distance', 0))

        # Active when beyond 1.5 SDEV
        return distance >= self.min_sdev_distance

    def generate_signal(self, data: pd.DataFrame, analysis: Dict,
                       current_time: datetime) -> Optional[AgentSignal]:
        """Generate mean reversion signal"""
        sdev = analysis.get('sdev_analysis', {})
        if not sdev.get('available'):
            return None

        daily = sdev.get('sdev_levels', {}).get('daily', {})
        distance = daily.get('sdev_distance', 0)
        reversion_prob = daily.get('reversion_probability', 0)
        levels = daily.get('levels', {})

        if abs(distance) < self.min_sdev_distance:
            return None

        if reversion_prob < self.min_probability:
            return None

        current_price = float(data['close'].iloc[-1])

        # Fade the extreme
        if distance > 0:  # Price above mean → SHORT
            signal_type = 'SHORT'
            stop_loss = levels.get('+2.0σ')  # Stop beyond 2σ
            take_profit = daily.get('open')  # Target = mean (session open)
        else:  # Price below mean → LONG
            signal_type = 'LONG'
            stop_loss = levels.get('-2.0σ')
            take_profit = daily.get('open')

        return AgentSignal(
            agent_name=self.name,
            signal_type=signal_type,
            probability=reversion_prob,
            confidence=min(abs(distance) / 2.0, 0.95),  # Higher confidence at extremes
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=self.position_size_pct,
            reasoning=f"Mean reversion from {distance:.2f}σ (probability={reversion_prob*100:.0f}%)",
            timestamp=current_time,
            metadata={'sdev_distance': distance, 'zone': daily.get('zone')}
        )


class HourSegmentAgent(BaseNQStatsAgent):
    """
    Agent 2: Hour Segment Specialist (First 20 Minutes)

    Edge: First 20 min of hour → 89% retracement probability

    Active: Minutes 0-20 of each hour
    Strategy: Trade retracements after sweep
    Win Probability: 89%
    """

    def __init__(self, config: Dict, analyzer: NQStatsAnalyzer):
        super().__init__("HourSegment_Retracement", config, analyzer)

    def should_be_active(self, current_time: datetime, analysis: Dict) -> bool:
        """Active during first 20 minutes of any hour"""
        hour_stats = analysis.get('hour_stats', {})
        if not hour_stats.get('available'):
            return False

        segment = hour_stats.get('segment', 2)
        is_9am = hour_stats.get('is_9am_hour', False)

        # Active in segment 1 (first 20 min), but NOT during 9am hour pre-open
        return segment == 1 and not is_9am

    def generate_signal(self, data: pd.DataFrame, analysis: Dict,
                       current_time: datetime) -> Optional[AgentSignal]:
        """Generate retracement signal"""
        hour_stats = analysis.get('hour_stats', {})
        if not hour_stats.get('available'):
            return None

        if hour_stats.get('segment') != 1:
            return None

        retracement_prob = hour_stats.get('retracement_probability', 0)
        if retracement_prob < self.min_probability:
            return None

        # Simple logic: Check if we swept previous hour high/low
        # In production, implement full sweep detection
        current_price = float(data['close'].iloc[-1])

        # Get hour open
        hour = current_time.hour
        minute = current_time.minute

        # Placeholder: In real implementation, detect sweep and trade retracement
        # For now, return monitoring signal

        return None  # Implement sweep detection logic


class RTHBreakAgent(BaseNQStatsAgent):
    """
    Agent 3: RTH Break Specialist

    Edge: RTH opens outside pRTH → 83.29% won't break opposite side

    Active: 9:30am - 4:00pm (RTH hours)
    Strategy: Trade WITH directional bias
    Win Probability: 83.29%
    """

    def __init__(self, config: Dict, analyzer: NQStatsAnalyzer):
        super().__init__("RTH_Break", config, analyzer)

    def should_be_active(self, current_time: datetime, analysis: Dict) -> bool:
        """Active during RTH when scenario is clear"""
        rth = analysis.get('rth_breaks', {})
        if not rth.get('available'):
            return False

        scenario = rth.get('scenario', '')

        # Active when opened outside pRTH (strong directional bias)
        return scenario in ['opened_above_pRTH', 'opened_below_pRTH']

    def generate_signal(self, data: pd.DataFrame, analysis: Dict,
                       current_time: datetime) -> Optional[AgentSignal]:
        """Generate RTH directional signal"""
        rth = analysis.get('rth_breaks', {})
        if not rth.get('available'):
            return None

        scenario = rth.get('scenario', '')
        bias = rth.get('bias', 'NEUTRAL')
        probability = rth.get('probability', 0)

        if bias == 'NEUTRAL' or probability < self.min_probability:
            return None

        current_price = float(data['close'].iloc[-1])
        prev_rth_high = rth.get('prev_rth_high')
        prev_rth_low = rth.get('prev_rth_low')

        if bias == 'BULLISH':
            signal_type = 'LONG'
            stop_loss = prev_rth_low  # 83.29% won't break here
            take_profit = current_price * 1.01  # 1% target
        else:  # BEARISH
            signal_type = 'SHORT'
            stop_loss = prev_rth_high
            take_profit = current_price * 0.99

        return AgentSignal(
            agent_name=self.name,
            signal_type=signal_type,
            probability=probability,
            confidence=0.83,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=self.position_size_pct * 1.5,  # 50% larger (high confidence)
            reasoning=f"RTH {scenario} - {bias} bias ({probability*100:.1f}% edge)",
            timestamp=current_time,
            metadata={'scenario': scenario, 'bias': bias}
        )


class NineAmContinuationAgent(BaseNQStatsAgent):
    """
    Agent 4: 9am Hour Continuation Specialist (STRONGEST EDGE)

    Edge: 9am hour close direction → 70% NY session continuation

    Active: After 10:00am (when 9am hour is complete)
    Strategy: Trade WITH 9am hour direction
    Win Probability: 70%
    """

    def __init__(self, config: Dict, analyzer: NQStatsAnalyzer):
        super().__init__("9am_Continuation", config, analyzer)

    def should_be_active(self, current_time: datetime, analysis: Dict) -> bool:
        """Active after 10am when 9am hour is complete"""
        hour_9am = analysis.get('9am_continuation', {})
        return hour_9am.get('available', False)

    def generate_signal(self, data: pd.DataFrame, analysis: Dict,
                       current_time: datetime) -> Optional[AgentSignal]:
        """Generate 9am continuation signal"""
        hour_9am = analysis.get('9am_continuation', {})
        if not hour_9am.get('available'):
            return None

        bias = hour_9am.get('bias', 'NEUTRAL')
        probability = hour_9am.get('ny_session_continuation_probability', 0)

        if bias == 'NEUTRAL' or probability < self.min_probability:
            return None

        current_price = float(data['close'].iloc[-1])

        # Use SDEV for TP/SL
        sdev = analysis.get('sdev_analysis', {}).get('sdev_levels', {}).get('daily', {})
        levels = sdev.get('levels', {})

        if bias == 'BULLISH':
            signal_type = 'LONG'
            stop_loss = levels.get('-1.0σ', current_price * 0.99)
            take_profit = levels.get('+1.0σ', current_price * 1.01)
        else:  # BEARISH
            signal_type = 'SHORT'
            stop_loss = levels.get('+1.0σ', current_price * 1.01)
            take_profit = levels.get('-1.0σ', current_price * 0.99)

        return AgentSignal(
            agent_name=self.name,
            signal_type=signal_type,
            probability=probability,
            confidence=0.70,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=self.position_size_pct * 2.0,  # DOUBLE size (strongest edge!)
            reasoning=f"9am hour closed {hour_9am.get('hour_9am_direction')} - {probability*100:.0f}% continuation",
            timestamp=current_time,
            metadata={'9am_direction': hour_9am.get('hour_9am_direction')}
        )


class InitialBalanceAgent(BaseNQStatsAgent):
    """
    Agent 5: Initial Balance Breakout Specialist

    Edge: IB close position → 74-81% directional break

    Active: After 10:30am (when IB is complete)
    Strategy: Trade IB breakout direction
    Win Probability: 74-81%
    """

    def __init__(self, config: Dict, analyzer: NQStatsAnalyzer):
        super().__init__("InitialBalance_Breakout", config, analyzer)

    def should_be_active(self, current_time: datetime, analysis: Dict) -> bool:
        """Active after 10:30am when IB is formed"""
        ib = analysis.get('initial_balance', {})
        return ib.get('available', False)

    def generate_signal(self, data: pd.DataFrame, analysis: Dict,
                       current_time: datetime) -> Optional[AgentSignal]:
        """Generate IB breakout signal"""
        ib = analysis.get('initial_balance', {})
        if not ib.get('available'):
            return None

        expected_break = ib.get('expected_break_direction', '')
        probability = ib.get('directional_break_probability', 0)
        ib_high = ib.get('ib_high')
        ib_low = ib.get('ib_low')

        if probability < self.min_probability:
            return None

        current_price = float(data['close'].iloc[-1])

        if expected_break == 'HIGH':
            signal_type = 'LONG'
            stop_loss = ib_low
            take_profit = ib_high + (ib_high - ib_low)  # IB range extension
        else:  # LOW
            signal_type = 'SHORT'
            stop_loss = ib_high
            take_profit = ib_low - (ib_high - ib_low)

        return AgentSignal(
            agent_name=self.name,
            signal_type=signal_type,
            probability=probability,
            confidence=probability,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=self.position_size_pct * 1.3,  # 30% larger
            reasoning=f"IB break expected {expected_break} ({probability*100:.0f}% probability)",
            timestamp=current_time,
            metadata={'ib_high': ib_high, 'ib_low': ib_low, 'close_position': ib.get('close_position')}
        )


class NoonCurveAgent(BaseNQStatsAgent):
    """
    Agent 6: Noon Curve Specialist

    Edge: AM extreme → 74.3% opposite extreme in PM

    Active: After 12:00pm (noon)
    Strategy: Trade for opposite extreme
    Win Probability: 74.3%
    """

    def __init__(self, config: Dict, analyzer: NQStatsAnalyzer):
        super().__init__("NoonCurve_Opposite", config, analyzer)

    def should_be_active(self, current_time: datetime, analysis: Dict) -> bool:
        """Active after noon"""
        noon_curve = analysis.get('noon_curve', {})
        return noon_curve.get('available', False) and noon_curve.get('is_past_noon', False)

    def generate_signal(self, data: pd.DataFrame, analysis: Dict,
                       current_time: datetime) -> Optional[AgentSignal]:
        """Generate noon curve signal"""
        noon_curve = analysis.get('noon_curve', {})
        if not noon_curve.get('available'):
            return None

        am_extreme = noon_curve.get('am_extreme', '')
        pm_expected = noon_curve.get('pm_expected_extreme', '')
        probability = noon_curve.get('opposite_extreme_probability', 0)

        if probability < self.min_probability:
            return None

        current_price = float(data['close'].iloc[-1])
        am_high = noon_curve.get('am_high')
        am_low = noon_curve.get('am_low')

        if pm_expected == 'LOW':
            signal_type = 'SHORT'
            stop_loss = am_high
            take_profit = am_low * 0.999  # Just below AM low
        else:  # HIGH
            signal_type = 'LONG'
            stop_loss = am_low
            take_profit = am_high * 1.001  # Just above AM high

        return AgentSignal(
            agent_name=self.name,
            signal_type=signal_type,
            probability=probability,
            confidence=0.74,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=self.position_size_pct,
            reasoning=f"Noon curve: AM set {am_extreme}, expecting PM {pm_expected} ({probability*100:.1f}%)",
            timestamp=current_time,
            metadata={'am_extreme': am_extreme, 'pm_expected': pm_expected}
        )


class SessionPatternAgent(BaseNQStatsAgent):
    """
    Agent 7: ALN Session Pattern Specialist

    Edge: Session relationships → 73-98% breakout probabilities

    Active: During NY session when pattern is clear
    Strategy: Trade highest probability session break
    Win Probability: 73-98%
    """

    def __init__(self, config: Dict, analyzer: NQStatsAnalyzer):
        super().__init__("SessionPattern_Breakout", config, analyzer)

    def should_be_active(self, current_time: datetime, analysis: Dict) -> bool:
        """Active when session pattern is clear"""
        session = analysis.get('session_pattern', {})
        return session.get('available', False) and session.get('probability', 0) >= 0.75

    def generate_signal(self, data: pd.DataFrame, analysis: Dict,
                       current_time: datetime) -> Optional[AgentSignal]:
        """Generate session breakout signal"""
        session = analysis.get('session_pattern', {})
        if not session.get('available'):
            return None

        pattern_type = session.get('pattern_type', '')
        probability = session.get('probability', 0)

        if probability < self.min_probability:
            return None

        current_price = float(data['close'].iloc[-1])
        london_high = session.get('london_high')
        london_low = session.get('london_low')

        # Logic depends on pattern type
        if pattern_type == 'london_engulfs_asia':
            # 98% NY breaks at least one side
            # Trade whichever side is closer
            if current_price > (london_high + london_low) / 2:
                signal_type = 'LONG'
                stop_loss = london_low
                take_profit = london_high * 1.005
            else:
                signal_type = 'SHORT'
                stop_loss = london_high
                take_profit = london_low * 0.995
        else:
            # Other patterns - similar logic
            return None

        return AgentSignal(
            agent_name=self.name,
            signal_type=signal_type,
            probability=probability,
            confidence=min(probability, 0.95),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=self.position_size_pct * 1.5,  # 50% larger (high probability)
            reasoning=f"Session pattern: {pattern_type} ({probability*100:.0f}% edge)",
            timestamp=current_time,
            metadata={'pattern_type': pattern_type}
        )


class MorningJudasAgent(BaseNQStatsAgent):
    """
    Agent 8: Morning Judas Continuation Specialist

    Edge: 9:40am-10:00am direction → 64-70% continuation to noon

    Active: After 10:00am
    Strategy: Trade WITH morning momentum (not against!)
    Win Probability: 64-70%
    """

    def __init__(self, config: Dict, analyzer: NQStatsAnalyzer):
        super().__init__("MorningJudas_Continuation", config, analyzer)

    def should_be_active(self, current_time: datetime, analysis: Dict) -> bool:
        """Active after 10am when judas is identifiable"""
        judas = analysis.get('morning_judas', {})
        return judas.get('available', False)

    def generate_signal(self, data: pd.DataFrame, analysis: Dict,
                       current_time: datetime) -> Optional[AgentSignal]:
        """Generate judas continuation signal"""
        judas = analysis.get('morning_judas', {})
        if not judas.get('available'):
            return None

        judas_type = judas.get('judas_type', '')
        outcome = judas.get('outcome', '')
        probability = judas.get('probability', 0)

        if outcome != 'CONTINUATION' or probability < self.min_probability:
            return None

        current_price = float(data['close'].iloc[-1])

        # Trade WITH continuation
        if judas_type == 'UP':
            signal_type = 'LONG'
            stop_loss = judas.get('price_940', current_price * 0.99)
            take_profit = current_price * 1.015  # 1.5% target
        else:  # DOWN
            signal_type = 'SHORT'
            stop_loss = judas.get('price_940', current_price * 1.01)
            take_profit = current_price * 0.985

        return AgentSignal(
            agent_name=self.name,
            signal_type=signal_type,
            probability=probability,
            confidence=probability,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=self.position_size_pct,
            reasoning=f"Morning Judas {judas_type} continuation ({probability*100:.0f}%)",
            timestamp=current_time,
            metadata={'judas_type': judas_type, 'outcome': outcome}
        )


# ============================================================================
# AGENT FACTORY
# ============================================================================

def create_all_agents(config: Dict, analyzer: NQStatsAnalyzer) -> List[BaseNQStatsAgent]:
    """
    Factory function to create all 8 specialized agents

    Args:
        config: Configuration for all agents
        analyzer: Shared NQ Stats analyzer

    Returns:
        List of all specialized agents
    """
    agents = [
        SDEVMeanReversionAgent(config.get('sdev_agent', {}), analyzer),
        HourSegmentAgent(config.get('hour_agent', {}), analyzer),
        RTHBreakAgent(config.get('rth_agent', {}), analyzer),
        NineAmContinuationAgent(config.get('9am_agent', {}), analyzer),
        InitialBalanceAgent(config.get('ib_agent', {}), analyzer),
        NoonCurveAgent(config.get('noon_agent', {}), analyzer),
        SessionPatternAgent(config.get('session_agent', {}), analyzer),
        MorningJudasAgent(config.get('judas_agent', {}), analyzer),
    ]

    logger.info(f"Created {len(agents)} specialized NQ Stats agents")
    return agents
