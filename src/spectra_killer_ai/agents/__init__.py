"""
NQ Stats Multi-Agent Trading System

8 Specialized agents, each focusing on one NQ Stats methodology:

1. SDEV Mean Reversion - Fade extremes (70-95% probability)
2. Hour Segment Specialist - First 20min retracements (89%)
3. RTH Break - Directional bias (83.29%)
4. 9am Continuation - Strongest edge (70%)
5. Initial Balance - Breakout specialist (74-81%)
6. Noon Curve - Opposite extremes (74.3%)
7. Session Pattern - ALN relationships (73-98%)
8. Morning Judas - Continuation (64-70%)

Coordination Modes:
- Best Signal: Take highest probability
- Confluence Voting: Combine agreeing signals
- Portfolio Diversification: Trade multiple simultaneously
"""

from .nq_stats_agents import (
    BaseNQStatsAgent,
    AgentSignal,
    AgentStatus,
    SDEVMeanReversionAgent,
    HourSegmentAgent,
    RTHBreakAgent,
    NineAmContinuationAgent,
    InitialBalanceAgent,
    NoonCurveAgent,
    SessionPatternAgent,
    MorningJudasAgent,
    create_all_agents
)

from .agent_coordinator import AgentCoordinator

__all__ = [
    # Base classes
    'BaseNQStatsAgent',
    'AgentSignal',
    'AgentStatus',

    # Specialized agents
    'SDEVMeanReversionAgent',
    'HourSegmentAgent',
    'RTHBreakAgent',
    'NineAmContinuationAgent',
    'InitialBalanceAgent',
    'NoonCurveAgent',
    'SessionPatternAgent',
    'MorningJudasAgent',

    # Coordination
    'AgentCoordinator',

    # Factory
    'create_all_agents',
]
