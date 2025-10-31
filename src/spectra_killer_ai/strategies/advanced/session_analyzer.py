"""
Session-Based Trading Analyzer
Implements trading session analysis based on Luckshury's session methodology
Focuses on high-liquidity sessions for optimal trading conditions
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Tuple, Optional
import pytz
import logging

logger = logging.getLogger(__name__)


class SessionAnalyzer:
    """Advanced trading session analysis and filtering"""
    
    def __init__(self, config: Dict):
        """
        Initialize session analyzer
        
        Args:
            config: Session analysis configuration
        """
        self.config = config
        
        # Session time zones and hours (UTC)
        self.sessions = {
            'sydney': {
                'timezone': 'Australia/Sydney',
                'start_utc': time(22, 0),   # 22:00 UTC
                'end_utc': time(7, 0),      # 07:00 UTC
                'liquidity_multiplier': 0.6,
                'volume_multiplier': 0.7,
                'risk_adjustment': 0.8
            },
            'tokyo': {
                'timezone': 'Asia/Tokyo',
                'start_utc': time(23, 0),   # 23:00 UTC
                'end_utc': time(8, 0),      # 08:00 UTC
                'liquidity_multiplier': 0.7,
                'volume_multiplier': 0.8,
                'risk_adjustment': 0.85
            },
            'london': {
                'timezone': 'Europe/London',
                'start_utc': time(8, 0),    # 08:00 UTC
                'end_utc': time(17, 0),     # 17:00 UTC
                'liquidity_multiplier': 1.3,
                'volume_multiplier': 1.4,
                'risk_adjustment': 1.2
            },
            'new_york': {
                'timezone': 'America/New_York',
                'start_utc': time(13, 0),   # 13:00 UTC
                'end_utc': time(22, 0),     # 22:00 UTC
                'liquidity_multiplier': 1.4,
                'volume_multiplier': 1.5,
                'risk_adjustment': 1.1
            },
            'overlap': {
                'timezone': 'UTC',
                'start_utc': time(13, 0),   # London/NY overlap
                'end_utc': time(17, 0),
                'liquidity_multiplier': 2.0,
                'volume_multiplier': 2.0,
                'risk_adjustment': 1.0
            }
        }
        
        # Trading preferences
        self.preferred_sessions = config.get('preferred_sessions', ['london', 'new_york', 'overlap'])
        self.min_liquidity_threshold = config.get('min_liquidity_threshold', 0.8)
        self.auto_session_filtering = config.get('auto_session_filtering', True)
        
        # Risk adjustments by session
        self.session_risk_multipliers = config.get('session_risk_multipliers', {
            'low_liquidity': 0.5,    # Reduce position size by 50%
            'normal_liquidity': 1.0,  # Normal position size
            'high_liquidity': 1.5     # Increase position size by 50%
        })
        
    def analyze(self, data: pd.DataFrame, current_time: Optional[datetime] = None) -> Dict:
        """
        Perform comprehensive session analysis
        
        Args:
            data: OHLCV data with datetime index
            current_time: Current time (defaults to now)
            
        Returns:
            Session analysis results
        """
        try:
            if current_time is None:
                current_time = datetime.now(pytz.UTC)
            
            # Convert to UTC if needed
            if current_time.tzinfo is None:
                current_time = pytz.UTC.localize(current_time)
            else:
                current_time = current_time.astimezone(pytz.UTC)
            
            # Determine current sessions
            current_sessions = self._get_current_sessions(current_time.time())
            
            # Analyze session characteristics
            session_analysis = self._analyze_session_characteristics(current_sessions)
            
            # Calculate session-specific parameters
            session_params = self._calculate_session_parameters(current_sessions)
            
            # Analyze historical session performance
            session_performance = self._analyze_session_performance(data, current_sessions)
            
            # Generate session-based signals
            session_signals = self._generate_session_signals(
                current_sessions, session_analysis, session_performance
            )
            
            # Determine trading recommendation
            trading_recommendation = self._get_trading_recommendation(
                current_sessions, session_analysis, session_params
            )
            
            return {
                'current_time': current_time.isoformat(),
                'current_sessions': current_sessions,
                'session_analysis': session_analysis,
                'session_parameters': session_params,
                'session_performance': session_performance,
                'session_signals': session_signals,
                'trading_recommendation': trading_recommendation,
                'market_hours_info': self._get_market_hours_info(current_time),
                'upcoming_sessions': self._get_upcoming_sessions(current_time)
            }
            
        except Exception as e:
            logger.error(f"Error in session analysis: {e}")
            return self._empty_result()
    
    def _get_current_sessions(self, current_time: time) -> List[str]:
        """Get active sessions for current time"""
        active_sessions = []
        
        for session_name, session_info in self.sessions.items():
            start_time = session_info['start_utc']
            end_time = session_info['end_utc']
            
            # Check if current time is within session hours
            if self._is_time_in_session(current_time, start_time, end_time):
                active_sessions.append(session_name)
        
        return active_sessions
    
    def _is_time_in_session(self, current_time: time, session_start: time, session_end: time) -> bool:
        """Check if current time falls within session hours"""
        if session_start <= session_end:
            # Normal session (e.g., 08:00-17:00)
            return session_start <= current_time <= session_end
        else:
            # Overnight session (e.g., 22:00-07:00)
            return current_time >= session_start or current_time <= session_end
    
    def _analyze_session_characteristics(self, current_sessions: List[str]) -> Dict:
        """Analyze characteristics of current sessions"""
        if not current_sessions:
            return {
                'liquidity_level': 'none',
                'volume_level': 'none',
                'risk_level': 'none',
                'is_high_liquidity': False,
                'is_preferred_session': False,
                'session_overlap': False
            }
        
        # Calculate aggregate characteristics
        total_liquidity = 0
        total_volume = 0
        total_risk = 0
        
        for session in current_sessions:
            session_info = self.sessions.get(session, {})
            total_liquidity += session_info.get('liquidity_multiplier', 1.0)
            total_volume += session_info.get('volume_multiplier', 1.0)
            total_risk += session_info.get('risk_adjustment', 1.0)
        
        avg_liquidity = total_liquidity / len(current_sessions)
        avg_volume = total_volume / len(current_sessions)
        avg_risk = total_risk / len(current_sessions)
        
        # Determine levels
        liquidity_level = self._categorize_level(avg_liquidity)
        volume_level = self._categorize_level(avg_volume)
        risk_level = self._categorize_level(avg_risk)
        
        # Special conditions
        is_overlap = 'overlap' in current_sessions
        is_high_liquidity = avg_liquidity >= 1.2
        is_preferred = any(session in self.preferred_sessions for session in current_sessions)
        
        return {
            'liquidity_level': liquidity_level,
            'volume_level': volume_level,
            'risk_level': risk_level,
            'is_high_liquidity': is_high_liquidity,
            'is_preferred_session': is_preferred,
            'session_overlap': is_overlap,
            'session_count': len(current_sessions),
            'primary_session': current_sessions[0] if current_sessions else None,
            'avg_liquidity_multiplier': avg_liquidity,
            'avg_volume_multiplier': avg_volume
        }
    
    def _categorize_level(self, value: float) -> str:
        """Categorize multiplier value as low/normal/high"""
        if value < 0.8:
            return 'low'
        elif value > 1.2:
            return 'high'
        else:
            return 'normal'
    
    def _calculate_session_parameters(self, current_sessions: List[str]) -> Dict:
        """Calculate session-specific trading parameters"""
        if not current_sessions:
            return self._get_default_parameters()
        
        # Combine session multipliers
        combined_liquidity = 1.0
        combined_volume = 1.0
        combined_risk = 1.0
        
        for session in current_sessions:
            session_info = self.sessions.get(session, {})
            combined_liquidity *= session_info.get('liquidity_multiplier', 1.0)
            combined_volume *= session_info.get('volume_multiplier', 1.0)
            combined_risk *= session_info.get('risk_adjustment', 1.0)
        
        # Apply filtering logic
        if self.auto_session_filtering and combined_liquidity < self.min_liquidity_threshold:
            # Reduce trading activity in low liquidity
            position_multiplier = self.session_risk_multipliers['low_liquidity']
            signal_reduction = 0.5
        elif combined_liquidity > 1.5:
            # Increase trading activity in high liquidity
            position_multiplier = self.session_risk_multipliers['high_liquidity']
            signal_reduction = 1.2
        else:
            position_multiplier = self.session_risk_multipliers['normal_liquidity']
            signal_reduction = 1.0
        
        return {
            'position_size_multiplier': position_multiplier,
            'signal_strength_multiplier': combined_volume,
            'risk_adjustment_factor': combined_risk,
            'liquidity_multiplier': combined_liquidity,
            'volume_multiplier': combined_volume,
            'signal_reduction_factor': signal_reduction,
            'min_confidence_threshold': 0.6 if combined_liquidity < 0.8 else 0.5,
            'max_positions_allowed': 2 if combined_liquidity < 1.0 else 3
        }
    
    def _analyze_session_performance(self, data: pd.DataFrame, current_sessions: List[str]) -> Dict:
        """Analyze historical performance during current sessions"""
        if len(data) < 50 or not current_sessions:
            return {
                'performance_available': False,
                'session_performance': {},
                'avg_returns': {},
                'volatility': {}
            }
        
        # Group data by session hours
        session_performance = {}
        
        for session in current_sessions:
            session_data = self._filter_by_session_hours(data, session)
            
            if len(session_data) > 10:
                returns = session_data['close'].pct_change().dropna()
                
                session_performance[session] = {
                    'sample_size': len(session_data),
                    'avg_return': np.mean(returns),
                    'std_return': np.std(returns),
                    'sharpe': (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0,
                    'avg_volume': np.mean(session_data['volume']) if 'volume' in session_data.columns else 0,
                    'volatility': np.std(session_data['high'] - session_data['low']),
                    'max_drawdown': self._calculate_max_drawdown(session_data['close']),
                    'win_rate': self._calculate_win_rate(returns)
                }
        
        # Calculate aggregate performance
        all_session_data = pd.DataFrame()
        for session in current_sessions:
            session_data = self._filter_by_session_hours(data, session)
            all_session_data = pd.concat([all_session_data, session_data])
        
        aggregate_performance = {}
        if len(all_session_data) > 10:
            returns = all_session_data['close'].pct_change().dropna()
            aggregate_performance = {
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'sharpe': (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0,
                'volatility': np.std(all_session_data['high'] - all_session_data['low']),
                'win_rate': self._calculate_win_rate(returns)
            }
        
        return {
            'performance_available': True,
            'session_performance': session_performance,
            'aggregate_performance': aggregate_performance,
            'performance_score': self._calculate_performance_score(aggregate_performance)
        }
    
    def _filter_by_session_hours(self, data: pd.DataFrame, session: str) -> pd.DataFrame:
        """Filter data by session trading hours"""
        session_info = self.sessions.get(session)
        if not session_info or not hasattr(data.index, 'hour'):
            return DataFrame()
        
        start_hour = session_info['start_utc'].hour
        end_hour = session_info['end_utc'].hour
        
        if start_hour <= end_hour:
            # Normal session
            mask = (data.index.hour >= start_hour) & (data.index.hour <= end_hour)
        else:
            # Overnight session
            mask = (data.index.hour >= start_hour) | (data.index.hour <= end_hour)
        
        return data[mask]
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate from returns"""
        positive_returns = (returns > 0).sum()
        total_returns = len(returns)
        return (positive_returns / total_returns) * 100 if total_returns > 0 else 0
    
    def _calculate_performance_score(self, performance: Dict) -> float:
        """Calculate overall performance score"""
        if not performance:
            return 0.5  # Neutral score
        
        sharpe = performance.get('sharpe', 0)
        win_rate = performance.get('win_rate', 50) / 100  # Convert to decimal
        
        # Simple scoring function
        score = (sharpe * 0.6) + (win_rate * 0.4)
        
        # Normalize to 0-1 range
        score = max(0, min(1, (score + 1) / 2))  # Shift from [-1,1] to [0,1]
        
        return score
    
    def _generate_session_signals(self, current_sessions: List[str], 
                                 session_analysis: Dict, 
                                 session_performance: Dict) -> List[Dict]:
        """Generate session-specific trading signals"""
        signals = []
        
        # High liquidity signal
        if session_analysis['is_high_liquidity']:
            signals.append({
                'type': 'ACTIVATE_TRADING',
                'source': 'session_liquidity',
                'confidence': 0.8,
                'reason': 'High liquidity session detected - optimal trading conditions'
            })
        
        # Low liquidity warning
        if session_analysis['liquidity_level'] == 'low':
            signals.append({
                'type': 'REDUCE_EXPOSURE',
                'source': 'session_liquidity',
                'confidence': 0.7,
                'reason': 'Low liquidity session - reduce position sizes'
            })
        
        # Preferred session bonus
        if session_analysis['is_preferred_session']:
            signals.append({
                'type': 'INCREASE_CONFIDENCE',
                'source': 'session_preference',
                'confidence': 0.6,
                'reason': 'Preferred trading session - higher confidence in signals'
            })
        
        # Session overlap signal (highest liquidity)
        if session_analysis['session_overlap']:
            signals.append({
                'type': 'MAXIMIZE_OPPORTUNITIES',
                'source': 'session_overlap',
                'confidence': 0.9,
                'reason': 'Session overlap - maximum liquidity and volatility expected'
            })
        
        # Performance-based adjustment
        performance_score = session_performance.get('performance_score', 0.5)
        if performance_score > 0.7:
            signals.append({
                'type': 'INCREASE_FREQUENCY',
                'source': 'session_performance',
                'confidence': performance_score * 0.5,
                'reason': 'Historically strong performance in current session'
            })
        elif performance_score < 0.3:
            signals.append({
                'type': 'DECREASE_FREQUENCY',
                'source': 'session_performance',
                'confidence': (1 - performance_score) * 0.5,
                'reason': 'Poor historical performance in current session'
            })
        
        return signals
    
    def _get_trading_recommendation(self, current_sessions: List[str], 
                                   session_analysis: Dict, 
                                   session_params: Dict) -> Dict:
        """Get overall trading recommendation"""
        if not current_sessions:
            return {
                'should_trade': False,
                'reason': 'Outside major trading sessions',
                'confidence': 0.9,
                'recommended_action': 'WAIT'
            }
        
        # Liquidity check
        if session_analysis['liquidity_level'] == 'low':
            return {
                'should_trade': False,
                'reason': 'Low liquidity session - unfavorable conditions',
                'confidence': 0.8,
                'recommended_action': 'REDUCE_POSITION_SIZE'
            }
        
        # High liquidity confirmation
        if session_analysis['is_high_liquidity']:
            base_confidence = 0.8
            recommended_action = 'INCREASE_POSITION_SIZE' if session_analysis['session_overlap'] else 'TRADE_NORMAL'
        else:
            base_confidence = 0.6
            recommended_action = 'TRADE_NORMAL'
        
        # Adjust for preferred sessions
        if session_analysis['is_preferred_session']:
            base_confidence += 0.1
        
        # Adjust for performance
        performance_score = session_params.get('session_performance', {}).get('performance_score', 0.5)
        if performance_score > 0.7:
            base_confidence += 0.1
        elif performance_score < 0.3:
            base_confidence -= 0.1
        
        base_confidence = max(0.3, min(0.9, base_confidence))
        
        return {
            'should_trade': base_confidence > 0.5,
            'reason': f"Session analysis: {session_analysis['liquidity_level']} liquidity, "
                     f"{'preferred' if session_analysis['is_preferred_session'] else 'non-preferred'} session",
            'confidence': base_confidence,
            'recommended_action': recommended_action,
            'position_size_adjustment': session_params['position_size_multiplier']
        }
    
    def _get_market_hours_info(self, current_time: datetime) -> Dict:
        """Get current market hours information"""
        market_hours = {}
        
        for session_name, session_info in self.sessions.items():
            start_time = current_time.replace(
                hour=session_info['start_utc'].hour,
                minute=session_info['start_utc'].minute,
                second=0,
                microsecond=0
            )
            end_time = current_time.replace(
                hour=session_info['end_utc'].hour,
                minute=session_info['end_utc'].minute,
                second=0,
                microsecond=0
            )
            
            # Handle overnight sessions
            if session_info['start_utc'] > session_info['end_utc']:
                if current_time.hour >= session_info['start_utc'].hour:
                    end_time = end_time + pd.Timedelta(days=1)
                elif current_time.hour < session_info['end_utc'].hour:
                    start_time = start_time - pd.Timedelta(days=1)
            
            is_active = self._is_time_in_session(current_time.time(), 
                                                session_info['start_utc'], 
                                                session_info['end_utc'])
            
            time_until_start = (start_time - current_time).total_seconds() / 3600
            time_until_end = (end_time - current_time).total_seconds() / 3600
            
            market_hours[session_name] = {
                'is_active': is_active,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'time_until_start': time_until_start if time_until_start > 0 else time_until_start + 24,
                'time_until_end': time_until_end,
                'liquidity_rating': session_info['liquidity_multiplier'],
                'volume_rating': session_info['volume_multiplier']
            }
        
        return market_hours
    
    def _get_upcoming_sessions(self, current_time: datetime) -> List[Dict]:
        """Get upcoming trading sessions"""
        upcoming = []
        
        for session_name, session_info in self.sessions.items():
            start_time = current_time.replace(
                hour=session_info['start_utc'].hour,
                minute=session_info['start_utc'].minute,
                second=0,
                microsecond=0
            )
            
            # Adjust for future start times
            if start_time <= current_time:
                start_time = start_time + pd.Timedelta(days=1)
            
            hours_until = (start_time - current_time).total_seconds() / 3600
            
            upcoming.append({
                'session': session_name,
                'start_time': start_time.isoformat(),
                'hours_until': hours_until,
                'liquidity_rating': session_info['liquidity_multiplier'],
                'is_preferred': session_name in self.preferred_sessions
            })
        
        # Sort by time until start
        upcoming.sort(key=lambda x: x['hours_until'])
        
        return upcoming[:3]  # Next 3 sessions
    
    def _get_default_parameters(self) -> Dict:
        """Get default parameters for non-trading hours"""
        return {
            'position_size_multiplier': 0.3,
            'signal_strength_multiplier': 0.5,
            'risk_adjustment_factor': 0.7,
            'liquidity_multiplier': 0.5,
            'volume_multiplier': 0.4,
            'signal_reduction_factor': 0.3,
            'min_confidence_threshold': 0.8,
            'max_positions_allowed': 1
        }
    
    def _empty_result(self) -> Dict:
        """Return empty analysis result"""
        return {
            'current_time': datetime.now(pytz.UTC).isoformat(),
            'current_sessions': [],
            'session_analysis': {},
            'session_parameters': self._get_default_parameters(),
            'session_performance': {'performance_available': False},
            'session_signals': [],
            'trading_recommendation': {
                'should_trade': False,
                'reason': 'Session analysis unavailable',
                'confidence': 0.0,
                'recommended_action': 'WAIT'
            },
            'market_hours_info': {},
            'upcoming_sessions': []
        }
