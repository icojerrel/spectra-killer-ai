"""
NQ Stats Analyzer - Probability-Based Trading Statistics
Implements statistical analysis based on 10-20 years of NQ historical data
Adapted for use with any liquid futures/forex instrument

Based on research from nqstats.com (2004-2025 data)
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pytz
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SessionType(Enum):
    """Trading session types"""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    RTH = "rth"  # Regular Trading Hours


class ProbabilityLevel(Enum):
    """Probability classification levels"""
    HIGH = "high"  # >= 75%
    MEDIUM = "medium"  # 51-75%
    LOW = "low"  # < 50%


@dataclass
class SDEVLevels:
    """Standard Deviation levels for a timeframe"""
    timeframe: str
    open_price: float
    sdev_value: float  # The 1.0 SDEV percentage (e.g., 1.376%)

    # Calculated levels
    plus_0_5: float = 0.0
    plus_1_0: float = 0.0
    plus_1_5: float = 0.0
    plus_2_0: float = 0.0
    minus_0_5: float = 0.0
    minus_1_0: float = 0.0
    minus_1_5: float = 0.0
    minus_2_0: float = 0.0

    # Probabilities (static based on normal distribution)
    prob_within_1_sdev: float = 68.27
    prob_above_minus_1_sdev: float = 84.13

    def __post_init__(self):
        """Calculate all SDEV levels"""
        sdev_points = self.open_price * (self.sdev_value / 100)

        self.plus_0_5 = self.open_price + (0.5 * sdev_points)
        self.plus_1_0 = self.open_price + (1.0 * sdev_points)
        self.plus_1_5 = self.open_price + (1.5 * sdev_points)
        self.plus_2_0 = self.open_price + (2.0 * sdev_points)

        self.minus_0_5 = self.open_price - (0.5 * sdev_points)
        self.minus_1_0 = self.open_price - (1.0 * sdev_points)
        self.minus_1_5 = self.open_price - (1.5 * sdev_points)
        self.minus_2_0 = self.open_price - (2.0 * sdev_points)

    def get_current_position(self, current_price: float) -> Tuple[float, str]:
        """
        Get current price position in SDEV terms

        Returns:
            Tuple of (sdev_distance, zone_description)
        """
        price_change = current_price - self.open_price
        sdev_points = self.open_price * (self.sdev_value / 100)
        sdev_distance = price_change / sdev_points if sdev_points > 0 else 0

        # Determine zone
        abs_distance = abs(sdev_distance)
        if abs_distance < 0.5:
            zone = "mean_zone"
        elif abs_distance < 1.0:
            zone = "normal_zone"
        elif abs_distance < 1.5:
            zone = "extended_zone"
        elif abs_distance < 2.0:
            zone = "outlier_zone"
        else:
            zone = "extreme_outlier"

        return sdev_distance, zone

    def get_reversion_probability(self, current_price: float) -> float:
        """
        Calculate mean reversion probability based on current SDEV position

        Returns:
            Probability (0-1) of reversion toward mean
        """
        sdev_distance, zone = self.get_current_position(current_price)
        abs_distance = abs(sdev_distance)

        # Probability increases with distance from mean
        if abs_distance < 0.5:
            return 0.3  # Low reversion pressure
        elif abs_distance < 1.0:
            return 0.5  # Moderate reversion pressure
        elif abs_distance < 1.5:
            return 0.7  # High reversion pressure
        elif abs_distance < 2.0:
            return 0.85  # Very high reversion pressure
        else:
            return 0.95  # Extreme reversion pressure


@dataclass
class HourStats:
    """Hour-by-hour statistical analysis"""
    hour: int
    segment: int  # 1, 2, or 3 (20-min segments)
    retracement_probability: float
    is_high_probability: bool  # >= 75%
    typical_behavior: str  # "wick_formation", "expansion", etc.


@dataclass
class InitialBalance:
    """Initial Balance (9:30am-10:30am) data"""
    high: float
    low: float
    range_size: float
    close_position: str  # "upper_half" or "lower_half"
    break_probability: float
    expected_break_direction: str  # "high" or "low"
    break_direction_probability: float


@dataclass
class SessionPattern:
    """Overnight session pattern analysis"""
    pattern_type: str  # "london_engulfs_asia", etc.
    probability: float
    expected_behavior: str
    risk_level: str


@dataclass
class NQStatsSignal:
    """Signal generated from NQ Stats analysis"""
    signal_type: str
    source: str
    probability: float
    confidence: float
    direction: Optional[str]
    reasoning: str
    supporting_stats: Dict[str, Any] = field(default_factory=dict)


class NQStatsAnalyzer:
    """
    Advanced NQ Stats-based probability analyzer

    Implements all major NQ Stats methodologies:
    - Hour Stats with 20-min segments
    - RTH Breaks
    - 1H Continuation
    - Morning Judas analysis
    - Initial Balance breaks
    - Net Change Standard Deviations
    - Noon Curve
    - ALN Session patterns
    """

    def __init__(self, config: Dict):
        """
        Initialize NQ Stats Analyzer

        Args:
            config: Configuration dictionary with SDEV values, etc.
        """
        self.config = config

        # Standard Deviation values (instrument-specific, from backtesting)
        # Default values for NQ - should be calibrated for each instrument
        self.sdev_values = config.get('sdev_values', {
            'monthly': 4.5,  # %
            'weekly': 2.1,   # %
            'daily': 1.376,  # %
            'hourly': 0.34   # %
        })

        # Probability thresholds
        self.high_probability_threshold = config.get('high_probability_threshold', 0.75)
        self.medium_probability_threshold = config.get('medium_probability_threshold', 0.51)

        # Trading hours (EST)
        self.rth_start = time(9, 30)   # 9:30am
        self.rth_end = time(16, 0)     # 4:00pm
        self.ib_start = time(9, 30)    # Initial Balance start
        self.ib_end = time(10, 30)     # Initial Balance end

        # Session definitions (EST)
        self.sessions = {
            SessionType.ASIAN: {'start': time(20, 0), 'end': time(2, 0)},
            SessionType.LONDON: {'start': time(2, 0), 'end': time(8, 0)},
            SessionType.NEW_YORK: {'start': time(8, 0), 'end': time(16, 0)}
        }

        # Hour segment probabilities (from NQ Stats research)
        self.hour_segment_probs = {
            1: {'retracement': 0.89, 'type': 'wick_formation'},  # First 20 min
            2: {'retracement': 0.47, 'type': 'expansion'},       # Middle 20 min
            3: {'retracement': 0.35, 'type': 'wick_formation'}   # Last 20 min
        }

    def analyze(self, data: pd.DataFrame, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Perform comprehensive NQ Stats analysis

        Args:
            data: OHLCV DataFrame with datetime index
            current_time: Current time (defaults to now)

        Returns:
            Comprehensive analysis results with probabilities and signals
        """
        try:
            if current_time is None:
                current_time = datetime.now(pytz.timezone('US/Eastern'))

            # Ensure timezone
            if current_time.tzinfo is None:
                current_time = pytz.timezone('US/Eastern').localize(current_time)

            analysis_results = {
                'timestamp': current_time.isoformat(),
                'current_time_est': current_time.strftime('%H:%M:%S %Z'),
            }

            # 1. SDEV Analysis (The Bread and Butter Edge)
            sdev_analysis = self._analyze_sdev_levels(data, current_time)
            analysis_results['sdev_analysis'] = sdev_analysis

            # 2. Hour Stats Analysis
            hour_stats = self._analyze_hour_stats(data, current_time)
            analysis_results['hour_stats'] = hour_stats

            # 3. RTH Breaks Analysis
            rth_analysis = self._analyze_rth_breaks(data, current_time)
            analysis_results['rth_breaks'] = rth_analysis

            # 4. 9am Hour Continuation
            continuation_analysis = self._analyze_9am_continuation(data, current_time)
            analysis_results['9am_continuation'] = continuation_analysis

            # 5. Morning Judas Analysis
            judas_analysis = self._analyze_morning_judas(data, current_time)
            analysis_results['morning_judas'] = judas_analysis

            # 6. Initial Balance Analysis
            ib_analysis = self._analyze_initial_balance(data, current_time)
            analysis_results['initial_balance'] = ib_analysis

            # 7. Noon Curve Analysis
            noon_curve = self._analyze_noon_curve(data, current_time)
            analysis_results['noon_curve'] = noon_curve

            # 8. ALN Session Pattern
            session_pattern = self._analyze_aln_sessions(data, current_time)
            analysis_results['session_pattern'] = session_pattern

            # 9. Generate integrated signals
            signals = self._generate_integrated_signals(analysis_results)
            analysis_results['signals'] = signals

            # 10. Calculate overall confidence score
            confidence = self._calculate_overall_confidence(analysis_results)
            analysis_results['overall_confidence'] = confidence

            return analysis_results

        except Exception as e:
            logger.error(f"Error in NQ Stats analysis: {e}", exc_info=True)
            return self._empty_result()

    def _analyze_sdev_levels(self, data: pd.DataFrame, current_time: datetime) -> Dict:
        """
        Analyze Standard Deviation levels for multiple timeframes

        Returns:
            SDEV analysis with levels, probabilities, and trading zones
        """
        try:
            if len(data) == 0:
                return {'available': False}

            current_price = data['close'].iloc[-1]

            # Calculate SDEV levels for each timeframe
            sdev_levels = {}

            # Daily SDEV
            daily_open = self._get_session_open(data, current_time, 'daily')
            if daily_open:
                daily_sdev = SDEVLevels('daily', daily_open, self.sdev_values['daily'])
                sdev_distance, zone = daily_sdev.get_current_position(current_price)
                reversion_prob = daily_sdev.get_reversion_probability(current_price)

                sdev_levels['daily'] = {
                    'open': daily_open,
                    'current_price': current_price,
                    'sdev_distance': round(sdev_distance, 2),
                    'zone': zone,
                    'reversion_probability': round(reversion_prob, 3),
                    'levels': {
                        '+0.5σ': round(daily_sdev.plus_0_5, 2),
                        '+1.0σ': round(daily_sdev.plus_1_0, 2),
                        '+1.5σ': round(daily_sdev.plus_1_5, 2),
                        '+2.0σ': round(daily_sdev.plus_2_0, 2),
                        '-0.5σ': round(daily_sdev.minus_0_5, 2),
                        '-1.0σ': round(daily_sdev.minus_1_0, 2),
                        '-1.5σ': round(daily_sdev.minus_1_5, 2),
                        '-2.0σ': round(daily_sdev.minus_2_0, 2),
                    },
                    'is_trend_day': abs(sdev_distance) > 1.5,
                    'rubber_band_tension': 'extreme' if abs(sdev_distance) > 2.0 else 'high' if abs(sdev_distance) > 1.5 else 'moderate' if abs(sdev_distance) > 1.0 else 'low'
                }

            # Hourly SDEV
            hourly_open = self._get_session_open(data, current_time, 'hourly')
            if hourly_open:
                hourly_sdev = SDEVLevels('hourly', hourly_open, self.sdev_values['hourly'])
                h_sdev_distance, h_zone = hourly_sdev.get_current_position(current_price)
                h_reversion_prob = hourly_sdev.get_reversion_probability(current_price)

                sdev_levels['hourly'] = {
                    'open': hourly_open,
                    'sdev_distance': round(h_sdev_distance, 2),
                    'zone': h_zone,
                    'reversion_probability': round(h_reversion_prob, 3),
                    'levels': {
                        '+1.0σ': round(hourly_sdev.plus_1_0, 2),
                        '-1.0σ': round(hourly_sdev.minus_1_0, 2),
                    }
                }

            return {
                'available': True,
                'sdev_levels': sdev_levels,
                'trading_signal': self._get_sdev_trading_signal(sdev_levels)
            }

        except Exception as e:
            logger.error(f"Error in SDEV analysis: {e}")
            return {'available': False}

    def _analyze_hour_stats(self, data: pd.DataFrame, current_time: datetime) -> Dict:
        """
        Analyze current hour statistics with 20-minute segments

        Returns:
            Hour stats analysis with segment probabilities
        """
        try:
            current_hour = current_time.hour
            current_minute = current_time.minute

            # Determine which 20-min segment we're in
            if current_minute < 20:
                segment = 1
            elif current_minute < 40:
                segment = 2
            else:
                segment = 3

            segment_info = self.hour_segment_probs[segment]

            # Special handling for 9am hour (9:00-9:30)
            is_9am_hour = (current_hour == 9 and current_minute < 30)

            return {
                'available': True,
                'current_hour': current_hour,
                'current_minute': current_minute,
                'segment': segment,
                'segment_type': segment_info['type'],
                'retracement_probability': segment_info['retracement'],
                'is_high_probability_segment': segment == 1,
                'is_9am_hour': is_9am_hour,
                '9am_hour_warning': 'High volatility expected at 9:30am open - first 20min high/low likely not final' if is_9am_hour else None,
                'trading_recommendation': self._get_hour_segment_recommendation(segment, is_9am_hour)
            }

        except Exception as e:
            logger.error(f"Error in hour stats analysis: {e}")
            return {'available': False}

    def _analyze_rth_breaks(self, data: pd.DataFrame, current_time: datetime) -> Dict:
        """
        Analyze RTH vs previous RTH relationship

        Returns:
            RTH break analysis with directional probabilities
        """
        try:
            # Get today's RTH open and yesterday's RTH range
            rth_open = self._get_rth_open(data, current_time)
            prev_rth_high, prev_rth_low = self._get_previous_rth_range(data, current_time)

            if not all([rth_open, prev_rth_high, prev_rth_low]):
                return {'available': False}

            # Determine scenario
            if rth_open > prev_rth_high:
                scenario = 'opened_above_pRTH'
                probability = 0.8329  # 83.29% won't break below pRTH low
                bias = 'BULLISH'
                unlikely_event = f"Break below {prev_rth_low:.2f}"
            elif rth_open < prev_rth_low:
                scenario = 'opened_below_pRTH'
                probability = 0.8329
                bias = 'BEARISH'
                unlikely_event = f"Break above {prev_rth_high:.2f}"
            else:
                scenario = 'opened_inside_pRTH'
                probability = 0.7266  # 72.66% will break at least one side
                bias = 'NEUTRAL'
                unlikely_event = "Staying inside pRTH range"

            return {
                'available': True,
                'scenario': scenario,
                'probability': probability,
                'bias': bias,
                'rth_open': rth_open,
                'prev_rth_high': prev_rth_high,
                'prev_rth_low': prev_rth_low,
                'unlikely_event': unlikely_event,
                'confidence': 'HIGH' if probability > 0.75 else 'MEDIUM',
                'trading_recommendation': f"Strong {bias} bias - trade WITH direction" if bias != 'NEUTRAL' else "Expect range break - prepare for breakout"
            }

        except Exception as e:
            logger.error(f"Error in RTH breaks analysis: {e}")
            return {'available': False}

    def _analyze_9am_continuation(self, data: pd.DataFrame, current_time: datetime) -> Dict:
        """
        Analyze 9am hour continuation probability

        Returns:
            9am continuation analysis with directional bias
        """
        try:
            # Check if we're past 10am (9am hour has closed)
            if current_time.hour < 10:
                return {
                    'available': False,
                    'reason': '9am hour not yet closed - wait until 10:00am'
                }

            # Get 9am hour candle (9:00am - 10:00am)
            hour_9am_open = self._get_hour_open(data, current_time, 9)
            hour_9am_close = self._get_hour_close(data, current_time, 9)

            if not all([hour_9am_open, hour_9am_close]):
                return {'available': False}

            # Determine direction
            if hour_9am_close > hour_9am_open:
                direction = 'GREEN'
                bias = 'BULLISH'
                session_probability = 0.70  # 70% NY session stays bullish
                full_session_probability = 0.67  # 67% entire session stays bullish
            else:
                direction = 'RED'
                bias = 'BEARISH'
                session_probability = 0.70
                full_session_probability = 0.67

            return {
                'available': True,
                'hour_9am_direction': direction,
                'bias': bias,
                'ny_session_continuation_probability': session_probability,
                'full_session_continuation_probability': full_session_probability,
                'confidence': 'HIGH',  # 70% is high probability
                'trading_recommendation': f"Strong {bias} bias for rest of session - trade WITH direction",
                'contrarian_warning': f"Trading against 9am bias has only 30% historical success rate"
            }

        except Exception as e:
            logger.error(f"Error in 9am continuation analysis: {e}")
            return {'available': False}

    def _analyze_morning_judas(self, data: pd.DataFrame, current_time: datetime) -> Dict:
        """
        Analyze morning Judas pattern (continuation vs reversal)

        Returns:
            Morning Judas analysis with continuation probabilities
        """
        try:
            # Need to be past 10:00am for this analysis
            if current_time.hour < 10:
                return {'available': False, 'reason': 'Too early for Judas analysis'}

            # Get key levels
            price_930 = self._get_price_at_time(data, current_time.replace(hour=9, minute=30))
            price_940 = self._get_price_at_time(data, current_time.replace(hour=9, minute=40))
            price_1000 = self._get_price_at_time(data, current_time.replace(hour=10, minute=0))

            if not all([price_930, price_940, price_1000]):
                return {'available': False, 'reason': 'Missing price data'}

            # Determine Judas type
            if price_940 > price_930:
                judas_type = 'UP'
                if price_1000 > price_940:
                    outcome = 'CONTINUATION'
                    probability = 0.64
                else:
                    outcome = 'REVERSAL'
                    probability = 0.36
            else:
                judas_type = 'DOWN'
                if price_1000 < price_940:
                    outcome = 'CONTINUATION'
                    probability = 0.70
                else:
                    outcome = 'REVERSAL'
                    probability = 0.30

            return {
                'available': True,
                'judas_type': judas_type,
                'outcome': outcome,
                'probability': probability,
                'price_930': price_930,
                'price_940': price_940,
                'price_1000': price_1000,
                'myth_busted': True,
                'key_insight': 'Morning Judas is primarily a CONTINUATION pattern, not reversal',
                'continuation_bias': judas_type,
                'trading_recommendation': f"Favor {judas_type} continuation - reversals only 30-36% probable"
            }

        except Exception as e:
            logger.error(f"Error in morning Judas analysis: {e}")
            return {'available': False}

    def _analyze_initial_balance(self, data: pd.DataFrame, current_time: datetime) -> Dict:
        """
        Analyze Initial Balance (9:30am-10:30am) breakout probabilities

        Returns:
            IB analysis with break probabilities and direction
        """
        try:
            # Need to be past 10:30am
            if current_time.hour < 10 or (current_time.hour == 10 and current_time.minute < 30):
                return {
                    'available': False,
                    'reason': 'IB still forming - wait until 10:30am',
                    'time_remaining': str(time(10, 30)) + ' EST'
                }

            # Get IB high and low (9:30am-10:30am)
            ib_data = data.between_time('09:30', '10:30')
            if len(ib_data) == 0:
                return {'available': False}

            ib_high = ib_data['high'].max()
            ib_low = ib_data['low'].min()
            ib_close = ib_data['close'].iloc[-1]
            ib_range = ib_high - ib_low
            ib_midpoint = (ib_high + ib_low) / 2

            # Determine close position
            if ib_close > ib_midpoint:
                close_position = 'upper_half'
                expected_break = 'HIGH'
                break_probability = 0.81  # 81% IB high breaks
            else:
                close_position = 'lower_half'
                expected_break = 'LOW'
                break_probability = 0.74  # 74% IB low breaks

            # Check if we're past noon (83% break probability by then)
            is_past_noon = current_time.hour >= 12
            overall_break_prob = 0.96  # 96% by EOD
            noon_break_prob = 0.83  # 83% by noon

            return {
                'available': True,
                'ib_high': ib_high,
                'ib_low': ib_low,
                'ib_range': ib_range,
                'ib_close': ib_close,
                'close_position': close_position,
                'expected_break_direction': expected_break,
                'directional_break_probability': break_probability,
                'overall_break_probability_eod': overall_break_prob,
                'break_probability_by_noon': noon_break_prob,
                'is_past_noon': is_past_noon,
                'confidence': 'HIGH',
                'trading_recommendation': f"Expect IB {expected_break} break with {break_probability*100:.0f}% probability",
                'target_level': ib_high if expected_break == 'HIGH' else ib_low
            }

        except Exception as e:
            logger.error(f"Error in IB analysis: {e}")
            return {'available': False}

    def _analyze_noon_curve(self, data: pd.DataFrame, current_time: datetime) -> Dict:
        """
        Analyze Noon Curve (8am-4pm with 12pm midpoint)

        Returns:
            Noon curve analysis with AM/PM structure expectations
        """
        try:
            # Get 8am-12pm (AM) and 12pm-4pm (PM) data
            am_data = data.between_time('08:00', '12:00')
            pm_data = data.between_time('12:00', '16:00')

            if len(am_data) == 0:
                return {'available': False, 'reason': 'No AM data available'}

            am_high = am_data['high'].max()
            am_low = am_data['low'].min()

            is_past_noon = current_time.hour >= 12

            # If past noon, we can check PM data
            pm_high = pm_data['high'].max() if len(pm_data) > 0 and is_past_noon else None
            pm_low = pm_data['low'].min() if len(pm_data) > 0 and is_past_noon else None

            # Determine AM extreme
            current_price = data['close'].iloc[-1]
            session_open = self._get_session_open(data, current_time, 'daily')

            # Simplified: check if AM went more up or down from open
            am_range_top = am_high - session_open if session_open else 0
            am_range_bottom = session_open - am_low if session_open else 0

            if am_range_top > am_range_bottom:
                am_extreme = 'HIGH'
                pm_expected_extreme = 'LOW'
                probability = 0.743  # 74.3%
            else:
                am_extreme = 'LOW'
                pm_expected_extreme = 'HIGH'
                probability = 0.743

            return {
                'available': True,
                'is_past_noon': is_past_noon,
                'am_high': am_high,
                'am_low': am_low,
                'am_extreme': am_extreme,
                'pm_expected_extreme': pm_expected_extreme,
                'opposite_extreme_probability': probability,
                'pm_high': pm_high,
                'pm_low': pm_low,
                'confidence': 'HIGH',
                'trading_recommendation': f"Expect PM to set {pm_expected_extreme} with {probability*100:.1f}% probability",
                'key_insight': '74.3% of time, high and low form on opposite sides of noon'
            }

        except Exception as e:
            logger.error(f"Error in noon curve analysis: {e}")
            return {'available': False}

    def _analyze_aln_sessions(self, data: pd.DataFrame, current_time: datetime) -> Dict:
        """
        Analyze Asian-London-New York session relationships

        Returns:
            Session pattern analysis with probabilities
        """
        try:
            # Get overnight session ranges (previous day 8pm to today 8am)
            asian_high, asian_low = self._get_session_range(data, current_time, SessionType.ASIAN)
            london_high, london_low = self._get_session_range(data, current_time, SessionType.LONDON)

            if not all([asian_high, asian_low, london_high, london_low]):
                return {'available': False, 'reason': 'Insufficient overnight session data'}

            # Determine pattern
            pattern_type = None
            probability = 0
            expected_behavior = ""

            # Pattern 1: London Engulfs Asia
            if london_high > asian_high and london_low < asian_low:
                pattern_type = "london_engulfs_asia"
                probability = 0.98
                expected_behavior = "98% NY breaks at least one side of London range"

            # Pattern 2: Asia Engulfs London (rare)
            elif asian_high > london_high and asian_low < london_low:
                pattern_type = "asia_engulfs_london"
                probability = 0.95
                expected_behavior = "95% NY breaks London high or low"

            # Pattern 3: London Partial Engulf Upwards
            elif london_high > asian_high and london_low >= asian_low:
                pattern_type = "london_partial_up"
                probability = 0.79
                expected_behavior = "79% NY breaks London high"

            # Pattern 4: London Partial Engulf Downwards
            elif london_low < asian_low and london_high <= asian_high:
                pattern_type = "london_partial_down"
                probability = 0.73
                expected_behavior = "73% NY breaks London low"
            else:
                pattern_type = "no_clear_pattern"
                probability = 0.50
                expected_behavior = "No clear directional edge"

            return {
                'available': True,
                'pattern_type': pattern_type,
                'probability': probability,
                'expected_behavior': expected_behavior,
                'asian_high': asian_high,
                'asian_low': asian_low,
                'london_high': london_high,
                'london_low': london_low,
                'confidence': 'HIGH' if probability >= 0.75 else 'MEDIUM',
                'trading_recommendation': self._get_session_pattern_recommendation(pattern_type, probability),
                'sequential_warning': 'If opposite side taken first, probability drops 40-50%' if probability >= 0.73 else None
            }

        except Exception as e:
            logger.error(f"Error in ALN session analysis: {e}")
            return {'available': False}

    def _generate_integrated_signals(self, analysis: Dict) -> List[NQStatsSignal]:
        """
        Generate integrated trading signals from all analyses

        Returns:
            List of NQStatsSignal objects with probabilities and reasoning
        """
        signals = []

        # SDEV Signal
        sdev_data = analysis.get('sdev_analysis', {})
        if sdev_data.get('available'):
            signal = sdev_data.get('trading_signal')
            if signal:
                signals.append(signal)

        # RTH Break Signal
        rth_data = analysis.get('rth_breaks', {})
        if rth_data.get('available') and rth_data.get('bias') != 'NEUTRAL':
            signals.append(NQStatsSignal(
                signal_type='DIRECTIONAL_BIAS',
                source='rth_breaks',
                probability=rth_data['probability'],
                confidence=0.8 if rth_data['probability'] > 0.8 else 0.6,
                direction=rth_data['bias'],
                reasoning=f"RTH opened {rth_data['scenario']} with {rth_data['probability']*100:.1f}% edge",
                supporting_stats={'scenario': rth_data['scenario'], 'unlikely_event': rth_data['unlikely_event']}
            ))

        # 9am Continuation Signal
        hour_9am = analysis.get('9am_continuation', {})
        if hour_9am.get('available'):
            signals.append(NQStatsSignal(
                signal_type='STRONG_BIAS',
                source='9am_continuation',
                probability=hour_9am['ny_session_continuation_probability'],
                confidence=0.7,
                direction=hour_9am['bias'],
                reasoning=f"9am hour closed {hour_9am['hour_9am_direction']} - {hour_9am['ny_session_continuation_probability']*100:.0f}% continuation probability",
                supporting_stats={'full_session_prob': hour_9am['full_session_continuation_probability']}
            ))

        # Initial Balance Signal
        ib_data = analysis.get('initial_balance', {})
        if ib_data.get('available'):
            direction = 'BULLISH' if ib_data['expected_break_direction'] == 'HIGH' else 'BEARISH'
            signals.append(NQStatsSignal(
                signal_type='BREAKOUT_EXPECTED',
                source='initial_balance',
                probability=ib_data['directional_break_probability'],
                confidence=0.75,
                direction=direction,
                reasoning=f"IB closed in {ib_data['close_position']} - {ib_data['directional_break_probability']*100:.0f}% probability of {ib_data['expected_break_direction']} break",
                supporting_stats={
                    'ib_high': ib_data['ib_high'],
                    'ib_low': ib_data['ib_low'],
                    'target_level': ib_data['target_level']
                }
            ))

        # Session Pattern Signal
        session_data = analysis.get('session_pattern', {})
        if session_data.get('available') and session_data.get('probability', 0) >= 0.70:
            signals.append(NQStatsSignal(
                signal_type='SESSION_PATTERN',
                source='aln_sessions',
                probability=session_data['probability'],
                confidence=0.7 if session_data['probability'] >= 0.75 else 0.6,
                direction=None,  # Pattern-specific
                reasoning=session_data['expected_behavior'],
                supporting_stats={'pattern_type': session_data['pattern_type']}
            ))

        return signals

    def _calculate_overall_confidence(self, analysis: Dict) -> Dict:
        """
        Calculate overall confidence score based on signal confluence

        Returns:
            Overall confidence metrics
        """
        signals = analysis.get('signals', [])

        if not signals:
            return {
                'score': 0.5,
                'level': 'NEUTRAL',
                'confluence_count': 0,
                'recommendation': 'WAIT'
            }

        # Count directional signals
        bullish_signals = [s for s in signals if s.direction == 'BULLISH']
        bearish_signals = [s for s in signals if s.direction == 'BEARISH']

        # Weight by probability
        bullish_weight = sum(s.probability * s.confidence for s in bullish_signals)
        bearish_weight = sum(s.probability * s.confidence for s in bearish_signals)

        # Calculate net confidence
        total_weight = bullish_weight + bearish_weight
        if total_weight > 0:
            net_confidence = (bullish_weight - bearish_weight) / total_weight
        else:
            net_confidence = 0

        # Normalize to 0-1 scale
        confidence_score = (net_confidence + 1) / 2

        # Determine level
        if confidence_score > 0.7:
            level = 'HIGH_BULLISH'
            recommendation = 'LONG'
        elif confidence_score > 0.6:
            level = 'MEDIUM_BULLISH'
            recommendation = 'LONG_SMALL'
        elif confidence_score < 0.3:
            level = 'HIGH_BEARISH'
            recommendation = 'SHORT'
        elif confidence_score < 0.4:
            level = 'MEDIUM_BEARISH'
            recommendation = 'SHORT_SMALL'
        else:
            level = 'NEUTRAL'
            recommendation = 'WAIT'

        return {
            'score': round(confidence_score, 3),
            'level': level,
            'confluence_count': len(signals),
            'bullish_signals': len(bullish_signals),
            'bearish_signals': len(bearish_signals),
            'recommendation': recommendation,
            'position_sizing': 'FULL' if confidence_score > 0.7 or confidence_score < 0.3 else 'REDUCED'
        }

    # Helper methods

    def _get_session_open(self, data: pd.DataFrame, current_time: datetime, timeframe: str) -> Optional[float]:
        """Get open price for specified timeframe"""
        try:
            if timeframe == 'daily':
                # Get today's first price
                today_data = data[data.index.date == current_time.date()]
                if len(today_data) > 0:
                    return today_data['open'].iloc[0]
            elif timeframe == 'hourly':
                # Get current hour's open
                hour_start = current_time.replace(minute=0, second=0, microsecond=0)
                hour_data = data[data.index >= hour_start]
                if len(hour_data) > 0:
                    return hour_data['open'].iloc[0]
            return None
        except:
            return None

    def _get_rth_open(self, data: pd.DataFrame, current_time: datetime) -> Optional[float]:
        """Get RTH open (9:30am) price"""
        try:
            rth_start = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            rth_data = data[data.index >= rth_start]
            if len(rth_data) > 0:
                return rth_data['open'].iloc[0]
            return None
        except:
            return None

    def _get_previous_rth_range(self, data: pd.DataFrame, current_time: datetime) -> Tuple[Optional[float], Optional[float]]:
        """Get previous day's RTH high and low"""
        try:
            prev_day = current_time - timedelta(days=1)
            prev_rth_data = data[data.index.date == prev_day.date()]
            prev_rth_data = prev_rth_data.between_time('09:30', '16:00')

            if len(prev_rth_data) > 0:
                return prev_rth_data['high'].max(), prev_rth_data['low'].min()
            return None, None
        except:
            return None, None

    def _get_hour_open(self, data: pd.DataFrame, current_time: datetime, hour: int) -> Optional[float]:
        """Get open price for specific hour"""
        try:
            target_time = current_time.replace(hour=hour, minute=0, second=0, microsecond=0)
            hour_data = data[data.index >= target_time]
            if len(hour_data) > 0:
                return hour_data['open'].iloc[0]
            return None
        except:
            return None

    def _get_hour_close(self, data: pd.DataFrame, current_time: datetime, hour: int) -> Optional[float]:
        """Get close price for specific hour"""
        try:
            start_time = current_time.replace(hour=hour, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(hours=1)
            hour_data = data[(data.index >= start_time) & (data.index < end_time)]
            if len(hour_data) > 0:
                return hour_data['close'].iloc[-1]
            return None
        except:
            return None

    def _get_price_at_time(self, data: pd.DataFrame, target_time: datetime) -> Optional[float]:
        """Get price at specific time"""
        try:
            # Find closest data point
            time_data = data[data.index >= target_time]
            if len(time_data) > 0:
                return time_data['close'].iloc[0]
            return None
        except:
            return None

    def _get_session_range(self, data: pd.DataFrame, current_time: datetime, session: SessionType) -> Tuple[Optional[float], Optional[float]]:
        """Get high and low for specific session"""
        try:
            session_times = self.sessions.get(session)
            if not session_times:
                return None, None

            # Handle overnight sessions
            if session == SessionType.ASIAN:
                # Previous day 8pm to today 2am
                prev_day = current_time - timedelta(days=1)
                start_time = prev_day.replace(hour=20, minute=0)
                end_time = current_time.replace(hour=2, minute=0)
            elif session == SessionType.LONDON:
                # Today 2am to 8am
                start_time = current_time.replace(hour=2, minute=0)
                end_time = current_time.replace(hour=8, minute=0)
            else:  # New York
                start_time = current_time.replace(hour=8, minute=0)
                end_time = current_time.replace(hour=16, minute=0)

            session_data = data[(data.index >= start_time) & (data.index < end_time)]

            if len(session_data) > 0:
                return session_data['high'].max(), session_data['low'].min()
            return None, None
        except:
            return None, None

    def _get_sdev_trading_signal(self, sdev_levels: Dict) -> Optional[NQStatsSignal]:
        """Generate trading signal from SDEV analysis"""
        try:
            daily = sdev_levels.get('daily', {})
            if not daily:
                return None

            zone = daily.get('zone')
            sdev_distance = daily.get('sdev_distance', 0)
            reversion_prob = daily.get('reversion_probability', 0)

            # Generate signal based on zone
            if zone in ['outlier_zone', 'extreme_outlier']:
                direction = 'BEARISH' if sdev_distance > 0 else 'BULLISH'  # Mean reversion
                return NQStatsSignal(
                    signal_type='MEAN_REVERSION',
                    source='sdev_analysis',
                    probability=reversion_prob,
                    confidence=0.7 if zone == 'extreme_outlier' else 0.6,
                    direction=direction,
                    reasoning=f"Price at {sdev_distance:.2f}σ - {zone} - high reversion probability {reversion_prob*100:.0f}%",
                    supporting_stats={'zone': zone, 'sdev_distance': sdev_distance}
                )
            elif daily.get('is_trend_day'):
                # Trend day - trade with momentum
                direction = 'BULLISH' if sdev_distance > 0 else 'BEARISH'
                return NQStatsSignal(
                    signal_type='TREND_DAY',
                    source='sdev_analysis',
                    probability=0.65,
                    confidence=0.6,
                    direction=direction,
                    reasoning=f"Trend day detected (>1.5σ) - trade WITH momentum",
                    supporting_stats={'sdev_distance': sdev_distance}
                )

            return None
        except:
            return None

    def _get_hour_segment_recommendation(self, segment: int, is_9am_hour: bool) -> str:
        """Get trading recommendation for hour segment"""
        if is_9am_hour:
            return "WAIT - High volatility expected at 9:30am open"

        if segment == 1:
            return "HIGH PROBABILITY - First 20 minutes ideal for retracement entries"
        elif segment == 2:
            return "EXPANSION PHASE - Trade breakouts and momentum"
        else:
            return "SECONDARY WICK ZONE - Prepare for next hour dynamics"

    def _get_session_pattern_recommendation(self, pattern_type: str, probability: float) -> str:
        """Get recommendation based on session pattern"""
        if pattern_type == "london_engulfs_asia":
            return "HIGHEST EDGE - Trade London range breakouts aggressively (98% probability)"
        elif pattern_type == "london_partial_up":
            return f"BULLISH BIAS - {probability*100:.0f}% probability of London high break"
        elif pattern_type == "london_partial_down":
            return f"BEARISH BIAS - {probability*100:.0f}% probability of London low break"
        else:
            return "MODERATE EDGE - Watch for first break direction"

    def _empty_result(self) -> Dict:
        """Return empty result when analysis fails"""
        return {
            'timestamp': datetime.now().isoformat(),
            'available': False,
            'error': 'Analysis unavailable',
            'signals': [],
            'overall_confidence': {
                'score': 0.5,
                'level': 'NEUTRAL',
                'confluence_count': 0,
                'recommendation': 'WAIT'
            }
        }
