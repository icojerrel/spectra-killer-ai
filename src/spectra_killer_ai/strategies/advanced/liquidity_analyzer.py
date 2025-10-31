"""
Advanced Liquidity Analyzer
Implements liquidity level detection and sweep detection based on Luckshury's methodology
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class LiquidityAnalyzer:
    """Advanced liquidity analysis with high probability level detection"""
    
    def __init__(self, config: Dict):
        """
        Initialize liquidity analyzer
        
        Args:
            config: Liquidity analysis configuration
        """
        self.config = config
        
        # Lookback periods for liquidity analysis
        self.swing_lookback = config.get('swing_lookback', 50)
        self.min_touches = config.get('min_touches', 3)
        self.tolerance_pips = config.get('tolerance_pips', 5)
        self.volume_confirmation = config.get('volume_confirmation', True)
        self.min_volume_multiplier = config.get('min_volume_multiplier', 1.5)
        
        # Session parameters
        self.high_liquidity_sessions = config.get('high_liquidity_sessions', ['london', 'new_york', 'overlap'])
        
    def analyze(self, data: pd.DataFrame, current_price: float) -> Dict:
        """
        Perform comprehensive liquidity analysis
        
        Args:
            data: OHLCV data with volume
            current_price: Current market price
            
        Returns:
            Liquidity analysis results
        """
        try:
            if len(data) < self.swing_lookback:
                logger.warning("Insufficient data for liquidity analysis")
                return self._empty_result()
            
            # Identify liquidity levels
            resistance_levels = self._identify_resistance_levels(data)
            support_levels = self._identify_support_levels(data)
            
            # Check for liquidity sweeps
            sweep_analysis = self._check_liquidity_sweeps(data, current_price, support_levels, resistance_levels)
            
            # Analyze current position relative to liquidity
            current_analysis = self._analyze_current_position(current_price, support_levels, resistance_levels)
            
            # Volume confirmation
            volume_analysis = self._analyze_volume_confirmation(data)
            
            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(
                sweep_analysis, current_analysis, volume_analysis
            )
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'sweep_analysis': sweep_analysis,
                'current_position': current_analysis,
                'volume_confirmation': volume_analysis,
                'liquidity_score': liquidity_score,
                'high_probability_levels': self._get_high_probability_levels(
                    support_levels, resistance_levels, volume_analysis
                ),
                'signals': self._generate_liquidity_signals(
                    sweep_analysis, current_analysis, liquidity_score
                )
            }
            
        except Exception as e:
            logger.error(f"Error in liquidity analysis: {e}")
            return self._empty_result()
    
    def _identify_resistance_levels(self, data: pd.DataFrame) -> List[Dict]:
        """Identify resistance levels using swing highs"""
        highs = data['high'].values
        volumes = data.get('volume', np.ones(len(data)))
        
        resistance_levels = []
        
        # Find swing highs
        for i in range(self.swing_lookback, len(highs) - self.swing_lookback):
            current_high = highs[i]
            
            # Check if this is a swing high
            is_swing_high = True
            for j in range(i - self.swing_lookback, i):
                if highs[j] >= current_high:
                    is_swing_high = False
                    break
            for j in range(i + 1, i + self.swing_lookback + 1):
                if j < len(highs) and highs[j] > current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                # Count touches near this level
                touches = 0
                touch_volumes = []
                
                for j in range(len(highs)):
                    if abs(highs[j] - current_high) <= self.tolerance_pips * 0.0001:  # Convert pips to price
                        touches += 1
                        touch_volumes.append(volumes[j])
                
                if touches >= self.min_touches:
                    resistance_levels.append({
                        'price': current_high,
                        'touches': touches,
                        'avg_volume': np.mean(touch_volumes) if touch_volumes else 0,
                        'strength': touches / 10.0,  # Normalize to 0-1
                        'type': 'resistance'
                    })
        
        # Sort by strength
        resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
        return resistance_levels[:5]  # Return top 5 levels
    
    def _identify_support_levels(self, data: pd.DataFrame) -> List[Dict]:
        """Identify support levels using swing lows"""
        lows = data['low'].values
        volumes = data.get('volume', np.ones(len(data)))
        
        support_levels = []
        
        # Find swing lows
        for i in range(self.swing_lookback, len(lows) - self.swing_lookback):
            current_low = lows[i]
            
            # Check if this is a swing low
            is_swing_low = True
            for j in range(i - self.swing_lookback, i):
                if lows[j] <= current_low:
                    is_swing_low = False
                    break
            for j in range(i + 1, i + self.swing_lookback + 1):
                if j < len(lows) and lows[j] < current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                # Count touches near this level
                touches = 0
                touch_volumes = []
                
                for j in range(len(lows)):
                    if abs(lows[j] - current_low) <= self.tolerance_pips * 0.0001:
                        touches += 1
                        touch_volumes.append(volumes[j])
                
                if touches >= self.min_touches:
                    support_levels.append({
                        'price': current_low,
                        'touches': touches,
                        'avg_volume': np.mean(touch_volumes) if touch_volumes else 0,
                        'strength': touches / 10.0,
                        'type': 'support'
                    })
        
        # Sort by strength
        support_levels.sort(key=lambda x: x['strength'], reverse=True)
        return support_levels[:5]  # Return top 5 levels
    
    def _check_liquidity_sweeps(self, data: pd.DataFrame, current_price: float, 
                               support_levels: List[Dict], resistance_levels: List[Dict]) -> Dict:
        """Check for recent liquidity sweeps"""
        if len(data) < 20:
            return {'sweep_detected': False, 'sweep_type': None, 'swept_level': None}
        
        recent_data = data.tail(20)
        recent_low = recent_data['low'].min()
        recent_high = recent_data['high'].max()
        
        # Check for support sweep (price went below support and came back)
        support_sweep = False
        swept_support = None
        for level in support_levels:
            if recent_low < level['price'] < current_price:
                support_sweep = True
                swept_support = level
                break
        
        # Check for resistance sweep (price went above resistance and came back)
        resistance_sweep = False
        swept_resistance = None
        for level in resistance_levels:
            if recent_high > level['price'] > current_price:
                resistance_sweep = True
                swept_resistance = level
                break
        
        if support_sweep and resistance_sweep:
            sweep_type = 'both'
        elif support_sweep:
            sweep_type = 'support'
        elif resistance_sweep:
            sweep_type = 'resistance'
        else:
            sweep_type = None
        
        return {
            'sweep_detected': support_sweep or resistance_sweep,
            'sweep_type': sweep_type,
            'swept_support': swept_support,
            'swept_resistance': swept_resistance,
            'reversal_potential': self._calculate_reversal_potential(
                support_sweep, resistance_sweep, swept_support, swept_resistance
            )
        }
    
    def _analyze_current_position(self, current_price: float, 
                                 support_levels: List[Dict], resistance_levels: List[Dict]) -> Dict:
        """Analyze current position relative to liquidity levels"""
        if not support_levels and not resistance_levels:
            return {'position': 'neutral', 'distance_to_nearest': None}
        
        # Find nearest support and resistance
        nearest_support = None
        nearest_resistance = None
        min_support_distance = float('inf')
        min_resistance_distance = float('inf')
        
        for level in support_levels:
            distance = current_price - level['price']
            if distance > 0 and distance < min_support_distance:
                min_support_distance = distance
                nearest_support = level
        
        for level in resistance_levels:
            distance = level['price'] - current_price
            if distance > 0 and distance < min_resistance_distance:
                min_resistance_distance = distance
                nearest_resistance = level
        
        # Determine position
        if not support_levels:
            position = 'below_all_resistance'
        elif not resistance_levels:
            position = 'above_all_support'
        elif min_support_distance < min_resistance_distance:
            position = 'near_support'
        elif min_resistance_distance < min_support_distance:
            position = 'near_resistance'
        else:
            position = 'neutral'
        
        return {
            'position': position,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'distance_to_support': min_support_distance if min_support_distance != float('inf') else None,
            'distance_to_resistance': min_resistance_distance if min_resistance_distance != float('inf') else None,
            'in_liquidity_zone': (min(min_support_distance, min_resistance_distance) if 
                                min_support_distance != float('inf') and 
                                min_resistance_distance != float('inf') else None) is not None and 
                               min(min_support_distance, min_resistance_distance) < self.tolerance_pips * 0.0001
        }
    
    def _analyze_volume_confirmation(self, data: pd.DataFrame) -> Dict:
        """Analyze volume for liquidity level confirmation"""
        if 'volume' not in data.columns:
            return {'volume_available': False}
        
        volumes = data['volume'].values
        avg_volume = np.mean(volumes[-20:])  # Recent average
        current_volume = volumes[-1]
        
        volume_spike = current_volume > avg_volume * self.min_volume_multiplier
        volume_trend = 'increasing' if len(volumes) > 5 and np.mean(volumes[-3:]) > np.mean(volumes[-10:-3]) else 'normal'
        
        return {
            'volume_available': True,
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
            'volume_spike': volume_spike,
            'volume_trend': volume_trend,
            'liquidity_confirmation': volume_spike or volume_trend == 'increasing'
        }
    
    def _calculate_liquidity_score(self, sweep_analysis: Dict, 
                                  current_analysis: Dict, volume_analysis: Dict) -> float:
        """Calculate overall liquidity score (0-1)"""
        score = 0.0
        
        # Sweep score (highest importance)
        if sweep_analysis['sweep_detected']:
            reversal_potential = sweep_analysis['reversal_potential']
            score += reversal_potential * 0.4
        
        # Position score
        if current_analysis['in_liquidity_zone']:
            score += 0.2
        elif current_analysis['position'] in ['near_support', 'near_resistance']:
            score += 0.15
        
        # Volume confirmation
        if volume_analysis.get('liquidity_confirmation', False):
            score += 0.2
        
        # Overall liquidity activity
        if volume_analysis.get('volume_spike', False):
            score += 0.2
        
        return min(score, 1.0)
    
    def _get_high_probability_levels(self, support_levels: List[Dict], 
                                    resistance_levels: List[Dict], 
                                    volume_analysis: Dict) -> List[Dict]:
        """Get high probability liquidity levels with volume confirmation"""
        all_levels = support_levels + resistance_levels
        high_prob_levels = []
        
        for level in all_levels:
            level_score = level['strength']
            
            # Boost score if volume confirmation
            if volume_analysis.get('volume_spike', False):
                level_score *= 1.2
            
            # High probability threshold
            if level_score > 0.3:
                level['probability_score'] = level_score
                high_prob_levels.append(level)
        
        # Sort by probability
        high_prob_levels.sort(key=lambda x: x['probability_score'], reverse=True)
        return high_prob_levels[:3]  # Top 3 levels
    
    def _generate_liquidity_signals(self, sweep_analysis: Dict, 
                                   current_analysis: Dict, 
                                   liquidity_score: float) -> List[Dict]:
        """Generate trading signals based on liquidity analysis"""
        signals = []
        
        # Sweep reversal signals
        if sweep_analysis['sweep_detected']:
            if sweep_analysis['sweep_type'] == 'support' and current_analysis['position'] != 'near_resistance':
                signals.append({
                    'type': 'BUY',
                    'source': 'liquidity_sweep',
                    'confidence': sweep_analysis['reversal_potential'] * 0.8,
                    'reason': 'Support sweep detected - potential reversal'
                })
            elif sweep_analysis['sweep_type'] == 'resistance' and current_analysis['position'] != 'near_support':
                signals.append({
                    'type': 'SELL',
                    'source': 'liquidity_sweep',
                    'confidence': sweep_analysis['reversal_potential'] * 0.8,
                    'reason': 'Resistance sweep detected - potential reversal'
                })
        
        # Liquidity zone signals
        if current_analysis['in_liquidity_zone'] and liquidity_score > 0.6:
            if current_analysis['position'] == 'near_support':
                signals.append({
                    'type': 'BUY',
                    'source': 'liquidity_zone',
                    'confidence': liquidity_score * 0.6,
                    'reason': 'At high probability support level'
                })
            elif current_analysis['position'] == 'near_resistance':
                signals.append({
                    'type': 'SELL',
                    'source': 'liquidity_zone',
                    'confidence': liquidity_score * 0.6,
                    'reason': 'At high probability resistance level'
                })
        
        return signals
    
    def _calculate_reversal_potential(self, support_sweep: bool, resistance_sweep: bool,
                                    swept_support: Optional[Dict], 
                                    swept_resistance: Optional[Dict]) -> float:
        """Calculate reversal potential after liquidity sweep"""
        base_potential = 0.7  # Base probability for any sweep
        
        # Adjust based on level strength
        if swept_support:
            base_potential += swept_support['strength'] * 0.3
        if swept_resistance:
            base_potential += swept_resistance['strength'] * 0.3
        
        return min(base_potential, 1.0)
    
    def _empty_result(self) -> Dict:
        """Return empty analysis result"""
        return {
            'support_levels': [],
            'resistance_levels': [],
            'sweep_analysis': {'sweep_detected': False},
            'current_position': {'position': 'neutral'},
            'volume_confirmation': {},
            'liquidity_score': 0.0,
            'high_probability_levels': [],
            'signals': []
        }
