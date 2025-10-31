"""
Advanced Volume Profile Analyzer
Implements volume profile analysis, market depth, and order flow indicators
Based on Luckshury's volume & order flow methodology
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class VolumeProfileAnalyzer:
    """Advanced volume profile and order flow analysis"""
    
    def __init__(self, config: Dict):
        """
        Initialize volume profile analyzer
        
        Args:
            config: Volume profile configuration
        """
        self.config = config
        
        # Volume profile parameters
        self.price_bins = config.get('price_bins', 50)  # Number of price levels
        self.min_volume_threshold = config.get('min_volume_threshold', 100)
        self.poc_weight = config.get('poc_weight', 0.5)  # Point of Control importance
        self.vwap_weight = config.get('vwap_weight', 0.3)
        
        # Order flow parameters
        self.buy_sell_threshold = config.get('buy_sell_threshold', 0.6)
        self.absorption_threshold = config.get('absorption_threshold', 2.0)
        self.climax_multiplier = config.get('climax_multiplier', 3.0)
        
        # Market depth simulation
        self.depth_levels = config.get('depth_levels', 10)
        self.liquidity_ratio = config.get('liquidity_ratio', 1.5)
        
    def analyze(self, data: pd.DataFrame, current_price: float) -> Dict:
        """
        Perform comprehensive volume profile and order flow analysis
        
        Args:
            data: OHLCV data with volume
            current_price: Current market price
            
        Returns:
            Volume profile analysis results
        """
        try:
            if len(data) < 100:
                logger.warning("Insufficient data for volume profile analysis")
                return self._empty_result()
            
            # Check if volume data is available
            if 'volume' not in data.columns:
                logger.warning("No volume data available")
                return self._empty_result()
            
            # Volume profile analysis
            volume_profile = self._calculate_volume_profile(data, current_price)
            poc_analysis = self._find_point_of_control(volume_profile, current_price)
            vwap_analysis = self._calculate_vwap_analysis(data, current_price)
            
            # Order flow analysis
            order_flow = self._analyze_order_flow(data, current_price)
            absorption = self._detect_absorption(data, current_price)
            climax = self._detect_volume_climax(data, current_price)
            
            # Market depth simulation
            market_depth = self._simulate_market_depth(data, current_price)
            
            # Volume imbalance analysis
            imbalance = self._analyze_volume_imbalance(data, order_flow)
            
            # Generate signals
            signals = self._generate_volume_signals(
                poc_analysis, vwap_analysis, order_flow, absorption, climax, market_depth
            )
            
            # Calculate volume confidence score
            confidence_score = self._calculate_volume_confidence(
                volume_profile, order_flow, absorption, climax
            )
            
            return {
                'volume_profile': volume_profile,
                'point_of_control': poc_analysis,
                'vwap': vwap_analysis,
                'order_flow': order_flow,
                'absorption': absorption,
                'volume_climax': climax,
                'market_depth': market_depth,
                'volume_imbalance': imbalance,
                'signals': signals,
                'confidence_score': confidence_score,
                'price_volume relationship': self._analyze_price_volume_relationship(data),
                'session_analysis': self._analyze_session_volume(data)
            }
            
        except Exception as e:
            logger.error(f"Error in volume profile analysis: {e}")
            return self._empty_result()
    
    def _calculate_volume_profile(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Calculate volume profile across price levels"""
        prices = data['close'].values
        volumes = data['volume'].values
        
        # Create price bins
        price_min = np.min(prices)
        price_max = np.max(prices)
        price_range = price_max - price_min
        bin_size = price_range / self.price_bins
        
        # Group volumes by price bins
        price_bins = []
        volume_bins = []
        
        for i in range(self.price_bins):
            bin_price = price_min + (i * bin_size) + (bin_size / 2)
            
            # Find volume in this price range
            mask = (prices >= price_min + i * bin_size) & (prices < price_min + (i + 1) * bin_size)
            bin_volume = np.sum(volumes[mask])
            
            price_bins.append(bin_price)
            volume_bins.append(bin_volume)
        
        # Find high volume nodes (top 20%)
        volume_threshold = np.percentile(volume_bins, 80)
        high_volume_nodes = []
        
        for i, (price, volume) in enumerate(zip(price_bins, volume_bins)):
            if volume >= volume_threshold:
                high_volume_nodes.append({
                    'price': price,
                    'volume': volume,
                    'strength': volume / np.max(volume_bins) if np.max(volume_bins) > 0 else 0
                })
        
        return {
            'price_levels': price_bins,
            'volumes': volume_bins,
            'high_volume_nodes': high_volume_nodes,
            'total_volume': np.sum(volumes),
            'avg_volume': np.mean(volume_bins),
            'volume_range': [np.min(volumes), np.max(volumes)]
        }
    
    def _find_point_of_control(self, volume_profile: Dict, current_price: float) -> Dict:
        """Find Point of Control (price with highest volume)"""
        price_levels = np.array(volume_profile['price_levels'])
        volumes = np.array(volume_profile['volumes'])
        
        # Find POC
        poc_index = np.argmax(volumes)
        poc_price = price_levels[poc_index]
        poc_volume = volumes[poc_index]
        
        # Calculate distance from current price
        distance_from_current = abs(current_price - poc_price)
        distance_percent = (distance_from_current / current_price) * 100
        
        # POC hierarchy (find secondary and tertiary POCs)
        sorted_indices = np.argsort(volumes)[::-1]
        
        poc_hierarchy = []
        for i, idx in enumerate(sorted_indices[:3]):
            poc_hierarchy.append({
                'level': i + 1,
                'price': price_levels[idx],
                'volume': volumes[idx],
                'strength': volumes[idx] / poc_volume
            })
        
        return {
            'price': poc_price,
            'volume': poc_volume,
            'distance_from_current': distance_from_current,
            'distance_percent': distance_percent,
            'hierarchy': poc_hierarchy,
            'at_poc': distance_percent < 0.5,  # Within 0.5% of POC
            'value_area': self._calculate_value_area(price_levels, volumes, poc_volume)
        }
    
    def _calculate_vwap_analysis(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Calculate Volume Weighted Average Price"""
        close_prices = data['close'].values
        volumes = data['volume'].values
        
        # Calculate VWAP
        vwap = np.sum(close_prices * volumes) / np.sum(volumes)
        
        # Calculate standard deviation bands
        vwap_variance = np.sum(volumes * (close_prices - vwap) ** 2) / np.sum(volumes)
        vwap_std = np.sqrt(vwap_variance)
        
        vwap_upper = vwap + vwap_std
        vwap_lower = vwap - vwap_std
        
        # Position relative to VWAP
        vwap_position = (current_price - vwap) / vwap_std
        
        return {
            'vwap': vwap,
            'upper_band': vwap_upper,
            'lower_band': vwap_lower,
            'standard_deviation': vwap_std,
            'current_position': vwap_position,
            'above_vwap': current_price > vwap,
            'vwap_distance_percent': ((current_price - vwap) / vwap) * 100,
            'vwap_band_width': ((vwap_upper - vwap_lower) / vwap) * 100
        }
    
    def _analyze_order_flow(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Analyze buying vs selling pressure from price action and volume"""
        close_prices = data['close'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        volumes = data['volume'].values
        
        # Calculate buying/selling pressure indicators
        buy_pressure = []
        sell_pressure = []
        
        for i in range(1, len(close_prices)):
            price_change = close_prices[i] - close_prices[i-1]
            
            # Volume at bid/ask estimation
            if price_change > 0:  # Price went up => buying pressure
                buy_volume = volumes[i]
                sell_volume = volumes[i] * 0.3  # Lower estimate
            elif price_change < 0:  # Price went down => selling pressure
                buy_volume = volumes[i] * 0.3
                sell_volume = volumes[i]
            else:  # No change
                buy_volume = volumes[i] * 0.5
                sell_volume = volumes[i] * 0.5
            
            buy_pressure.append(buy_volume)
            sell_pressure.append(sell_volume)
        
        # Calculate cumulative pressure
        cumulative_buy = np.cumsum(buy_pressure)
        cumulative_sell = np.cumsum(sell_pressure)
        
        # Recent pressure (last 10 periods)
        recent_buy = np.sum(buy_pressure[-10:]) if len(buy_pressure) >= 10 else np.sum(buy_pressure)
        recent_sell = np.sum(sell_pressure[-10:]) if len(sell_pressure) >= 10 else np.sum(sell_pressure)
        
        # Calculate net pressure
        total_buy = np.sum(buy_pressure)
        total_sell = np.sum(sell_pressure)
        
        net_pressure = (total_buy - total_sell) / (total_buy + total_sell)
        recent_net_pressure = (recent_buy - recent_sell) / (recent_buy + recent_sell)
        
        # Determine dominant flow
        if recent_net_pressure > self.buy_sell_threshold:
            flow_direction = 'strong_buy'
        elif recent_net_pressure > 0:
            flow_direction = 'buy'
        elif recent_net_pressure < -self.buy_sell_threshold:
            flow_direction = 'strong_sell'
        else:
            flow_direction = 'sell'
        
        return {
            'buy_pressure': float(total_buy),
            'sell_pressure': float(total_sell),
            'net_pressure': float(net_pressure),
            'recent_net_pressure': float(recent_net_pressure),
            'flow_direction': flow_direction,
            'buy_sell_ratio': total_buy / total_sell if total_sell > 0 else float('inf'),
            'cumulative_buy': cumulative_buy[-1] if len(cumulative_buy) > 0 else 0,
            'cumulative_sell': cumulative_sell[-1] if len(cumulative_sell) > 0 else 0,
            'pressure_strength': abs(recent_net_pressure)
        }
    
    def _detect_absorption(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Detect absorption patterns (large volume with small price movement)"""
        close_prices = data['close'].values
        high_low_ranges = data['high'].values - data['low'].values
        volumes = data['volume'].values
        
        absorption_areas = []
        absorption_strength = []
        
        # Look for absorption in recent data
        for i in range(10, min(50, len(data))):
            recent_volume = np.mean(volumes[i:i+5])
            avg_range = np.mean(high_low_ranges[i:i+5])
            volume_ratio = volumes[i-1] / recent_volume if recent_volume > 0 else 1
            
            # High volume with small range = potential absorption
            if volume_ratio > 2.0 and avg_range < np.mean(high_low_ranges) * 0.5:
                absorption_areas.append({
                    'index': i,
                    'price': close_prices[i],
                    'volume_ratio': volume_ratio,
                    'strength': volume_ratio / (avg_range + 1e-6)
                })
        
        # Sort by strength
        absorption_areas.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'absorption_detected': len(absorption_areas) > 0,
            'absorption_areas': absorption_areas[:3],  # Top 3 areas
            'current_absorption': self._check_current_absorption(
                current_price, absorption_areas, close_prices[-1]
            ),
            'absorption_strength': np.max([area['strength'] for area in absorption_areas]) if absorption_areas else 0
        }
    
    def _detect_volume_climax(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Detect volume climax (unusually high volume)"""
        volumes = data['volume'].values
        close_prices = data['close'].values
        
        # Calculate average volume
        avg_volume = np.mean(volumes[:-10]) if len(volumes) > 10 else np.mean(volumes)
        
        # Find climax points
        climax_points = []
        for i, volume in enumerate(volumes):
            if volume > avg_volume * self.climax_multiplier:
                climax_points.append({
                    'index': i,
                    'volume': volume,
                    'volume_ratio': volume / avg_volume if avg_volume > 0 else 1,
                    'price': close_prices[i],
                    'type': 'buying' if i > 0 and close_prices[i] > close_prices[i-1] else 'selling'
                })
        
        # Check for current climax
        current_climax = volumes[-1] > avg_volume * self.climax_multiplier
        
        return {
            'climax_detected': len(climax_points) > 0,
            'climax_points': climax_points,
            'current_climax': current_climax,
            'climax_type': 'buying' if current_climax and len(close_prices) > 1 and close_prices[-1] > close_prices[-2] else 'selling',
            'climax_strength': volumes[-1] / avg_volume if current_climax and avg_volume > 0 else 0
        }
    
    def _simulate_market_depth(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Simulate market depth order book"""
        recent_data = data.tail(20)
        volumes = recent_data['volume'].values
        high_prices = recent_data['high'].values
        low_prices = recent_data['low'].values
        
        # Simulate bid/ask orders based on recent price action
        avg_volume = np.mean(volumes)
        
        # Generate simulated order book
        depth_levels = []
        
        for i in range(self.depth_levels):
            level = i + 1
            price_offset = level * 0.0001  # 1 pip per level
            
            # Buy orders (bid side)
            bid_volume = avg_volume * np.random.uniform(0.5, 1.5) / level
            buy_orders = {
                'level': level,
                'price': current_price - price_offset,
                'volume': bid_volume,
                'type': 'bid',
                'cumulative_volume': bid_volume
            }
            
            # Sell orders (ask side)
            ask_volume = avg_volume * np.random.uniform(0.5, 1.5) / level
            sell_orders = {
                'level': level,
                'price': current_price + price_offset,
                'volume': ask_volume,
                'type': 'ask',
                'cumulative_volume': ask_volume
            }
            
            depth_levels.extend([buy_orders, sell_orders])
        
        # Calculate cumulative volumes
        buy_orders = [o for o in depth_levels if o['type'] == 'bid']
        sell_orders = [o for o in depth_levels if o['type'] == 'ask']
        
        buy_orders.sort(key=lambda x: x['price'], reverse=True)
        sell_orders.sort(key=lambda x: x['price'])
        
        cumulative_buy = 0
        for order in buy_orders:
            cumulative_buy += order['volume']
            order['cumulative_volume'] = cumulative_buy
        
        cumulative_sell = 0
        for order in sell_orders:
            cumulative_sell += order['volume']
            order['cumulative_volume'] = cumulative_sell
        
        # Calculate spread and liquidity
        best_bid = buy_orders[0]['price'] if buy_orders else current_price - 0.0001
        best_ask = sell_orders[0]['price'] if sell_orders else current_price + 0.0001
        spread = best_ask - best_bid
        
        return {
            'buy_orders': buy_orders[:self.depth_levels],
            'sell_orders': sell_orders[:self.depth_levels],
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'spread_pips': spread * 10000,
            'total_bid_volume': cumulative_buy,
            'total_ask_volume': cumulative_sell,
            'liquidity_ratio': cumulative_buy / cumulative_sell if cumulative_sell > 0 else 1,
            'market_depth': self.depth_levels
        }
    
    def _analyze_volume_imbalance(self, data: pd.DataFrame, order_flow: Dict) -> Dict:
        """Analyze volume imbalance between buying and selling"""
        buy_pressure = order_flow.get('buy_pressure', 0)
        sell_pressure = order_flow.get('sell_pressure', 0)
        
        total_volume = buy_pressure + sell_pressure
        buy_percentage = (buy_pressure / total_volume) * 100 if total_volume > 0 else 50
        sell_percentage = (sell_pressure / total_volume) * 100 if total_volume > 0 else 50
        
        imbalance = buy_percentage - sell_percentage
        
        # Determine imbalance category
        if imbalance > 30:
            imbalance_type = 'strong_buy'
        elif imbalance > 10:
            imbalance_type = 'moderate_buy'
        elif imbalance > -10:
            imbalance_type = 'balanced'
        elif imbalance > -30:
            imbalance_type = 'moderate_sell'
        else:
            imbalance_type = 'strong_sell'
        
        return {
            'buy_percentage': buy_percentage,
            'sell_percentage': sell_percentage,
            'imbalance': imbalance,
            'imbalance_type': imbalance_type,
            'has_significant_imbalance': abs(imbalance) > 20
        }
    
    def _calculate_value_area(self, price_levels: np.ndarray, volumes: np.ndarray, poc_volume: float) -> Dict:
        """Calculate value area (68% of volume)"""
        # Sort by volume
        sorted_indices = np.argsort(volumes)[::-1]
        total_volume = np.sum(volumes)
        target_volume = total_volume * 0.68
        
        cumulative_volume = 0
        value_area_prices = []
        
        for idx in sorted_indices:
            cumulative_volume += volumes[idx]
            value_area_prices.append(price_levels[idx])
            
            if cumulative_volume >= target_volume:
                break
        
        if value_area_prices:
            value_area_high = np.max(value_area_prices)
            value_area_low = np.min(value_area_prices)
            value_area_range = value_area_high - value_area_low
        else:
            value_area_high = value_area_low = value_area_range = 0
        
        return {
            'high': value_area_high,
            'low': value_area_low,
            'range': value_area_range,
            'volume_covered': cumulative_volume,
            'percentage_covered': (cumulative_volume / total_volume) * 100 if total_volume > 0 else 0
        }
    
    def _check_current_absorption(self, current_price: float, absorption_areas: List, last_price: float) -> bool:
        """Check if current price is near an absorption area"""
        if not absorption_areas:
            return False
        
        closest_area = min(absorption_areas, key=lambda x: abs(x['price'] - current_price))
        distance = abs(closest_area['price'] - current_price)
        
        # Within 0.1% of absorption area
        return (distance / current_price) < 0.001
    
    def _analyze_price_volume_relationship(self, data: pd.DataFrame) -> Dict:
        """Analyze relationship between price and volume movements"""
        price_changes = data['close'].diff().values
        volumes = data['volume'].values
        
        # Remove NaN values
        mask = ~np.isnan(price_changes)
        price_changes = price_changes[mask]
        volumes = volumes[mask][1:]  # Skip first since price_change has one less element
        
        # Calculate correlation
        if len(volumes) > 1 and len(price_changes) > 1:
            correlation = np.corrcoef(np.abs(price_changes), volumes)[0, 1]
        else:
            correlation = 0
        
        # Identify divergences
        price_up_volume_down = 0
        price_down_volume_up = 0
        
        for i in range(1, len(price_changes)):
            if price_changes[i] > 0 and volumes[i] < volumes[i-1]:
                price_up_volume_down += 1
            elif price_changes[i] < 0 and volumes[i] > volumes[i-1]:
                price_down_volume_up += 1
        
        return {
            'correlation': correlation,
            'price_up_volume_down': price_up_volume_down,
            'price_down_volume_up': price_down_volume_up,
            'divergence_count': price_up_volume_down + price_down_volume_up,
            'volume_confirms_price': correlation > 0.5
        }
    
    def _analyze_session_volume(self, data: pd.DataFrame) -> Dict:
        """Analyze volume by trading session"""
        if not hasattr(data.index, 'hour'):
            return {'session_analysis_available': False}
        
        hours = data.index.hour
        volumes = data['volume'].values
        
        # Define sessions
        asian_hours = list(range(0, 8))
        london_hours = list(range(8, 16))
        ny_hours = list(range(13, 22))
        overlap_hours = list(range(13, 16))
        
        # Calculate volume by session
        asian_volume = np.sum(volumes[np.isin(hours, asian_hours)])
        london_volume = np.sum(volumes[np.isin(hours, london_hours)])
        ny_volume = np.sum(volumes[np.isin(hours, ny_hours)])
        overlap_volume = np.sum(volumes[np.isin(hours, overlap_hours)])
        
        total_volume = asian_volume + london_volume + ny_volume
        
        return {
            'session_analysis_available': True,
            'asian_volume': asian_volume,
            'london_volume': london_volume,
            'ny_volume': ny_volume,
            'overlap_volume': overlap_volume,
            'asian_percentage': (asian_volume / total_volume) * 100 if total_volume > 0 else 0,
            'london_percentage': (london_volume / total_volume) * 100 if total_volume > 0 else 0,
            'ny_percentage': (ny_volume / total_volume) * 100 if total_volume > 0 else 0,
            'highest_volume_session': max([
                ('asian', asian_volume), ('london', london_volume), 
                ('ny', ny_volume), ('overlap', overlap_volume)
            ], key=lambda x: x[1])[0]
        }
    
    def _generate_volume_signals(self, poc_analysis: Dict, vwap_analysis: Dict, 
                                order_flow: Dict, absorption: Dict, 
                                climax: Dict, market_depth: Dict) -> List[Dict]:
        """Generate trading signals based on volume analysis"""
        signals = []
        
        # POV-based signals
        if poc_analysis.get('at_poc', False):
            flow_direction = order_flow.get('flow_direction', 'neutral')
            if flow_direction in ['buy', 'strong_buy']:
                signals.append({
                    'type': 'BUY',
                    'source': 'volume_profile',
                    'confidence': 0.7,
                    'reason': 'Price at POC with buying pressure'
                })
            elif flow_direction in ['sell', 'strong_sell']:
                signals.append({
                    'type': 'SELL',
                    'source': 'volume_profile',
                    'confidence': 0.7,
                    'reason': 'Price at POC with selling pressure'
                })
        
        # VWAP-based signals
        vwap_position = vwap_analysis.get('current_position', 0)
        if abs(vwap_position) > 2:  # More than 2 standard deviations from VWAP
            if vwap_analysis.get('above_vwap', False) and order_flow.get('flow_direction') in ['sell', 'strong_sell']:
                signals.append({
                    'type': 'SELL',
                    'source': 'vwap_reversion',
                    'confidence': min(abs(vwap_position) / 4, 0.8),
                    'reason': 'Far above VWAP with selling pressure'
                })
            elif not vwap_analysis.get('above_vwap', True) and order_flow.get('flow_direction') in ['buy', 'strong_buy']:
                signals.append({
                    'type': 'BUY',
                    'source': 'vwap_reversion',
                    'confidence': min(abs(vwap_position) / 4, 0.8),
                    'reason': 'Far below VWAP with buying pressure'
                })
        
        # Order flow signals
        flow_strength = order_flow.get('pressure_strength', 0)
        if flow_strength > 0.8:
            flow_direction = order_flow.get('flow_direction')
            signals.append({
                'type': 'BUY' if 'buy' in flow_direction else 'SELL',
                'source': 'order_flow',
                'confidence': flow_strength * 0.6,
                'reason': f'Strong {flow_direction} pressure detected'
            })
        
        # Absorption signals
        if absorption.get('current_absorption', False):
            signals.append({
                'type': 'HOLD',
                'source': 'absorption',
                'confidence': 0.6,
                'reason': 'Absorption pattern detected - wait for breakout'
            })
        
        # Climax signals
        if climax.get('current_climax', False):
            signals.append({
                'type': 'HOLD',
                'source': 'volume_climax',
                'confidence': 0.7,
                'reason': 'Volume climax detected - potential exhaustion'
            })
        
        return signals
    
    def _calculate_volume_confidence(self, volume_profile: Dict, order_flow: Dict, 
                                    absorption: Dict, climax: Dict) -> float:
        """Calculate confidence score for volume analysis"""
        confidence = 0.0
        
        # Volume profile confidence
        if volume_profile.get('total_volume', 0) > self.min_volume_threshold:
            confidence += 0.2
        
        # Order flow confidence
        flow_strength = order_flow.get('pressure_strength', 0)
        confidence += flow_strength * 0.3
        
        # Absorption confidence
        if absorption.get('absorption_detected', False):
            confidence += 0.2
        
        # Climax confidence
        if climax.get('climax_detected', False):
            confidence += 0.1
        
        # Market depth confidence
        if volume_profile.get('total_volume', 0) > 0:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _empty_result(self) -> Dict:
        """Return empty analysis result"""
        return {
            'volume_profile': {},
            'point_of_control': {},
            'vwap': {},
            'order_flow': {},
            'absorption': {},
            'volume_climax': {},
            'market_depth': {},
            'volume_imbalance': {},
            'signals': [],
            'confidence_score': 0.0,
            'price_volume relationship': {},
            'session_analysis': {}
        }
