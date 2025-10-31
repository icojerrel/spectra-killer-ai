"""
Order Flow Analyzer
Implements advanced order flow analysis, market microstructure, and absorption patterns
Based on Luckshury's order flow methodology
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class OrderFlowAnalyzer:
    """Advanced order flow and market microstructure analysis"""
    
    def __init__(self, config: Dict):
        """
        Initialize order flow analyzer
        
        Args:
            config: Order flow analysis configuration
        """
        self.config = config
        
        # Market depth parameters
        self.depth_levels = config.get('depth_levels', 10)
        self.min_order_size = config.get('min_order_size', 0.01)
        self.max_order_size = config.get('max_order_size', 10.0)
        
        # Order flow thresholds
        self.aggressive_threshold = config.get('aggressive_threshold', 0.7)
        self.absorption_threshold = config.get('absorption_threshold', 2.0)
        self.exhaustion_threshold = config.get('exhaustion_threshold', 0.8)
        
        # Market microstructure parameters
        self.tick_size = config.get('tick_size', 0.0001)  # Default for forex
        self.lot_size = config.get('lot_size', 100000)
        self.commission_per_lot = config.get('commission_per_lot', 7.0)
        
        # Time windows for analysis
        self.microstructure_window = config.get('microstructure_window', 5)
        self.imbalance_window = config.get('imbalance_window', 10)
        self.absorption_window = config.get('absorption_window', 20)
        
    def analyze(self, data: pd.DataFrame, current_price: float, 
                market_data: Optional[Dict] = None) -> Dict:
        """
        Perform comprehensive order flow analysis
        
        Args:
            data: OHLCV data with volume
            current_price: Current market price
            market_data: Additional market data (bid/ask, depth, etc.)
            
        Returns:
            Order flow analysis results
        """
        try:
            if len(data) < self.microstructure_window * 3:
                logger.warning("Insufficient data for order flow analysis")
                return self._empty_result()
            
            # Market microstructure analysis
            microstructure = self._analyze_market_microstructure(data, current_price)
            
            # Volume flow analysis
            volume_flow = self._analyze_volume_flow(data, current_price)
            
            # Order imbalance detection
            imbalance = self._detect_order_imbalance(data, current_price)
            
            # Absorption patterns
            absorption = self._detect_absorption_patterns(data, current_price)
            
            # Exhaustion patterns
            exhaustion = self._detect_exhaustion_patterns(data, current_price)
            
            # Market pressure analysis
            pressure = self._analyze_market_pressure(data, current_price)
            
            # Liquidity analysis
            liquidity = self._analyze_liquidity_conditions(data, current_price)
            
            # Generate order flow signals
            signals = self._generate_order_flow_signals(
                microstructure, volume_flow, imbalance, absorption, exhaustion, pressure
            )
            
            # Calculate order flow confidence
            confidence = self._calculate_order_flow_confidence(
                microstructure, volume_flow, imbalance, absorption, exhaustion
            )
            
            return {
                'market_microstructure': microstructure,
                'volume_flow': volume_flow,
                'order_imbalance': imbalance,
                'absorption_patterns': absorption,
                'exhaustion_patterns': exhaustion,
                'market_pressure': pressure,
                'liquidity_analysis': liquidity,
                'signals': signals,
                'confidence_score': confidence,
                'trade_execution_recommendations': self._get_execution_recommendations(
                    signals, microstructure, liquidity
                ),
                'risk_assessment': self._assess_order_flow_risks(
                    microstructure, imbalance, exhaustion, liquidity
                )
            }
            
        except Exception as e:
            logger.error(f"Error in order flow analysis: {e}")
            return self._empty_result()
    
    def _analyze_market_microstructure(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Analyze market microstructure from price action"""
        close_prices = data['close'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        volumes = data['volume'].values
        
        # Calculate price changes and volatility
        price_changes = np.diff(close_prices)
        true_ranges = high_prices - low_prices
        
        # Microstructure indicators
        price_efficiency = self._calculate_price_efficiency(close_prices)
        volatility_ratio = self._calculate_volatility_ratio(price_changes, true_ranges)
        
        # Spread estimation (simplified)
        avg_true_range = np.mean(true_rates := true_rates if 'true_rates' in locals() else true_ranges[-20:])
        estimated_spread = avg_true_range * 0.1  # Rough estimation
        
        # Tick analysis
        tick_dx = self._calculate_tick_dx(close_prices)
        tick_imbalance = self._calculate_tick_imbalance(close_prices, volumes)
        
        # Price momentum
        momentum = self._calculate_price_momentum(close_prices)
        
        return {
            'price_efficiency': price_efficiency,
            'volatility_ratio': volatility_ratio,
            'estimated_spread': estimated_spread,
            'spread_pips': estimated_spread * 10000,
            'tick_dx': tick_dx,
            'tick_imbalance': tick_imbalance,
            'price_momentum': momentum,
            'market_regime': self._classify_market_regime(price_efficiency, volatility_ratio),
            'microstructure_score': self._calculate_microstructure_score(ticket_dx, price_efficiency)
        }
    
    def _analyze_volume_flow(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Analyze volume flow patterns"""
        prices = data['close'].values
        volumes = data['volume'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        
        # Estimate buying vs selling volume from price action
        buying_volume = []
        selling_volume = []
        
        for i in range(1, len(prices)):
            price_change = prices[i] - prices[i-1]
            price_range = high_prices[i] - low_prices[i]
            
            if price_change > 0:  # Up move
                close_position = (prices[i] - low_prices[i]) / price_range
                buy_vol = volumes[i] * (0.5 + close_position * 0.5)
                sell_vol = volumes[i] * (0.5 - close_position * 0.3)
            elif price_change < 0:  # Down move
                close_position = (high_prices[i] - prices[i]) / price_range
                buy_vol = volumes[i] * (0.5 - close_position * 0.3)
                sell_vol = volumes[i] * (0.5 + close_position * 0.5)
            else:  # No change
                buy_vol = volumes[i] * 0.5
                sell_vol = volumes[i] * 0.5
            
            buying_volume.append(buy_vol)
            selling_volume.append(sell_vol)
        
        buying_volume = np.array(buying_volume)
        selling_volume = np.array(selling_volume)
        
        # Flow metrics
        net_flow = buying_volume - selling_volume
        flow_ratio = np.sum(buying_volume) / np.sum(selling_volume) if np.sum(selling_volume) > 0 else float('inf')
        
        # Recent flow (last 10 bars)
        recent_buy = np.sum(buying_volume[-10:])
        recent_sell = np.sum(selling_volume[-10:])
        recent_net_flow = recent_buy - recent_sell
        recent_flow_ratio = recent_buy / recent_sell if recent_sell > 0 else float('inf')
        
        # Flow persistence
        flow_directions = np.sign(net_flow)
        persistence = self._calculate_persistence(flow_directions)
        
        return {
            'total_buying_volume': float(np.sum(buying_volume)),
            'total_selling_volume': float(np.sum(selling_volume)),
            'net_flow': float(np.sum(net_flow)),
            'flow_ratio': flow_ratio,
            'recent_buy': float(recent_buy),
            'recent_sell': float(recent_sell),
            'recent_net_flow': float(recent_net_flow),
            'recent_flow_ratio': recent_flow_ratio,
            'flow_persistence': persistence,
            'flow_direction': 'buying' if flow_ratio > 1.2 else 'selling' if flow_ratio < 0.8 else 'neutral',
            'flow_strength': abs(np.log(flow_ratio)) if flow_ratio != float('inf') else 3.0,
            'accumulation_distribution': self._ calculate_accumulation_distribution(prices, volumes)
        }
    
    def _detect_order_imbalance(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Detect order imbalance indicators"""
        prices = data['close'].values
        volumes = data['volume'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        
        # Calculate order imbalance using volume-weighted price action
        price_moves = np.diff(prices)
        price_ranges = high_prices[1:] - low_prices[1:]
        
        # Imbalance calculation
        positive_pressure = []
        negative_pressure = []
        
        for i in range(len(price_moves)):
            move = price_moves[i]
            range_val = price_ranges[i]
            
            if move > 0:  # Buying pressure
                pressure_strength = (move / range_val) * volumes[i+1] if range_val > 0 else 0
                positive_pressure.append(pressure_strength)
                negative_pressure.append(0)
            elif move < 0:  # Selling pressure
                pressure_strength = abs(move / range_val) * volumes[i+1] if range_val > 0 else 0
                positive_pressure.append(0)
                negative_pressure.append(pressure_strength)
            else:  # No move
                positive_pressure.append(0)
                negative_pressure.append(0)
        
        total_positive = np.sum(positive_pressure)
        total_negative = np.sum(negative_pressure)
        
        # Calculate imbalance metrics
        imbalance_ratio = total_positive / total_negative if total_negative > 0 else float('inf')
        imbalance_score = (total_positive - total_negative) / (total_positive + total_negative)
        
        # Recent imbalance (last 10 bars)
        recent_pos = np.sum(positive_pressure[-10:])
        recent_neg = np.sum(negative_pressure[-10:])
        recent_imbalance = (recent_pos - recent_neg) / (recent_pos + recent_neg)
        
        # Imbalance classification
        if imbalance_score > 0.3:
            imbalance_type = 'strong_buying'
        elif imbalance_score > 0.1:
            imbalance_type = 'moderate_buying'
        elif imbalance_score > -0.1:
            imbalance_type = 'balanced'
        elif imbalance_score > -0.3:
            imbalance_type = 'moderate_selling'
        else:
            imbalance_type = 'strong_selling'
        
        # Imbalance persistence
        recent_directions = np.sign([positive_pressure[i] - negative_pressure[i] for i in range(min(20, len(positive_pressure)))])
        persistence = self._calculate_persistence(recent_directions)
        
        return {
            'imbalance_score': imbalance_score,
            'imbalance_ratio': imbalance_ratio,
            'imbalance_type': imbalance_type,
            'recent_imbalance': recent_imbalance,
            'buying_pressure': float(total_positive),
            'selling_pressure': float(total_negative),
            'imbalance_persistence': persistence,
            'significant_imbalance': abs(imbalance_score) > 0.2,
            'imbalance_strength': abs(imbalance_score),
            'market_side': 'buy' if imbalance_score > 0 else 'sell'
        }
    
    def _detect_absorption_patterns(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Detect absorption patterns (large volume with small price movement)"""
        prices = data['close'].values
        volumes = data['volume'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        
        absorption_zones = []
        
        # Look for absorption in recent data
        for i in range(self.absorption_window, min(len(prices), 50)):
            window_prices = prices[i-self.absorption_window:i]
            window_volumes = volumes[i-self.absorption_window:i]
            window_ranges = high_prices[i-self.absorption_window:i] - low_prices[i-self.absorption_window:i]
            
            # Calculate metrics
            avg_volume = np.mean(window_volumes)
            price_range = max(window_prices) - min(window_prices)
            avg_range = np.mean(window_ranges)
            
            # Absorption criteria: high volume, low range
            volume_spike = window_volumes[-1] / avg_volume if avg_volume > 0 else 1
            range_compression = avg_range / np.mean(window_ranges[:-10:-10:]) if len(window_ranges) > 10 else 1
            
            if volume_spike > self.absorption_threshold and range_compression < 0.5:
                absorption_zones.append({
                    'index': i,
                    'price': prices[i],
                    'volume_multiplier': volume_spike,
                    'range_compression': range_compression,
                    'absorption_strength': volume_spike / (range_compression + 1e-6),
                    'direction': 'selling' if i < len(prices) and prices[i] < prices[i-1] else 'buying'
                })
        
        # Check for current absorption
        current_absorption = self._check_current_absorption(data.tail(self.absorption_window))
        
        return {
            'absorption_detected': len(absorption_zones) > 0,
            'absorption_zones': absorption_zones[:3],  # Top 3 zones
            'current_absorption': current_absorption,
            'max_absorption_strength': max([z['absorption_strength'] for z in absorption_zones]) if absorption_zones else 0,
            'absorption_count': len(absorption_zones),
            'absorption_frequency': len(absorption_zones) / (len(prices) / self.absorption_window)
        }
    
    def _detect_exhaustion_patterns(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Detect exhaustion patterns (climax, capitulation)"""
        prices = data['close'].values
        volumes = data['volume'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        
        price_changes = np.diff(prices)
        
        # Climax detection (extreme volume with large price move)
        avg_volume = np.mean(volumes[:-10]) if len(volumes) > 10 else np.mean(volumes)
        avg_change = np.mean(np.abs(price_changes[:-10])) if len(price_changes) > 10 else 0
        
        climax_points = []
        
        for i in range(10, len(prices)):
            current_volume = volumes[i]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            price_magnitude = abs(prices[i] - prices[i-1]) / avg_change if avg_change > 0 else 0
            
            # Climax criteria
            if volume_ratio > self.exhaustion_threshold and price_magnitude > 2:
                climax_type = 'buying_climax' if prices[i] > prices[i-1] else 'selling_climax'
                climax_points.append({
                    'index': i,
                    'price': prices[i],
                    'volume_ratio': volume_ratio,
                    'price_magnitude': price_magnitude,
                    'climax_type': climax_type,
                    'strength': volume_ratio * price_magnitude
                })
        
        # Capitulation detection (accelerating selling)
        recent_changes = price_changes[-20:]
        if len(recent_changes) >= 10:
            acceleration = np.std(recent_changes[-10:]) - np.std(recent_changes[-20:-10])
            capitulation = acceleration > 0 and np.mean(recent_changes[-5:]) < 0
        else:
            acceleration = 0
            capitulation = False
        
        # Exhaustion indicators
        divergence_detected = self._detect_volume_price_divergence(prices, volumes)
        
        return {
            'climax_detected': len(climax_points) > 0,
            'climax_points': climax_points[-3:],  # Last 3 points
            'capitulation_detected': capitulation,
            'price_acceleration': acceleration,
            'divergence_detected': divergence_detected,
            'exhaustion_strength': max([c['strength'] for c in climax_points]) if climax_points else 0,
            'current_exhaustion': self._check_current_exhaustion(prices[-1], volumes[-1], avg_volume)
        }
    
    def _analyze_market_pressure(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Analyze overall market pressure and momentum"""
        prices = data['close'].values
        volumes = data['volume'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        
        # Calculate pressure indicators
        price_momentum = self._calculate_price_momentum(prices)
        volume_momentum = self._calculate_volume_momentum(volumes)
        
        # Pressure zones
        resistance_pressure = self._calculate_resistance_pressure(prices, high_prices)
        support_pressure = self._calculate_support_pressure(prices, low_prices)
        
        # Breakout/breakdown potential
        breakout_potential = self._calculate_breakout_potential(prices, volumes)
        
        # Market participation
        participation_rate = self._calculate_market_participation(volumes)
        
        return {
            'price_momentum': price_momentum,
            'volume_momentum': volume_momentum,
            'resistance_pressure': resistance_pressure,
            'support_pressure': support_pressure,
            'breakout_potential': breakout_potential,
            'market_participation': participation_rate,
            'overall_pressure': self._calculate_overall_pressure(
                price_momentum, volume_momentum, breakout_potential
            ),
            'pressure_direction': self._determine_pressure_direction(prices, volumes),
            'pressure_intensity': self._calculate_pressure_intensity(prices, volumes)
        }
    
    def _analyze_liquidity_conditions(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Analyze current liquidity conditions"""
        volumes = data['volume'].values
        prices = data['close'].values
        
        # Volume volatility (liquidity proxy)
        volume_volatility = np.std(volumes[-20:]) / np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
        
        # Price stability
        price_stability = 1 / (1 + np.std(prices[-20:])) if len(prices) >= 20 else 1
        
        # Market depth estimation
        estimated_depth = self._estimate_market_depth(volumes)
        
        # Liquidity score
        liquidity_score = self._calculate_liquidity_score(
            np.mean(volumes[-20:]), volume_volatility, price_stability, estimated_depth
        )
        
        return {
            'volume_volatility': volume_volatility,
            'price_stability': price_stability,
            'estimated_depth': estimated_depth,
            'liquidity_score': liquidity_score,
            'liquidity_level': self._classify_liquidity_level(liquidity_score),
            'execution_quality': self._estimate_execution_quality(liquidity_score, volume_volatility),
            'slippage_estimate': self._estimate_slippage(liquidity_score, volume_volatility)
        }
    
    def _calculate_price_efficiency(self, prices: np.ndarray) -> float:
        """Calculate price efficiency (how much price moves in one direction)"""
        price_changes = np.diff(prices)
        total_movement = np.sum(np.abs(price_changes))
        net_movement = abs(prices[-1] - prices[0])
        
        return net_movement / total_movement if total_movement > 0 else 1
    
    def _calculate_volatility_ratio(self, price_changes: np.ndarray, true_ranges: np.ndarray) -> float:
        """Calculate volatility ratio"""
        change_volatility = np.std(price_changes) if len(price_changes) > 1 else 0
        range_volatility = np.std(true_ranges) if len(true_ranges) > 1 else 0
        
        return change_volatility / range_volatility if range_volatility > 0 else 1
    
    def _calculate_tick_dx(self, prices: np.ndarray) -> float:
        """Calculate directional movement index-like indicator"""
        if len(prices) < 14:
            return 0
        
        up_moves = []
        down_moves = []
        
        for i in range(1, len(prices)):
            up = prices[i] - prices[i-1] if prices[i] > prices[i-1] else 0
            down = prices[i-1] - prices[i] if prices[i] < prices[i-1] else 0
            up_moves.append(up)
            down_moves.append(down)
        
        avg_up = np.mean(up_moves[-14:]) if len(up_moves) >= 14 else np.mean(up_moves)
        avg_down = np.mean(down_moves[-14:]) if len(down_moves) >= 14 else np.mean(down_moves)
        
        sum_moves = avg_up + avg_down
        dx = (abs(avg_up - avg_down) / sum_moves) * 100 if sum_moves > 0 else 0
        
        return dx
    
    def _calculate_tick_imbalance(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate tick imbalance from price and volume"""
        if len(prices) < 2 or len(volumes) < 2:
            return 0
        
        price_changes = np.diff(prices)
        volume_changes = np.diff(volumes)
        
        # Buy ticks (price up with volume increase)
        buy_ticks = np.sum((price_changes > 0) & (volume_changes > 0))
        
        # Sell ticks (price down with volume increase)
        sell_ticks = np.sum((price_changes < 0) & (volume_changes > 0))
        
        total_ticks = buy_ticks + sell_ticks
        
        return (buy_ticks - sell_ticks) / total_ticks if total_ticks > 0 else 0
    
    def _calculate_price_momentum(self, prices: np.ndarray) -> Dict:
        """Calculate price momentum indicators"""
        if len(prices) < 10:
            return {'short_momentum': 0, 'medium_momentum': 0, 'long_momentum': 0}
        
        short_momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        medium_momentum = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        long_momentum = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
        
        return {
            'short_momentum': short_momentum,
            'medium_momentum': medium_momentum,
            'long_momentum': long_momentum,
            'momentum_strength': max(abs(short_momentum), abs(medium_momentum), abs(long_momentum))
        }
    
    def _calculate_volume_momentum(self, volumes: np.ndarray) -> Dict:
        """Calculate volume momentum"""
        if len(volumes) < 10:
            return {'volume_trend': 0, 'volume_acceleration': 0}
        
        recent_avg = np.mean(volumes[-5:])
        historical_avg = np.mean(volumes[-20:-5]) if len(volumes) >= 20 else np.mean(volumes[:-5])
        
        volume_trend = (recent_avg - historical_avg) / historical_avg if historical_avg > 0 else 0
        
        # Volume acceleration
        very_recent_avg = np.mean(volumes[-3:]) if len(volumes) >= 3 else recent_avg
        volume_acceleration = (very_recent_avg - recent_avg) / recent_avg if recent_avg > 0 else 0
        
        return {
            'volume_trend': volume_trend,
            'volume_acceleration': volume_acceleration
        }
    
    def _calculate_persistence(self, directions: np.ndarray) -> float:
        """Calculate persistence of directional movement"""
        if len(directions) < 2:
            return 0
        
        consecutive_same = 0
        for i in range(1, len(directions)):
            if directions[i] == directions[i-1]:
                consecutive_same += 1
        
        return consecutive_same / (len(directions) - 1)
    
    def _generate_order_flow_signals(self, microstructure: Dict, volume_flow: Dict,
                                   imbalance: Dict, absorption: Dict, 
                                   exhaustion: Dict, pressure: Dict) -> List[Dict]:
        """Generate trading signals from order flow analysis"""
        signals = []
        
        # Strong imbalance signals
        if imbalance['significant_imbalance'] and imbalance['imbalance_strength'] > 0.4:
            signals.append({
                'type': 'BUY' if imbalance['market_side'] == 'buy' else 'SELL',
                'source': 'order_imbalance',
                'confidence': imbalance['imbalance_strength'] * 0.8,
                'reason': f"Strong {imbalance['imbalance_type']} pressure detected"
            })
        
        # Absorption signals
        if absorption['current_absorption']['detected']:
            signals.append({
                'type': 'HOLD',
                'source': 'absorption',
                'confidence': 0.7,
                'reason': 'Absorption pattern detected - wait for breakout'
            })
        
        # Exhaustion signals
        if exhaustion['climax_detected']:
            signals.append({
                'type': 'REVERSE_POSITION',
                'source': 'exhaustion',
                'confidence': min(exhaustion['exhaustion_strength'] / 5, 0.8),
                'reason': 'Volume climax detected - potential reversal'
            })
        
        # Pressure signals
        if pressure['breakout_potential'] > 0.7:
            signals.append({
                'type': 'PREPARE_BREAKOUT',
                'source': 'market_pressure',
                'confidence': pressure['breakout_potential'] * 0.6,
                'reason': 'High breakout potential detected'
            })
        
        return signals
    
    def _calculate_order_flow_confidence(self, microstructure: Dict, volume_flow: Dict,
                                       imbalance: Dict, absorption: Dict, exhaustion: Dict) -> float:
        """Calculate confidence score for order flow analysis"""
        confidence = 0.0
        
        # Microstructure confidence
        if microstructure.get('price_efficiency', 0) > 0.7:
            confidence += 0.2
        
        # Volume flow confidence
        if volume_flow.get('flow_strength', 0) > 1.0:
            confidence += 0.3
        
        # Imbalance confidence
        if imbalance.get('significant_imbalance', False):
            confidence += 0.3
        
        # Absorption confidence
        if absorption.get('absorption_detected', False):
            confidence += 0.1
        
        # Exhaustion confidence
        if exhaustion.get('climax_detected', False):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _empty_result(self) -> Dict:
        """Return empty analysis result"""
        return {
            'market_microstructure': {},
            'volume_flow': {},
            'order_imbalance': {},
            'absorption_patterns': {},
            'exhaustion_patterns': {},
            'market_pressure': {},
            'liquidity_analysis': {},
            'signals': [],
            'confidence_score': 0.0,
            'trade_execution_recommendations': {},
            'risk_assessment': {}
        }
