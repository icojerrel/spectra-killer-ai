"""
Technical Analysis Strategy - Enhanced with Luckshury's Methodology
Implements traditional technical indicators with advanced liquidity, volume, and order flow integration
"""

import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Dict, Optional, Tuple
import logging

# Import advanced analyzers
from ..advanced.liquidity_analyzer import LiquidityAnalyzer
try:
    from ..advanced.volume_profile import VolumeProfileAnalyzer
except ImportError:
    VolumeProfileAnalyzer = None
from ..advanced.session_analyzer import SessionAnalyzer
from ..advanced.order_flow_analyzer import OrderFlowAnalyzer

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Technical analysis with advanced liquidity, volume, and order flow integration"""
    
    def __init__(self, config: Dict):
        """
        Initialize technical analyzer
        
        Args:
            config: Technical analysis configuration
        """
        self.config = config
        
        # Traditional parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 65)
        self.rsi_oversold = config.get('rsi_oversold', 35)
        
        self.ema_short = config.get('ema_short', 9)
        self.ema_long = config.get('ema_long', 21)
        
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2)
        
        # Enhanced weights (Luckshury methodology)
        self.rsi_weight = config.get('rsi_weight', 0.25)
        self.ema_weight = config.get('ema_weight', 0.20)
        self.bb_weight = config.get('bb_weight', 0.15)
        self.liquidity_weight = config.get('liquidity_weight', 0.20)  # New
        self.volume_weight = config.get('volume_weight', 0.15)        # New
        self.session_weight = config.get('session_weight', 0.05)      # New
        
        # Initialize advanced analyzers
        self.liquidity_analyzer = LiquidityAnalyzer(config.get('liquidity', {}))
        self.volume_analyzer = VolumeProfileAnalyzer(config.get('volume_profile', {})) if VolumeProfileAnalyzer else None
        self.session_analyzer = SessionAnalyzer(config.get('session', {}))
        self.orderflow_analyzer = OrderFlowAnalyzer(config.get('order_flow', {}))
        
        # Feature integration settings
        self.enable_liquidity_analysis = config.get('enable_liquidity_analysis', True)
        self.enable_volume_profile = config.get('enable_volume_profile', True)
        self.enable_session_filtering = config.get('enable_session_filtering', True)
        self.enable_order_flow = config.get('enable_order_flow', True)
    
    async def analyze(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Enhanced analyze with liquidity, volume profile, and session integration
        
        Args:
            data: OHLCV data DataFrame with datetime index
            
        Returns:
            Enhanced signal dictionary or None if insufficient data
        """
        try:
            if len(data) < max(self.rsi_period, self.ema_long, self.bb_period):
                logger.warning("Insufficient data for technical analysis")
                return None
            
            close_prices = data['close']
            current_price = float(close_prices.iloc[-1])
            
            # Traditional indicators
            rsi = self._calculate_rsi(close_prices)
            ema_short = self._calculate_ema(close_prices, self.ema_short)
            ema_long = self._calculate_ema(close_prices, self.ema_long)
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close_prices)
            
            # Current values
            current_rsi = rsi.iloc[-1]
            current_ema_short = ema_short.iloc[-1]
            current_ema_long = ema_long.iloc[-1]
            current_bb_upper = bb_upper.iloc[-1]
            current_bb_lower = bb_lower.iloc[-1]
            
            # Traditional signals
            rsi_signal, rsi_confidence = self._analyze_rsi(current_rsi)
            ema_signal, ema_confidence = self._analyze_ema(ema_short, ema_long)
            bb_signal, bb_confidence = self._analyze_bollinger(current_price, current_bb_upper, current_bb_lower)
            
            # Enhanced analysis
            enhanced_signals = {}
            enhanced_confidences = {}
            
            # Liquidity analysis (Luckshury method)
            if self.enable_liquidity_analysis:
                liquidity_analysis = self.liquidity_analyzer.analyze(data, current_price)
                liquidity_signal, liquidity_confidence = self._analyze_liquidity_signals(liquidity_analysis)
                enhanced_signals['liquidity'] = liquidity_signal
                enhanced_confidences['liquidity'] = liquidity_confidence
            else:
                liquidity_analysis = None
                enhanced_signals['liquidity'] = 'HOLD'
                enhanced_confidences['liquidity'] = 0.0
            
            # Volume profile analysis
            if self.enable_volume_profile and self.volume_analyzer:
                volume_analysis = self.volume_analyzer.analyze(data, current_price)
                volume_signal, volume_confidence = self._analyze_volume_signals(volume_analysis)
                enhanced_signals['volume'] = volume_signal
                enhanced_confidences['volume'] = volume_confidence
            else:
                volume_analysis = None
                enhanced_signals['volume'] = 'HOLD'
                enhanced_confidences['volume'] = 0.0
            
            # Session analysis
            if self.enable_session_filtering:
                session_analysis = self.session_analyzer.analyze(data)
                session_multiplier = session_analysis['session_parameters']['signal_reduction_factor']
                session_recommendation = session_analysis['trading_recommendation']
            else:
                session_analysis = None
                session_multiplier = 1.0
                session_recommendation = {'should_trade': True, 'confidence': 1.0}
            
            # Order flow analysis
            if self.enable_order_flow:
                orderflow_analysis = self.orderflow_analyzer.analyze(data, current_price)
                orderflow_signal, orderflow_confidence = self._analyze_orderflow_signals(orderflow_analysis)
                enhanced_signals['orderflow'] = orderflow_signal
                enhanced_confidences['orderflow'] = orderflow_confidence
            else:
                orderflow_analysis = None
                enhanced_signals['orderflow'] = 'HOLD'
                enhanced_confidences['orderflow'] = 0.0
            
            # Enhanced signal combination with Luckshury weighting
            combined_signal, combined_confidence = self._combine_enhanced_signals(
                rsi_signal, rsi_confidence,
                ema_signal, ema_confidence,
                bb_signal, bb_confidence,
                enhanced_signals,
                enhanced_confidences
            )
            
            # Apply session filtering
            final_confidence = combined_confidence * session_multiplier
            if not session_recommendation['should_trade']:
                final_signal = 'HOLD'
                final_confidence = min(final_confidence, 0.3)
            else:
                final_signal = combined_signal
            
            return {
                'signal': final_signal,
                'confidence': final_confidence,
                'traditional_analysis': {
                    'rsi': {
                        'value': float(current_rsi),
                        'signal': rsi_signal,
                        'confidence': rsi_confidence
                    },
                    'ema': {
                        'short': float(current_ema_short),
                        'long': float(current_ema_long),
                        'signal': ema_signal,
                        'confidence': ema_confidence
                    },
                    'bollinger': {
                        'upper': float(current_bb_upper),
                        'middle': float(bb_middle.iloc[-1]),
                        'lower': float(current_bb_lower),
                        'signal': bb_signal,
                        'confidence': bb_confidence
                    }
                },
                'enhanced_analysis': {
                    'liquidity': liquidity_analysis,
                    'volume_profile': volume_analysis,
                    'session': session_analysis,
                    'order_flow': orderflow_analysis
                },
                'enhanced_signals': enhanced_signals,
                'session_filter_applied': self.enable_session_filtering,
                'metadata': {
                    'current_price': current_price,
                    'analysis_time': pd.Timestamp.now().isoformat(),
                    'methodology': 'Luckshury Enhanced',
                    'session_multiplier': session_multiplier,
                    'liquidity_boost': bool(enhanced_confidences.get('liquidity', 0) > 0.6),
                    'volume_confirmation': bool(enhanced_confidences.get('volume', 0) > 0.5)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced technical analysis: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate RSI using Wilder's smoothing"""
        if period is None:
            period = self.rsi_period
            
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Wilder's smoothing
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        # RSI calculation
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle_band = prices.rolling(window=self.bb_period).mean()
        rolling_std = prices.rolling(window=self.bb_period).std()
        
        upper_band = middle_band + (rolling_std * self.bb_std)
        lower_band = middle_band - (rolling_std * self.bb_std)
        
        return upper_band, middle_band, lower_band
    
    def _analyze_rsi(self, rsi_value: float) -> Tuple[str, float]:
        """Analyze RSI value and generate signal"""
        if rsi_value >= self.rsi_overbought + 5:
            return "SELL", 0.8
        elif rsi_value >= self.rsi_overbought:
            return "SELL", 0.6
        elif rsi_value <= self.rsi_oversold - 5:
            return "BUY", 0.8
        elif rsi_value <= self.rsi_oversold:
            return "BUY", 0.6
        else:
            return "HOLD", 0.3
    
    def _analyze_ema(self, ema_short: pd.Series, ema_long: pd.Series) -> Tuple[str, float]:
        """Analyze EMA crossover and generate signal"""
        if len(ema_short) < 2 or len(ema_long) < 2:
            return "HOLD", 0.3
        
        # Current and previous values
        current_short = ema_short.iloc[-1]
        current_long = ema_long.iloc[-1]
        prev_short = ema_short.iloc[-2]
        prev_long = ema_long.iloc[-2]
        
        # Check for crossover
        if (current_short > current_long) and (prev_short <= prev_long):
            return "BUY", 0.75  # Bullish crossover
        elif (current_short < current_long) and (prev_short >= prev_long):
            return "SELL", 0.75  # Bearish crossover
        elif current_short > current_long:
            return "BUY", 0.6  # Bullish momentum
        else:
            return "SELL", 0.6  # Bearish momentum
    
    def _analyze_bollinger(self, price: float, upper: float, lower: float) -> Tuple[str, float]:
        """Analyze Bollinger Bands position"""
        # Calculate position within bands
        band_width = upper - lower
        position = (price - lower) / band_width
        
        if position >= 0.95:
            return "SELL", 0.7  # Near upper band
        elif position <= 0.05:
            return "BUY", 0.7   # Near lower band
        elif position >= 0.8:
            return "SELL", 0.5  # Upper region
        elif position <= 0.2:
            return "BUY", 0.5   # Lower region
        else:
            return "HOLD", 0.3  # Middle region
    
    def _combine_signals(self, rsi_signal: str, rsi_conf: float,
                        ema_signal: str, ema_conf: float,
                        bb_signal: str, bb_conf: float) -> Tuple[str, float]:
        """Combine individual signals using weighted voting"""
        signals = [rsi_signal, ema_signal, bb_signal]
        confidences = [rsi_conf * self.rsi_weight, ema_conf * self.ema_weight, bb_conf * self.bb_weight]
        
        # Weighted voting
        buy_score = sum(conf for sig, conf in zip(signals, confidences) if sig == "BUY")
        sell_score = sum(conf for sig, conf in zip(signals, confidences) if sig == "SELL")
        hold_score = sum(conf for sig, conf in zip(signals, confidences) if sig == "HOLD")
        
        # Determine final signal
        if buy_score > sell_score and buy_score > hold_score:
            final_signal = "BUY"
            final_confidence = min(buy_score + hold_score * 0.5, 1.0)
        elif sell_score > buy_score and sell_score > hold_score:
            final_signal = "SELL"
            final_confidence = min(sell_score + hold_score * 0.5, 1.0)
        else:
            final_signal = "HOLD"
            final_confidence = buy_score + sell_score
        
        return final_signal, final_confidence
    
    def _analyze_liquidity_signals(self, liquidity_analysis: Dict) -> Tuple[str, float]:
        """Analyze liquidity analysis and generate signal"""
        if not liquidity_analysis or not liquidity_analysis.get('signals'):
            return 'HOLD', 0.0
        
        signals = liquidity_analysis['signals']
        if not signals:
            return 'HOLD', 0.0
        
        # Get highest confidence signal
        best_signal = max(signals, key=lambda x: x['confidence'])
        
        return best_signal['type'], best_signal['confidence']
    
    def _analyze_volume_signals(self, volume_analysis: Dict) -> Tuple[str, float]:
        """Analyze volume profile and generate signal"""
        if not volume_analysis or not volume_analysis.get('signals'):
            return 'HOLD', 0.0
        
        signals = volume_analysis['signals']
        if not signals:
            return 'HOLD', 0.0
        
        # Get highest confidence signal
        best_signal = max(signals, key=lambda x: x['confidence'])
        
        return best_signal['type'], best_signal['confidence']
    
    def _analyze_orderflow_signals(self, orderflow_analysis: Dict) -> Tuple[str, float]:
        """Analyze order flow and generate signal"""
        if not orderflow_analysis or not orderflow_analysis.get('signals'):
            return 'HOLD', 0.0
        
        signals = orderflow_analysis['signals']
        if not signals:
            return 'HOLD', 0.0
        
        # Get highest confidence signal
        best_signal = max(signals, key=lambda x: x['confidence'])
        
        return best_signal['type'], best_signal['confidence']
    
    def _combine_enhanced_signals(self, rsi_signal: str, rsi_conf: float,
                                 ema_signal: str, ema_conf: float,
                                 bb_signal: str, bb_conf: float,
                                 enhanced_signals: Dict, enhanced_confidences: Dict) -> Tuple[str, float]:
        """Combine all signals using Luckshury's enhanced weighting"""
        
        # Traditional signals
        traditional_signals = [rsi_signal, ema_signal, bb_signal]
        traditional_confidences = [
            rsi_conf * self.rsi_weight,
            ema_conf * self.ema_weight,
            bb_conf * self.bb_weight
        ]
        
        # Enhanced signals
        enhanced_signal_list = []
        enhanced_confidence_list = []
        
        if enhanced_signals.get('liquidity', 'HOLD') != 'HOLD':
            enhanced_signal_list.append(enhanced_signals['liquidity'])
            enhanced_confidence_list.append(enhanced_confidences['liquidity'] * self.liquidity_weight)
        
        if enhanced_signals.get('volume', 'HOLD') != 'HOLD':
            enhanced_signal_list.append(enhanced_signals['volume'])
            enhanced_confidence_list.append(enhanced_confidences['volume'] * self.volume_weight)
        
        if enhanced_signals.get('orderflow', 'HOLD') != 'HOLD':
            enhanced_signal_list.append(enhanced_signals['orderflow'])
            enhanced_confidence_list.append(enhanced_confidences['orderflow'] * self.session_weight)  # Using session_weight as placeholder
        
        # Combine all signals
        all_signals = traditional_signals + enhanced_signal_list
        all_confidences = traditional_confidences + enhanced_confidence_list
        
        # Weighted voting
        buy_score = sum(conf for sig, conf in zip(all_signals, all_confidences) if sig == "BUY")
        sell_score = sum(conf for sig, conf in zip(all_signals, all_confidences) if sig == "SELL")
        hold_score = sum(conf for sig, conf in zip(all_signals, all_confidences) if sig == "HOLD")
        
        # Luckshury bias towards liquidity and volume confirmation
        if enhanced_confidences.get('liquidity', 0) > 0.6:
            if enhanced_signals.get('liquidity') == "BUY":
                buy_score *= 1.2
            elif enhanced_signals.get('liquidity') == "SELL":
                sell_score *= 1.2
        
        if enhanced_confidences.get('volume', 0) > 0.5:
            if enhanced_signals.get('volume') == "BUY":
                buy_score *= 1.1
            elif enhanced_signals.get('volume') == "SELL":
                sell_score *= 1.1
        
        # Determine final signal
        if buy_score > sell_score and buy_score > hold_score:
            final_signal = "BUY"
            final_confidence = min(buy_score + hold_score * 0.3, 1.0)
        elif sell_score > buy_score and sell_score > hold_score:
            final_signal = "SELL"
            final_confidence = min(sell_score + hold_score * 0.3, 1.0)
        else:
            final_signal = "HOLD"
            final_confidence = buy_score + sell_score
        
        return final_signal, final_confidence
    
    def get_indicator_values(self, data: pd.DataFrame) -> Dict:
        """Get current indicator values for monitoring"""
        try:
            close_prices = data['close']
            
            rsi = self._calculate_rsi(close_prices)
            ema_short = self._calculate_ema(close_prices, self.ema_short)
            ema_long = self._calculate_ema(close_prices, self.ema_long)
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close_prices)
            
            return {
                'rsi': float(rsi.iloc[-1]),
                'ema_short': float(ema_short.iloc[-1]),
                'ema_long': float(ema_long.iloc[-1]),
                'bb_upper': float(bb_upper.iloc[-1]),
                'bb_middle': float(bb_middle.iloc[-1]),
                'bb_lower': float(bb_lower.iloc[-1]),
                'price': float(close_prices.iloc[-1]),
            }
        except Exception as e:
            logger.error(f"Error getting indicator values: {e}")
            return {}
