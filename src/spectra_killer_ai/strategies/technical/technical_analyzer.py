"""
Technical Analysis Strategy
Implements traditional technical indicators and signal generation
"""

import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Technical analysis strategy with classic indicators"""
    
    def __init__(self, config: Dict):
        """
        Initialize technical analyzer
        
        Args:
            config: Technical analysis configuration
        """
        self.config = config
        
        # RSI parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 65)
        self.rsi_oversold = config.get('rsi_oversold', 35)
        
        # EMA parameters
        self.ema_short = config.get('ema_short', 9)
        self.ema_long = config.get('ema_long', 21)
        
        # Bollinger Bands parameters
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2)
        
        # Signal weights
        self.rsi_weight = config.get('rsi_weight', 0.4)
        self.ema_weight = config.get('ema_weight', 0.4)
        self.bb_weight = config.get('bb_weight', 0.2)
    
    async def analyze(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Analyze market data and generate signals
        
        Args:
            data: OHLCV data DataFrame
            
        Returns:
            Signal dictionary or None if insufficient data
        """
        try:
            if len(data) < max(self.rsi_period, self.ema_long, self.bb_period):
                logger.warning("Insufficient data for technical analysis")
                return None
            
            close_prices = data['close']
            
            # Calculate indicators
            rsi = self._calculate_rsi(close_prices)
            ema_short = self._calculate_ema(close_prices, self.ema_short)
            ema_long = self._calculate_ema(close_prices, self.ema_long)
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close_prices)
            
            # Get current values
            current_rsi = rsi.iloc[-1]
            current_ema_short = ema_short.iloc[-1]
            current_ema_long = ema_long.iloc[-1]
            current_price = close_prices.iloc[-1]
            current_bb_upper = bb_upper.iloc[-1]
            current_bb_lower = bb_lower.iloc[-1]
            
            # Generate individual signals
            rsi_signal, rsi_confidence = self._analyze_rsi(current_rsi)
            ema_signal, ema_confidence = self._analyze_ema(ema_short, ema_long)
            bb_signal, bb_confidence = self._analyze_bollinger(current_price, current_bb_upper, current_bb_lower)
            
            # Combine signals
            combined_signal, combined_confidence = self._combine_signals(
                rsi_signal, rsi_confidence,
                ema_signal, ema_confidence,
                bb_signal, bb_confidence
            )
            
            return {
                'signal': combined_signal,
                'confidence': combined_confidence,
                'indicators': {
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
                'metadata': {
                    'current_price': float(current_price),
                    'analysis_time': pd.Timestamp.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
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
