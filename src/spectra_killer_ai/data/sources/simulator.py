"""
XAUUSD Market Simulator
Realistic market data generation for XAUUSD testing and backtesting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class XAUUSDSimulator:
    """Professional XAUUSD market simulator with realistic price dynamics"""
    
    def __init__(self, config: Dict):
        """
        Initialize XAUUSD simulator
        
        Args:
            config: Simulator configuration
        """
        self.config = config
        
        # Base parameters for XAUUSD
        self.base_price = config.get('base_price', 2050.0)
        self.base_volatility = config.get('volatility', 0.002)  # 0.2% per tick
        self.trend_bias = config.get('trend_bias', 0.0001)
        self.mean_reversion_strength = config.get('mean_reversion_strength', 0.1)
        
        # Market session parameters (gold trades 24/5 with varying volatility)
        self.sessions = {
            'sydney': {'hours': range(22, 8), 'vol_mult': 0.7, 'volume_mult': 0.6},
            'tokyo': {'hours': range(23, 8), 'vol_mult': 0.8, 'volume_mult': 0.7},
            'london': {'hours': range(8, 16), 'vol_mult': 1.3, 'volume_mult': 1.4},
            'ny_session': {'hours': range(13, 22), 'vol_mult': 1.4, 'volume_mult': 1.5}
        }
        
        # Market microstructure
        self.tick_size = 0.01  # Minimum price movement for gold
        self.spread = 0.2     # Typical spread in points
        
        # Random seed for reproducibility
        self.seed = config.get('seed', 42)
        np.random.seed(self.seed)
        
        logger.info(f"XAUUSDSimulator initialized with base_price: ${self.base_price}")
    
    def generate_data(self, timeframe: str = '5M', periods: int = 288,
                     start_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Generate realistic XAUUSD market data
        
        Args:
            timeframe: Timeframe ('1M', '5M', '15M', '1H', '4H', '1D')
            periods: Number of periods to generate
            start_time: Start time for data generation
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert timeframe to frequency
            freq_map = {
                '1M': '1min', '5M': '5min', '15M': '15min',
                '1H': '1h', '4H': '4h', '1D': '1D'
            }
            freq = freq_map.get(timeframe, '5min')
            
            # Generate time index
            if start_time is None:
                start_time = datetime.now() - timedelta(minutes=5 * periods)
            
            date_range = pd.date_range(start=start_time, periods=periods, freq=freq)
            
            # Generate price data using advanced market model
            prices = self._generate_price_series(date_range, freq)
            
            # Generate OHLCV data
            ohlc_data = self._generate_ohlcv_data(date_range, prices, freq)
            
            # Create DataFrame
            df = pd.DataFrame(ohlc_data)
            df.set_index('timestamp', inplace=True)
            
            # Validate generated data
            self._validate_generated_data(df)
            
            logger.debug(f"Generated {len(df)} candles for {timeframe} timeframe")
            return df
            
        except Exception as e:
            logger.error(f"Error generating market data: {e}")
            # Return minimal valid data as fallback
            return self._create_fallback_data(periods)
    
    def _generate_price_series(self, date_range: pd.DatetimeIndex, freq: str) -> np.ndarray:
        """Generate realistic price series using market microstructure model"""
        num_periods = len(date_range)
        prices = np.zeros(num_periods)
        prices[0] = self.base_price
        
        for i in range(1, num_periods):
            timestamp = date_range[i]
            
            # Get session-specific parameters
            session_params = self._get_session_parameters(timestamp.hour)
            
            # Calculate dynamic volatility
            current_volatility = self.base_volatility * session_params['vol_mult']
            
            # Add intraday volatility pattern
            hour_factor = self._get_hour_factor(timestamp.hour)
            current_volatility *= hour_factor
            
            # Mean reversion component
            if i >= 20:
                short_ma = np.mean(prices[max(0, i-20):i])
                mean_reversion_force = self.mean_reversion_strength * (short_ma - prices[i-1]) / prices[i-1]
            else:
                mean_reversion_force = 0
            
            # Trend component
            trend_component = self.trend_bias * session_params['vol_mult']
            
            # Market microstructure noise
            microstructure_noise = np.random.normal(0, current_volatility * 0.1)
            
            # Random walk component
            random_component = np.random.normal(0, current_volatility)
            
            # Jump component (rare price jumps)
            jump_component = 0
            if np.random.random() < 0.001:  # 0.1% chance of jump
                jump_size = np.random.choice([-1, 1]) * np.random.uniform(0.005, 0.02)
                jump_component = jump_size
            
            # Calculate price change
            price_change = (
                mean_reversion_force + 
                trend_component + 
                random_component + 
                microstructure_noise + 
                jump_component
            )
            
            # Apply price change
            new_price = prices[i-1] * (1 + price_change)
            
            # Ensure price stays reasonable
            if new_price < self.base_price * 0.5 or new_price > self.base_price * 2:
                new_price = prices[i-1]  # Reject extreme moves
            
            prices[i] = new_price
        
        return prices
    
    def _get_session_parameters(self, hour: int) -> Dict:
        """Get session-specific market parameters"""
        for session, params in self.sessions.items():
            if hour in params['hours']:
                return {
                    'vol_mult': params['vol_mult'],
                    'volume_mult': params['volume_mult']
                }
        return {'vol_mult': 1.0, 'volume_mult': 1.0}
    
    def _get_hour_factor(self, hour: int) -> float:
        """Get intraday volatility factor"""
        if 8 <= hour <= 16:  # London/New York overlap
            return 1.2
        elif 0 <= hour <= 6:  # Asian session
            return 0.8
        else:
            return 1.0
    
    def _generate_ohlcv_data(self, date_range: pd.DatetimeIndex, prices: np.ndarray, 
                           freq: str) -> list:
        """Generate realistic OHLCV data from price series"""
        ohlc_data = []
        
        for i, (timestamp, close_price) in enumerate(zip(date_range, prices)):
            # Get session parameters for volume
            session_params = self._get_session_parameters(timestamp.hour)
            
            # Calculate realistic OHLC range
            volatility = self.base_volatility * session_params['vol_mult']
            base_range = close_price * volatility * np.random.uniform(0.5, 2.0)
            
            # Generate open price (with possible gap from previous close)
            if i == 0:
                open_price = close_price * np.random.uniform(0.9995, 1.0005)
            else:
                gap_size = np.random.normal(0, volatility * 0.2)
                open_price = prices[i-1] * (1 + gap_size)
            
            # Generate realistic high/low based on candle type
            if close_price > open_price:  # Bullish candle
                upward_move = (close_price - open_price)
                high = max(open_price, close_price) + np.random.uniform(0, upward_move * 0.3)
                low = min(open_price, close_price) - np.random.uniform(0, base_range * 0.2)
            else:  # Bearish candle
                downward_move = (open_price - close_price)
                high = max(open_price, close_price) + np.random.uniform(0, base_range * 0.2)
                low = min(open_price, close_price) - np.random.uniform(0, downward_move * 0.3)
            
            # Ensure OHLC relationships are correct
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Round to tick size
            open_price = round(open_price / self.tick_size) * self.tick_size
            high = round(high / self.tick_size) * self.tick_size
            low = round(low / self.tick_size) * self.tick_size
            close_price = round(close_price / self.tick_size) * self.tick_size
            
            # Generate realistic volume
            base_volume = 5000 * session_params['volume_mult']
            volume_noise = np.random.uniform(0.3, 3.0)
            volume = int(base_volume * volume_noise)
            
            ohlc_data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume,
                'spread': self.spread
            })
        
        return ohlc_data
    
    def _validate_generated_data(self, df: pd.DataFrame) -> None:
        """Validate generated market data for realism"""
        try:
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")
            
            # Check OHLC relationships
            # Ensure high >= open,close and low <= open,close
            invalid_ohlc = (
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ).any()
            
            if invalid_ohlc:
                raise ValueError("Invalid OHLC relationships found")
            
            # Check for zero or negative prices
            zero_prices = (df[['open', 'high', 'low', 'close']] <= 0).any().any()
            if zero_prices:
                raise ValueError("Zero or negative prices found")
            
            # Check for reasonable price movements
            price_changes = df['close'].pct_change().dropna()
            extreme_moves = (price_changes.abs() > 0.05).sum()  # More than 5% moves
            if extreme_moves > len(df) * 0.01:  # More than 1% extreme moves
                logger.warning(f"High number of extreme price moves: {extreme_moves}")
            
            logger.debug("Generated data validation passed")
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise
    
    def _create_fallback_data(self, periods: int) -> pd.DataFrame:
        """Create minimal valid fallback data"""
        logger.warning("Creating fallback data due to generation error")
        
        start_time = datetime.now() - timedelta(minutes=5 * periods)
        date_range = pd.date_range(start=start_time, periods=periods, freq='5min')
        
        # Generate simple linear price series with small random movements
        base_price = self.base_price
        prices = [base_price + i * 0.01 + np.random.normal(0, 0.5) for i in range(periods)]
        
        data = []
        for i, (timestamp, price) in enumerate(zip(date_range, prices)):
            data.append({
                'timestamp': timestamp,
                'open': price,
                'high': price + 0.5,
                'low': price - 0.5,
                'close': price,
                'volume': np.random.randint(1000, 10000),
                'spread': self.spread
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_market_stats(self, data: pd.DataFrame) -> Dict:
        """Get market statistics for generated data"""
        try:
            price_range = (data['high'].max() - data['low'].min())
            avg_range = (data['high'] - data['low']).mean()
            total_volume = data['volume'].sum()
            volatility = data['close'].pct_change().std() * 100
            
            return {
                'period_start': data.index[0].isoformat(),
                'period_end': data.index[-1].isoformat(),
                'records': len(data),
                'price_range': float(price_range),
                'min_price': float(data['low'].min()),
                'max_price': float(data['high'].max()),
                'current_price': float(data['close'].iloc[-1]),
                'avg_range': float(avg_range),
                'total_volume': int(total_volume),
                'avg_volume': int(total_volume / len(data)),
                'volatility': float(volatility),
            }
            
        except Exception as e:
            logger.error(f"Error calculating market stats: {e}")
            return {}
    
    def create_realistic_gap(self, overnight: bool = True) -> float:
        """Create realistic price gap"""
        if overnight:
            # Overnight gaps (larger)
            return np.random.normal(0, self.base_volatility * 3)
        else:
            # Intraday gaps (smaller)
            return np.random.normal(0, self.base_volatility * 0.5)
    
    def simulate_news_impact(self, magnitude: str = 'medium') -> float:
        """Simulate news impact on price"""
        impact_multipliers = {
            'low': 1.5,
            'medium': 3.0,
            'high': 5.0,
            'extreme': 10.0
        }
        
        mult = impact_multipliers.get(magnitude, 3.0)
        return np.random.choice([-1, 1]) * np.random.uniform(0, self.base_volatility * mult)
