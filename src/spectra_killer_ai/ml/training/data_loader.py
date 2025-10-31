"""
Market data loader for ML training
Handles loading and preprocessing of historical market data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging

from ..data.sources.mt5_connector import MT5Connector
from ..data.sources.simulator import XAUUSDSimulator

logger = logging.getLogger(__name__)


class MarketDataLoader:
    """Loads and processes market data for ML training"""
    
    def __init__(self, use_real_data: bool = False):
        """
        Initialize data loader
        
        Args:
            use_real_data: Use real MT5 data or simulated data
        """
        self.use_real_data = use_real_data
        
        if use_real_data:
            self.data_source = MT5Connector(demo_mode=True)
        else:
            self.data_source = XAUUSDSimulator()
        
        logger.info(f"Data loader initialized (real_data: {use_real_data})")
    
    def load_training_data(self, 
                          symbol: str = "XAUUSD",
                          timeframe: str = "M5",
                          days: int = 365,
                          validation_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and validation data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            days: Number of days of data
            validation_split: Validation split ratio
            
        Returns:
            Tuple of (train_data, val_data)
        """
        logger.info(f"Loading {days} days of {symbol} {timeframe} data...")
        
        # Get historical data
        if self.use_real_data:
            data = self._load_mt5_data(symbol, timeframe, days)
        else:
            data = self._load_simulated_data(timeframe, days)
        
        if data is None or len(data) == 0:
            logger.error("Failed to load market data")
            return pd.DataFrame(), pd.DataFrame()
        
        # Preprocess data
        processed_data = self._preprocess_data(data)
        
        # Split into train/validation
        split_point = int(len(processed_data) * (1 - validation_split))
        
        train_data = processed_data.iloc[:split_point].copy()
        val_data = processed_data.iloc[split_point:].copy()
        
        logger.info(f"Loaded {len(data)} total rows -> {len(train_data)} train, {len(val_data)} validation")
        
        return train_data, val_data
    
    def _load_mt5_data(self, symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
        """Load data from MetaTrader 5"""
        try:
            if not self.data_source.is_connected():
                if not self.data_source.connect():
                    logger.error("Failed to connect to MT5")
                    return None
            
            # Get historical rates
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1,
            }
            
            mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_M5)
            
            # Calculate number of candles
            timeframes_per_day = {
                'M1': 1440, 'M5': 288, 'M15': 96, 'M30': 48,
                'H1': 24, 'H4': 6, 'D1': 1
            }
            
            num_candles = days * timeframes_per_day.get(timeframe, 288)
            
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, num_candles)
            
            if rates is None or len(rates) == 0:
                logger.error(f"No data received from MT5 for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            logger.info(f"Loaded {len(df)} rows from MT5")
            return df
            
        except Exception as e:
            logger.error(f"Error loading MT5 data: {e}")
            return None
    
    def _load_simulated_data(self, timeframe: str, days: int) -> pd.DataFrame:
        """Load simulated market data"""
        try:
            timeframes_per_day = {
                'M1': 1440, 'M5': 288, 'M15': 96, 'M30': 48,
                'H1': 24, 'H4': 6, 'D1': 1
            }
            
            num_candles = days * timeframes_per_day.get(timeframe, 288)
            
            # Generate data
            data = self.data_source.generate_data(
                timeframe=timeframe,
                periods=num_candles
            )
            
            logger.info(f"Generated {len(data)} rows of simulated data")
            return data
            
        except Exception as e:
            logger.error(f"Error generating simulated data: {e}")
            return pd.DataFrame()
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess market data for ML training
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Preprocessed DataFrame with features
        """
        try:
            # Create copy to avoid modifying original
            df = data.copy()
            
            # Validate data
            self._validate_ohlc_data(df)
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Calculate price changes
            df['price_change'] = df['close'].pct_change()
            df['price_change_abs'] = df['price_change'].abs()
            df['volume_change'] = df['volume'].pct_change()
            
            # Calculate returns at different horizons
            for period in [1, 5, 15, 30, 60]:
                df[f'return_{period}m'] = df['close'].pct_change(period)
            
            # Calculate volatility
            df['volatility_5'] = df['close'].rolling(window=5).std()
            df['volatility_15'] = df['close'].rolling(window=15).std()
            df['volatility_60'] = df['close'].rolling(window=60).std()
            
            # Calculate ATR (Average True Range)
            df['tr_low'] = df['high'] - df['low']
            df['tr_close'] = abs(df['high'] - df['close'].shift(1))
            df['tr_open'] = abs(df['low'] - df['close'].shift(1))
            df['true_range'] = df[['tr_low', 'tr_close', 'tr_open']].max(axis=1)
            df['atr'] = df['true_range'].rolling(window=14).mean()
            
            # Calculate market session features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = df.index.dayofweek >= 5
            
            # Session identification (Sydney, Tokyo, London, NY)
            df['session'] = df['hour'].apply(self._get_trading_session)
            df['session_encoded'] = df['session'].map(self._session_encoding)
            
            # Calculate rolling statistics
            for window in [10, 20, 50]:
                df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
                df[f'close_std_{window}'] = df['close'].rolling(window).std()
                df[f'close_zscore_{window}'] = (df['close'] - df[f'close_mean_{window}']) / df[f'close_std_{window}']
            
            # Calculate high/low price positions
            df['high_position'] = (df['high'] - df['low']) / df['low']
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Calculate candle patterns
            df['is_doji'] = abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1
            df['is_hammer'] = ((df['high'] - df['close']) < 0.25 * (df['high'] - df['low'])) & \
                               ((df['close'] - df['low']) > 0.6 * (df['high'] - df['low']))
            df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
            
            # Drop rows with NaN values
            initial_len = len(df)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            logger.info(f"Preprocessed {len(df)} rows (dropped {initial_len - len(df)} NaN rows)")
            
            # Create target variable
            df = self._create_target_variables(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return data
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Moving Averages
            for period in [5, 10, 20, 50, 100]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # EMA crossovers
            df['ema_5_cross_20'] = (df['ema_5'] > df['ema_20']) & (df['ema_5'].shift(1) <= df['ema_20'].shift(1))
            df['ema_20_cross_50'] = (df['ema_20'] > df['ema_50']) & (df['ema_20'].shift(1) <= df['ema_50'].shift(1))
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['close'].rolling(bb_period).mean()
            df['bb_std'] = df['close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * bb_std)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * bb_std)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Stochastic Oscillator
            lowest_low = df['low'].rolling(14).min()
            highest_high = df['high'].rolling(14).max()
            df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def _create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for supervised learning"""
        try:
            # Target for classification (buy/sell/hold)
            future_returns = {}
            
            # Look ahead returns at different horizons
            for minutes in [5, 15, 30, 60]:
                future_returns[f'target_return_{minutes}m'] = df['close'].shift(-minutes) / df['close'] - 1
            
            # Classification targets
            for minutes in [5, 15, 30, 60]:
                future_return = future_returns[f'target_return_{minutes}m']
                
                # Buy signal if return > threshold
                buy_threshold = 0.001  # 0.1% profit
                # Sell signal if return < negative threshold
                sell_threshold = -0.001  # -0.1% loss
                
                target = pd.Series(0, index=df.index)  # Default: HOLD (0)
                target[future_return > buy_threshold] = 1   # BUY (1)
                target[future_return < sell_threshold] = -1  # SELL (-1)
                
                df[f'target_{minutes}m'] = target
            
            # Regression targets
            df['target_return_5m_reg'] = df['close'].shift(-5) / df['close'] - 1
            df['target_return_15m_reg'] = df['close'].shift(-15) / df['close'] - 1
            df['target_return_30m_reg'] = df['close'].shift(-30) / df['close'] - 1
            df['target_return_60m_reg'] = df['close'].shift(-60) / df['close'] - 1
            
            # Add volatility targets
            for minutes in [5, 15, 30, 60]:
                volatility_target = df[f'return_{minutes}m'].rolling(20).std() * df[f'return_{minutes}m'].shift(-minutes)
                df[f'target_volatility_{minutes}m'] = volatility_target
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating target variables: {e}")
            return df
    
    def _get_trading_session(self, hour: int) -> str:
        """Get trading session based on hour"""
        if 22 <= hour or hour < 2:
            return 'sydney'
        elif 0 <= hour < 8:
            return 'tokyo'
        elif 8 <= hour < 16:
            return 'london'
        else:  # 16 <= hour < 22
            return 'new_york'
    
    def _session_encoding(self, session: str) -> int:
        """Encode trading session as integer"""
        encoding = {'sydney': 0, 'tokyo': 1, 'london': 2, 'new_york': 3}
        return encoding.get(session, 0)
    
    def _validate_ohlc_data(self, df: pd.DataFrame) -> None:
        """Validate OHLC data integrity"""
        if len(df) == 0:
            raise ValueError("Empty dataframe provided")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check OHLC relationships
        invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
        invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
        
        if invalid_high.any() or invalid_low.any():
            logger.warning(f"Found {invalid_high.sum()} invalid high values and {invalid_low.sum()} invalid low values")
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for training"""
        return [
            # Price features
            'open', 'high', 'low', 'close', 'volume',
            'price_change', 'price_change_abs', 'volume_change',
            
            # Returns
            'return_1m', 'return_5m', 'return_15m', 'return_30m', 'return_60m',
            
            # Volatility
            'volatility_5', 'volatility_15', 'volatility_60', 'atr',
            
            # Technical indicators
            'rsi', 'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_position',
            'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d',
            
            # Time features
            'hour', 'day_of_week', 'session_encoded',
            
            # Pattern features
            'is_doji', 'is_hammer', 'body_size', 'high_position', 'close_position',
            
            # Rolling stats
            'close_zscore_10', 'close_zscore_20', 'close_zscore_50',
        ]
    
    def get_target_columns(self) -> List[str]:
        """Get list of target columns for training"""
        return [
            # Classification targets
            'target_5m', 'target_15m', 'target_30m', 'target_60m',
            
            # Regression targets
            'target_return_5m_reg', 'target_return_15m_reg', 
            'target_return_30m_reg', 'target_return_60m_reg',
            
            # Volatility targets
            'target_volatility_5m', 'target_volatility_15m',
            'target_volatility_30m', 'target_volatility_60m',
        ]
