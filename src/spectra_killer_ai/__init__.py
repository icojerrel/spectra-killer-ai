"""
Spectra Killer AI - Advanced Trading System

Professional AI-driven trading platform with:
- CNN deep learning pattern recognition
- Hybrid AI + technical analysis signals
- Real-time MT5 integration
- Advanced risk management
- Comprehensive backtesting engine
- Professional web dashboard

Version: 2.0.0 - Production Ready
"""

__version__ = "2.0.0"
__author__ = "Spectra Killer AI Team"
__license__ = "MIT"
__description__ = "Advanced AI Trading System with CNN Deep Learning"

# Core imports
from .core.engine import SpectraTradingEngine
from .core.risk_manager import RiskManager
from .core.portfolio import Portfolio
from .core.position import Position

# Strategy imports
from .strategies.technical.technical_analyzer import TechnicalAnalyzer
from .strategies.ml.cnn_trader import CNNTradingStrategy

# Data imports
from .data.sources.mt5_connector import MT5Connector
from .data.sources.simulator import XAUUSDSimulator

# Import for backward compatibility
SpectraTradingBot = SpectraTradingEngine

# Convenience functions
def create_bot(config=None):
    """Create a new trading bot instance"""
    return SpectraTradingEngine(config or get_default_config())

async def quick_analysis():
    """Quick XAUUSD market analysis"""
    bot = create_bot()
    return await bot.quick_analysis()

def get_default_config():
    """Get default configuration"""
    return {
        'trading': {
            'symbol': 'XAUUSD',
            'timeframe': 'M5',
            'initial_balance': 10000,
        },
        'risk_management': {
            'max_risk_per_trade': 0.02,
            'max_positions': 3,
            'stop_loss_pips': 20,
            'take_profit_pips': 40,
        },
        'ai': {
            'enabled': True,
            'model_type': 'cnn_hybrid',
            'confidence_threshold': 0.65,
        }
    }

# Export public API
__all__ = [
    # Core classes
    'SpectraTradingEngine',
    'SpectraTradingBot',  # Backward compatibility
    'RiskManager',
    'Portfolio',
    'Position',
    
    # Strategy classes
    'TechnicalAnalyzer',
    'CNNTradingStrategy',
    
    # Data classes
    'MT5Connector',
    'XAUUSDSimulator',
    
    # Functions
    'create_bot',
    'quick_analysis',
    'get_default_config',
    
    # Metadata
    '__version__',
    '__author__',
    '__license__',
]
