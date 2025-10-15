"""
Data sources module
"""

from .sources.simulator import XAUUSDSimulator
from .sources.mt5_connector import MT5Connector

__all__ = [
    'XAUUSDSimulator',
    'MT5Connector',
]
