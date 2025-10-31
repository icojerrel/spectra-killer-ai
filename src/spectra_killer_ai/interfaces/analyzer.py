"""
Quick Analysis Interface for Spectra Killer AI
"""

from ..core.engine import SpectraTradingEngine


async def quick_analysis():
    """Quick XAUUSD market analysis function"""
    engine = SpectraTradingEngine()
    return await engine.quick_analysis()
