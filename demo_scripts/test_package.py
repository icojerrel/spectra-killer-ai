#!/usr/bin/env python3
# Test the new package structure

from spectra_killer_ai.core.bot import SpectraTradingBot
from spectra_killer_ai.data.simulator import create_realistic_xau_data
from spectra_killer_ai.trading.indicators import analyze_xau_5m

print('Package imports successful!')
print('Creating bot...')
bot = SpectraTradingBot()
print('Bot created successfully!')

print('Testing quick analysis...')
data = create_realistic_xau_data()
analysis = analyze_xau_5m(data)
signal = analysis['combined']['signal']
confidence = analysis['combined']['confidence']
print(f'Analysis: {signal} ({confidence}% confidence)')

print('Package test completed successfully!')