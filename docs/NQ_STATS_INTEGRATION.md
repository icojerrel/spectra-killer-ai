# NQ Stats Integration Guide

## Overview

This guide explains how to integrate the NQ Stats probability-based analysis into the Spectra Killer AI trading system.

## What is NQ Stats?

NQ Stats is a comprehensive metrics library based on **10-20 years of Nasdaq-100 E-Mini (NQ) historical data (2004-2025)**. It provides probability-based trading signals derived from:

- Hour-by-hour price behavior patterns
- Session relationships (Asian/London/New York)
- Standard deviation-based mean reversion
- Initial Balance breakout probabilities
- RTH (Regular Trading Hours) patterns
- And much more...

## Files Added

### Documentation
- `docs/NQ_STATS_COMPLETE_GUIDE.md` - Comprehensive guide to all NQ Stats methodologies
- `docs/NQ_STATS_INTEGRATION.md` - This integration guide

### Code
- `src/spectra_killer_ai/strategies/advanced/nq_stats_analyzer.py` - Main analyzer implementation
- `src/spectra_killer_ai/strategies/advanced/__init__.py` - Updated to export NQ Stats components

### Configuration
- `config/nq_stats_config.yaml` - Complete configuration with all probabilities and parameters

### Demo
- `demo_scripts/nq_stats_demo.py` - Demonstration script showing usage

## Quick Start

### 1. Run the Demo

```bash
python demo_scripts/nq_stats_demo.py
```

This will generate sample data and show all NQ Stats analysis outputs.

### 2. Basic Usage in Code

```python
from spectra_killer_ai.strategies.advanced import NQStatsAnalyzer
import pandas as pd
import yaml

# Load configuration
with open('config/nq_stats_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize analyzer
analyzer = NQStatsAnalyzer(config)

# Analyze market data (OHLCV DataFrame with datetime index)
results = analyzer.analyze(data)

# Access signals
signals = results['signals']
confidence = results['overall_confidence']

print(f"Trading recommendation: {confidence['recommendation']}")
print(f"Confidence level: {confidence['level']}")
print(f"Number of signals: {len(signals)}")
```

### 3. Integration with Existing Trading Strategy

```python
from spectra_killer_ai.strategies.advanced import NQStatsAnalyzer
from spectra_killer_ai.core import RiskManager

# In your trading strategy class:

def __init__(self, config):
    self.nq_stats = NQStatsAnalyzer(config['nq_stats'])
    self.risk_manager = RiskManager(config['risk'])

async def generate_signal(self, data):
    # Run NQ Stats analysis
    nq_analysis = self.nq_stats.analyze(data)

    # Get overall confidence
    confidence = nq_analysis['overall_confidence']

    # Use SDEV levels for dynamic TP/SL
    sdev_levels = nq_analysis['sdev_analysis']['sdev_levels']['daily']
    tp_level = sdev_levels['levels']['+1.0σ']
    sl_level = sdev_levels['levels']['-1.5σ']

    # Check 9am hour bias
    hour_9am = nq_analysis.get('9am_continuation', {})
    if hour_9am.get('available'):
        bias = hour_9am['bias']
        # Use bias to filter signals

    # Adjust position size based on confluence
    base_position = 1.0
    position_multiplier = confidence['position_sizing']  # FULL or REDUCED

    # Generate your signal with NQ Stats enhancement
    return {
        'direction': confidence['recommendation'],
        'confidence': confidence['score'],
        'take_profit': tp_level,
        'stop_loss': sl_level,
        'position_size': base_position * position_multiplier
    }
```

## Key Features

### 1. Standard Deviation Analysis

The "Bread and Butter Edge" - Probability-based levels:

```python
results = analyzer.analyze(data)
sdev = results['sdev_analysis']['sdev_levels']['daily']

# Get current position
current_price = data['close'].iloc[-1]
print(f"Current SDEV distance: {sdev['sdev_distance']}σ")
print(f"Zone: {sdev['zone']}")
print(f"Reversion probability: {sdev['reversion_probability']*100:.1f}%")

# Access levels
print(f"+1.0σ level: {sdev['levels']['+1.0σ']}")
print(f"-1.0σ level: {sdev['levels']['-1.0σ']}")
```

### 2. 9am Hour Continuation (70% Edge)

```python
hour_9am = results['9am_continuation']
if hour_9am['available']:
    print(f"9am bias: {hour_9am['bias']}")
    print(f"Continuation probability: {hour_9am['ny_session_continuation_probability']*100:.0f}%")
    # Trade WITH this bias for 70% historical edge
```

### 3. RTH Breaks (83.29% Edge)

```python
rth = results['rth_breaks']
if rth['available']:
    print(f"Scenario: {rth['scenario']}")
    print(f"Bias: {rth['bias']}")
    print(f"Probability: {rth['probability']*100:.1f}%")
    # Strong directional indicator
```

### 4. Initial Balance (74-81% Edge)

```python
ib = results['initial_balance']
if ib['available']:
    print(f"Expected break: {ib['expected_break_direction']}")
    print(f"Probability: {ib['directional_break_probability']*100:.0f}%")
    print(f"Target: {ib['target_level']}")
```

### 5. Session Patterns (73-98% Edge)

```python
session = results['session_pattern']
if session['available']:
    print(f"Pattern: {session['pattern_type']}")
    print(f"Probability: {session['probability']*100:.0f}%")
    print(f"Expected: {session['expected_behavior']}")
```

## Signal Types

The analyzer generates these signal types:

| Signal Type | Source | Description |
|------------|--------|-------------|
| `MEAN_REVERSION` | sdev_analysis | Price at extreme SDEV - high reversion probability |
| `TREND_DAY` | sdev_analysis | Price extended beyond 1.5σ - momentum trade |
| `DIRECTIONAL_BIAS` | rth_breaks | RTH opened outside pRTH - strong bias |
| `STRONG_BIAS` | 9am_continuation | 9am hour closed - 70% continuation |
| `BREAKOUT_EXPECTED` | initial_balance | IB formed - 74-81% breakout coming |
| `SESSION_PATTERN` | aln_sessions | Overnight pattern detected - 73-98% edge |

## Configuration

All probabilities and parameters are configurable in `config/nq_stats_config.yaml`:

```yaml
# Adjust SDEV values for your instrument
sdev_values:
  daily: 1.376  # NQ default - calibrate for XAUUSD

# Position sizing based on confluence
position_sizing:
  single_signal: 1.0
  two_signals: 1.25
  three_signals: 1.5
  four_plus_signals: 2.0

# Risk management
risk_management:
  sdev_stop_loss_multiplier: 1.5
  sdev_take_profit_multiplier: 1.0
```

## Adapting for Different Instruments

NQ Stats is based on NQ futures, but can be adapted for other instruments:

### For XAUUSD (Gold):

1. **Calibrate SDEV values** through backtesting:
   ```python
   # Run 1-2 years of historical data
   returns = data['close'].pct_change().dropna()
   daily_sdev = returns.std() * 100  # Convert to percentage
   ```

2. **Adjust probabilities** based on instrument characteristics:
   - Gold has different session dynamics
   - Volatility patterns may differ
   - Test and validate each probability metric

3. **Update configuration**:
   ```yaml
   calibration:
     instrument: "XAUUSD"
     volatility_adjustment: 1.2  # Gold more volatile
   ```

### For ES (S&P 500 E-Mini):

ES is highly correlated with NQ, so probabilities should transfer well with minor adjustments:

```yaml
sdev_values:
  daily: 0.95  # ES typically less volatile than NQ
```

## Best Practices

### 1. Use Multiple Confirmations

Don't trade on a single signal. Look for confluence:

```python
confidence = results['overall_confidence']
if confidence['confluence_count'] >= 3:
    # High confidence - full position
    position_size = 2.0
elif confidence['confluence_count'] >= 2:
    # Medium confidence - reduced position
    position_size = 1.25
else:
    # Single signal - minimum position or wait
    position_size = 0.5
```

### 2. Time-Based Filtering

Adjust strategy based on hour segments:

```python
hour_stats = results['hour_stats']
if hour_stats['segment'] == 1:
    # First 20 minutes - high retracement probability
    strategy = 'mean_reversion'
elif hour_stats['segment'] == 2:
    # Middle 20 minutes - expansion phase
    strategy = 'momentum'
```

### 3. Dynamic Risk Management

Use SDEV levels for adaptive stops:

```python
sdev_levels = results['sdev_analysis']['sdev_levels']['daily']
current_position = sdev_levels['sdev_distance']

if abs(current_position) > 1.5:
    # At extremes - tighter stops
    stop_multiplier = 1.0
else:
    # Normal range - standard stops
    stop_multiplier = 1.5
```

### 4. Respect the 9am Hour

```python
hour_9am = results['9am_continuation']
if hour_9am['available']:
    # Filter all signals against 9am bias
    if signal['direction'] != hour_9am['bias']:
        # Going against 9am hour - reduce confidence
        signal['confidence'] *= 0.7
```

## Performance Tracking

Track signal performance to validate probabilities:

```python
from collections import defaultdict

signal_tracker = defaultdict(lambda: {'correct': 0, 'total': 0})

def track_signal_outcome(signal, outcome):
    """Track if signal was correct"""
    signal_tracker[signal.source]['total'] += 1
    if outcome == 'correct':
        signal_tracker[signal.source]['correct'] += 1

def get_signal_accuracy():
    """Calculate actual vs. expected probability"""
    for source, stats in signal_tracker.items():
        actual_prob = stats['correct'] / stats['total']
        print(f"{source}: {actual_prob*100:.1f}% accuracy")
```

## Backtesting

Test NQ Stats on historical data:

```python
from datetime import datetime, timedelta
import pandas as pd

# Load historical data
data = load_historical_data(
    symbol='XAUUSD',
    start_date='2020-01-01',
    end_date='2025-01-01'
)

# Test each day
results = []
for date in pd.date_range(start='2020-01-01', end='2025-01-01'):
    daily_data = data[data.index.date <= date.date()]
    analysis = analyzer.analyze(daily_data, date)

    # Store signals and outcomes
    results.append({
        'date': date,
        'signals': analysis['signals'],
        'confidence': analysis['overall_confidence']
    })

# Validate probabilities
validate_probability_accuracy(results)
```

## Troubleshooting

### Issue: "available": False

**Cause**: Insufficient data for analysis

**Solution**: Ensure you have at least 2-3 days of intraday data (5-minute bars or better)

### Issue: No signals generated

**Cause**: No high-probability setups detected

**Solution**: This is normal - NQ Stats filters for quality setups only

### Issue: Probabilities seem inaccurate

**Cause**: Instrument characteristics differ from NQ

**Solution**: Recalibrate SDEV values and validate probabilities through backtesting

## Resources

- **Complete Methodology**: See `docs/NQ_STATS_COMPLETE_GUIDE.md`
- **Source Research**: nqstats.com (2004-2025 data)
- **Configuration**: `config/nq_stats_config.yaml`
- **Demo Script**: `demo_scripts/nq_stats_demo.py`

## Contributing

To improve NQ Stats integration:

1. Backtest on different instruments
2. Validate probabilities
3. Share calibration results
4. Suggest enhancements

## License

NQ Stats analysis methodology is based on publicly available research from nqstats.com.

Implementation in Spectra Killer AI is licensed under MIT License.

---

**Built with ❤️ for the Spectra Killer AI Trading System**

*Trade smart, trade safe, trade with probabilities*
