# XAUUSD Calibration Guide for NQ Stats

## Overview

NQ Stats is based on 10-20 years of Nasdaq-100 E-Mini futures (NQ) data. To use it effectively with XAUUSD (Gold), you need to **calibrate** the statistical parameters to match gold's unique characteristics.

This guide explains how to adapt NQ Stats for XAUUSD trading.

---

## Why Calibration is Necessary

Different instruments have different:
- **Volatility profiles** - Gold vs Tech futures
- **Session dynamics** - Metals vs Equities markets
- **Standard deviations** - Price movement patterns
- **Probability distributions** - Mean reversion characteristics

**Key Differences: NQ vs XAUUSD**

| Characteristic | NQ | XAUUSD |
|---------------|-----|---------|
| **Primary Market** | US Equity Futures | Global Metals |
| **Primary Session** | NY Session | London + NY |
| **Volatility** | Medium-High | Medium |
| **Mean Reversion** | Moderate | Strong |
| **News Sensitivity** | Earnings, Fed | Geopolitics, USD, Rates |
| **Typical Daily Range** | 1-2% | 0.5-1.5% |

---

## Step 1: Calculate XAUUSD SDEV Values

The most important calibration is **standard deviation (SDEV)** values.

### Method 1: Simple Historical Calculation

```python
import pandas as pd
import numpy as np

# Load 1-2 years of XAUUSD historical data
# data = load_from_mt5() or load_from_csv()

# Calculate daily returns
daily_data = data.resample('1D').last()
returns = daily_data['close'].pct_change().dropna()

# Calculate standard deviation (as percentage)
daily_sdev = returns.std() * 100

print(f"XAUUSD Daily SDEV: {daily_sdev:.3f}%")

# Similarly for other timeframes
hourly_data = data.resample('1H').last()
hourly_returns = hourly_data['close'].pct_change().dropna()
hourly_sdev = hourly_returns.std() * 100

print(f"XAUUSD Hourly SDEV: {hourly_sdev:.3f}%")
```

### Method 2: Rolling Window Calculation

For more robust results, use rolling window:

```python
# 252 trading days window
window_size = 252

rolling_std = returns.rolling(window=window_size).std()
median_std = rolling_std.median() * 100

print(f"XAUUSD Daily SDEV (median): {median_std:.3f}%")
```

### Expected Values for XAUUSD

Based on typical gold volatility:

| Timeframe | NQ Default | Expected XAUUSD | Notes |
|-----------|-----------|-----------------|-------|
| **Monthly** | 4.5% | 3.5-4.0% | Lower |
| **Weekly** | 2.1% | 1.5-2.0% | Lower |
| **Daily** | 1.376% | 0.8-1.2% | Lower |
| **Hourly** | 0.34% | 0.2-0.3% | Lower |

**Gold is typically less volatile than NQ**, so expect lower SDEV values.

### Update Configuration

Once calculated, update `config/nq_stats_config.yaml`:

```yaml
# Calibration for XAUUSD
calibration:
  instrument: "XAUUSD"
  calibration_date: "2025-01-01"
  calibration_period_years: 2

# SDEV values calibrated for XAUUSD
sdev_values:
  monthly: 3.8    # Calibrated from historical data
  weekly: 1.8     # Calibrated from historical data
  daily: 1.0      # Calibrated from historical data
  hourly: 0.25    # Calibrated from historical data
```

---

## Step 2: Validate Session Patterns

Session dynamics differ between NQ and XAUUSD.

### Key Differences

**NQ Session Importance:**
1. New York (highest)
2. London (high)
3. Asian (lower)

**XAUUSD Session Importance:**
1. London (highest) - Primary gold trading center
2. New York (high)
3. Asian (moderate) - Strong participation from China/India

### Validation Steps

1. **Collect 6 months of data** across all sessions
2. **Analyze breakout probabilities** for each pattern
3. **Compare to NQ Stats probabilities**
4. **Adjust expectations** accordingly

### Expected Adjustments

```yaml
# ALN Session Pattern Probabilities - XAUUSD Adjusted
aln_sessions:
  london_engulfs_asia:
    ny_breaks_london_probability: 0.95    # Slightly lower than NQ (0.98)
    confidence: "HIGH"                     # Still high confidence

  london_partial_up:
    london_high_break_probability: 0.75   # Adjusted from 0.79
    confidence: "HIGH"
```

**Note:** Major patterns should still hold, but probabilities may shift by 2-5%.

---

## Step 3: Test RTH Break Probabilities

RTH (Regular Trading Hours) for gold includes London + NY sessions.

### XAUUSD RTH Definition

For gold, consider using:
- **Start**: 2:00am EST (London open)
- **End**: 4:00pm EST (NY close)

Or traditional:
- **Start**: 9:30am EST (NY equity open)
- **End**: 4:00pm EST

### Validation Process

```python
# Test RTH break probability
opened_outside_count = 0
broke_opposite_count = 0

for day in historical_days:
    rth_open = get_rth_open(day)
    prev_rth_high, prev_rth_low = get_prev_rth_range(day)

    if rth_open > prev_rth_high or rth_open < prev_rth_low:
        opened_outside_count += 1

        # Check if broke opposite side
        daily_high = get_daily_high(day)
        daily_low = get_daily_low(day)

        if rth_open > prev_rth_high and daily_low < prev_rth_low:
            broke_opposite_count += 1
        elif rth_open < prev_rth_low and daily_high > prev_rth_high:
            broke_opposite_count += 1

probability = 1 - (broke_opposite_count / opened_outside_count)
print(f"XAUUSD RTH Break Probability: {probability*100:.1f}%")
# Compare to NQ: 83.29%
```

**Expected:** 78-85% (similar to NQ)

---

## Step 4: Validate 9am Hour Continuation

The 9am hour (9:00-10:00am EST) may be less significant for gold since:
- Gold trading begins at London open (2am EST)
- 9am is mid-session for gold, not opening hour

### Alternative: London Open Hour

Consider using **2am-3am hour** (London open) instead:

```yaml
# Continuation probabilities - XAUUSD Alternative
continuation:
  hour_2am:  # London open
    session_continuation_probability: 0.65-0.70
    confidence: "HIGH"

  hour_9am:  # NY equity open
    ny_session_continuation_probability: 0.60-0.65
    confidence: "MEDIUM"  # Lower than NQ
```

### Validation Steps

1. Collect hourly data for 1-2 years
2. Test 2am hour (London) and 9am hour (NY) separately
3. Calculate continuation probabilities
4. Use whichever shows stronger edge

---

## Step 5: Test Initial Balance

Initial Balance (9:30-10:30am) may work differently for gold.

### XAUUSD Alternatives

**Option 1: Traditional IB (9:30-10:30am)**
- Still valid for NY session trades
- May have lower break probability (~85-92% vs 96%)

**Option 2: London Initial Balance (2:00-3:00am)**
- Captures primary market open
- Likely stronger edge for gold

**Option 3: Combined IB (2:00-3:00am + 9:30-10:30am)**
- Takes range from both opens
- Most conservative approach

### Validation

```python
# Test IB break probability
ib_formed_count = 0
ib_broken_count = 0

for day in historical_days:
    ib_high, ib_low = get_ib_range(day, start='09:30', end='10:30')
    daily_high, daily_low = get_daily_high_low(day)

    ib_formed_count += 1

    if daily_high > ib_high or daily_low < ib_low:
        ib_broken_count += 1

probability = ib_broken_count / ib_formed_count
print(f"XAUUSD IB Break Probability: {probability*100:.1f}%")
# Compare to NQ: 96%
```

**Expected:** 88-94% (slightly lower than NQ)

---

## Step 6: Run Backtest Validation

Use the validation script to test all probabilities:

```bash
python demo_scripts/backtest_nq_stats_validation.py --days 365 --instrument XAUUSD
```

This will:
1. Load XAUUSD historical data
2. Run all NQ Stats analyses
3. Track actual vs. expected probabilities
4. Generate calibration report

### Interpreting Results

**Good Calibration:**
- Mean Absolute Error < 5%
- Max Error < 10%
- All signals within ±5% of expected

**Needs Recalibration:**
- Mean Error > 10%
- Multiple signals off by >10%
- Systematic bias in one direction

---

## Step 7: Update Configuration

After validation, update your config file:

```yaml
# config/nq_stats_config_xauusd.yaml

# Instrument-specific calibration
calibration:
  instrument: "XAUUSD"
  calibration_period_years: 2
  calibration_date: "2025-01-01"
  validated: true
  validation_samples: 500
  mean_calibration_error: 3.2  # From backtest

# Calibrated SDEV values
sdev_values:
  monthly: 3.8    # From historical calculation
  weekly: 1.8     # From historical calculation
  daily: 1.0      # From historical calculation
  hourly: 0.25    # From historical calculation

# Adjusted probabilities (if needed)
rth_breaks:
  opened_outside_prth_probability: 0.81  # Adjusted from 0.8329

initial_balance:
  break_by_eod_probability: 0.92  # Adjusted from 0.96
  upper_half_close:
    high_break_probability: 0.78  # Adjusted from 0.81
  lower_half_close:
    low_break_probability: 0.72   # Adjusted from 0.74

# Session-specific settings for gold
sessions:
  asian:
    start: "20:00"
    end: "02:00"
    liquidity: 0.7           # Higher for gold (China/India)
  london:
    start: "02:00"
    end: "08:00"
    liquidity: 1.5           # Primary gold market
  new_york:
    start: "08:00"
    end: "16:00"
    liquidity: 1.3

# Gold-specific trading hours
trading_hours:
  primary_session_start: "02:00"  # London open
  primary_session_end: "16:00"     # NY close
  ib_start: "02:00"                # London IB
  ib_end: "03:00"
  ib_alt_start: "09:30"            # NY IB alternative
  ib_alt_end: "10:30"
```

---

## Quick Calibration Checklist

- [ ] Calculate XAUUSD SDEV values (daily, hourly)
- [ ] Update `sdev_values` in config
- [ ] Test on 6+ months historical data
- [ ] Run backtest validation script
- [ ] Review calibration error report
- [ ] Adjust probabilities if error > 5%
- [ ] Forward test on demo account for 1 month
- [ ] Monitor live performance metrics
- [ ] Recalibrate quarterly

---

## Common Issues & Solutions

### Issue 1: SDEV Levels Too Wide/Narrow

**Symptom:** Price frequently breaks ±2σ or never reaches ±1σ

**Solution:** Recalculate SDEV on longer data period (2+ years)

### Issue 2: Session Patterns Don't Match

**Symptom:** ALN probabilities off by >10%

**Solution:** Gold follows different session dynamics - adjust expectations or use London-centric patterns

### Issue 3: IB Never Breaks

**Symptom:** IB break probability much lower than expected

**Solution:** Try London IB (2-3am) instead of NY IB (9:30-10:30am)

### Issue 4: 9am Hour No Edge

**Symptom:** 9am continuation probability ~50%

**Solution:** Use London open hour (2am-3am) or NY equity open micro-structure

---

## Recommended Approach for XAUUSD

1. **Start with NQ default values** - Use as baseline
2. **Calculate XAUUSD SDEV** - Most critical calibration
3. **Test on 6 months data** - Validate patterns
4. **Adjust by 5-10% max** - Don't over-fit
5. **Forward test 1 month** - Demo account validation
6. **Go live with reduced size** - Scale up gradually
7. **Recalibrate quarterly** - Markets change

---

## Example: Complete XAUUSD Calibration

```python
# calibrate_xauusd.py
import pandas as pd
import numpy as np

# 1. Load 2 years of XAUUSD data
data = pd.read_csv('xauusd_historical.csv', parse_dates=['timestamp'], index_col='timestamp')

# 2. Calculate SDEV
daily_returns = data['close'].resample('1D').last().pct_change().dropna()
daily_sdev = daily_returns.std() * 100

hourly_returns = data['close'].resample('1H').last().pct_change().dropna()
hourly_sdev = hourly_returns.std() * 100

print(f"XAUUSD Daily SDEV: {daily_sdev:.3f}%")
print(f"XAUUSD Hourly SDEV: {hourly_sdev:.3f}%")

# 3. Test RTH Breaks
# ... (validation code)

# 4. Generate calibrated config
calibrated_config = {
    'sdev_values': {
        'daily': round(daily_sdev, 3),
        'hourly': round(hourly_sdev, 3)
    }
}

# 5. Save
import yaml
with open('config/nq_stats_config_xauusd.yaml', 'w') as f:
    yaml.dump(calibrated_config, f)

print("✅ Calibration complete!")
```

---

## Additional Resources

- **NQ Stats Complete Guide**: `docs/NQ_STATS_COMPLETE_GUIDE.md`
- **Integration Guide**: `docs/NQ_STATS_INTEGRATION.md`
- **Backtest Script**: `demo_scripts/backtest_nq_stats_validation.py`

---

**Remember:** Calibration is an ongoing process. Monitor performance and recalibrate quarterly or when market regime changes significantly.

*Last Updated: November 2025*
