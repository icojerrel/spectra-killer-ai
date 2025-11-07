# NQ Stats - Complete Trading Statistics Guide

## Overview

NQ Stats is a comprehensive metrics library for Nasdaq-100 E-Mini (NQ) traders based on **10-20 years of historical data (2004-2025)**. This document catalogs various metrics, statistics, and probabilities that can enhance trading strategies through quantitative, math-derived approaches.

---

## Table of Contents

1. [Hour Stats](#1-hour-stats)
2. [RTH Breaks](#2-rth-breaks-regular-trading-hours)
3. [1H Continuation](#3-1h-continuation)
4. [The Morning Judas](#4-the-morning-judas)
5. [Initial Balance Breaks](#5-initial-balance-breaks)
6. [Net Change Standard Deviations](#6-net-change-standard-deviations)
7. [Noon Curve](#7-noon-curve)
8. [ALN Sessions](#8-aln-sessions)
9. [Implementation Guide](#implementation-guide)

---

## 1. Hour Stats

### Concept

Hour Stats analyzes each individual hour during the NY session, examining probability-based price behavior patterns.

### Key Metrics

#### Sweep & Retracement Probabilities

When the current hour opens within the previous hour's range and sweeps the previous hour's high/low, what is the probability that price returns to the current hour's open?

**Probability Rankings:**

| Ranking | Probability Range |
|---------|-------------------|
| **High** | ≥75% |
| **Medium** | 51% - 75% |
| **Low** | <50% |

#### Time-Segmented Analysis (20-minute segments)

Each hour is divided into three 20-minute segments with distinct behavioral patterns:

| Segment | Time Range | Characteristics | Retracement Probability |
|---------|-----------|-----------------|------------------------|
| **First** | 0-20 min | Highest retracement probability, wick formation | Up to 89% |
| **Second** | 20-40 min | Candle expansion phase | ~47% |
| **Third** | 40-60 min | Wick formation | Lowest |

### High/Low Formation Times

- **First segment (0-20 min)**: Primary wick formation zone
- **Second segment (20-40 min)**: Expansion phase - body formation
- **Third segment (40-60 min)**: Secondary wick formation zone

#### 9am Hour Anomaly

The 9am hour (9:00-9:30am) is unique due to the 9:30am NY market open volatility injection:

- Any high/low formed in the **first 20 minutes** is statistically unlikely to be the final high/low
- Wait for post-market-open price action before confirming levels

### Trading Applications

1. **Entry Timing**: Focus on first 20 minutes for retracement entries (highest probability)
2. **Risk Management**: Avoid early hour commitments during 9am hour
3. **Stop Loss Placement**: Use segment boundaries as dynamic stop levels

---

## 2. RTH Breaks (Regular Trading Hours)

### Definition

- **RTH**: Regular Trading Hours = NY Session (9:30am - 4:00pm EST)
- **pRTH**: Previous day's RTH range (previous day 9:30am - 4:00pm EST)

### Statistical Data (10-year sample: 2015-2025)

#### Scenario 1: RTH Opens OUTSIDE pRTH

When today's RTH open is above pRTH high or below pRTH low:

- **83.29% probability** that price will NOT break the opposite side of pRTH
- **Strong directional bias indicator**

**Example:**
- RTH opens above pRTH high → 83.29% chance price won't break below pRTH low
- RTH opens below pRTH low → 83.29% chance price won't break above pRTH high

#### Scenario 2: RTH Opens WITHIN pRTH

When today's RTH open is inside yesterday's RTH range:

- **72.66% probability** that price will break at least ONE side of pRTH
- Low probability for:
  - Breaking both sides (expansion)
  - Staying completely within pRTH (consolidation)

### Trading Applications

1. **Directional Bias**: Use RTH open position relative to pRTH as primary bias indicator
2. **Breakout Trading**: When opening outside pRTH, favor continuation direction
3. **Range Trading**: When opening inside pRTH, prepare for range breakout (72.66%)

---

## 3. 1H Continuation

### Concept

Continuation probability analysis: When a specific hour closes green/red, what does the rest of the session do?

### Data Analysis

#### 6pm Hour (Asian Open)

**When 6pm hour closes green:**
- Entire session green: **59%** (weak edge)
- NY session when 6pm was red: **44%** chance of red (inverted - actually 56% green!)

**Interpretation:** 6pm hour has limited predictive power for full session direction.

#### 9am Hour (Strongest Statistical Edge!)

**When 9am hour (9:00am-10:00am) closes green:**
- Entire session green: **67%**
- NY session green (4pm > 9:30am): **70%**

**When 9am hour closes red:**
- Inverse probabilities apply
- Entire session red: **67%**
- NY session red: **70%**

**Key Insight:** The 9am hour close provides the **strongest statistical edge** for intraday bias formation.

### Trading Applications

1. **Daily Bias**: Use 9am hour close as primary directional filter (70% accuracy)
2. **Position Management**: Align all intraday positions with 9am hour direction
3. **Contrarian Caution**: Trading against 9am hour direction requires higher conviction (30% success)

---

## 4. The Morning Judas

### Concept

Analysis of price behavior around specific morning time intervals: Does price reverse (Judas) or continue?

### Statistical Data (10 years: 2014-2024)

#### 9:30am - 9:40am - 10:00am Analysis

**Up Judas Scenario (9:40am > 9:30am):**
- False swing (10:00am < 9:30am): **36%**
- **Continuation (10:00am > 9:40am): 64%** ← Higher probability

**Down Judas Scenario (9:40am < 9:30am):**
- False swing (10:00am > 9:30am): **30%**
- **Continuation (10:00am < 9:40am): 70%** ← Higher probability

#### "10am Reversal" Myth-Bust

Testing the 10am reversal theory using 9:30am - 10:00am - 12:00pm analysis:

**Up Judas (10:00am > 9:30am):**
- False swing by noon (12:00pm < 9:30am): **32%**
- **Continuation by noon (12:00pm > 10:00am): 68%** ← Higher probability

**Down Judas (10:00am < 9:30am):**
- False swing by noon (12:00pm > 9:30am): **41%**
- **Continuation by noon (12:00pm < 10:00am): 59%** ← Higher probability

### Key Findings

**MYTH BUSTED:** The "morning Judas" is actually a **continuation pattern**, not a reversal pattern!

- Continuation occurs **59-70% of the time**
- Reversals occur only **30-41% of the time**
- Morning momentum tends to persist through lunch

### Trading Applications

1. **Trend Following**: Trade WITH the 9:40am-10:00am direction, not against it
2. **Reversal Caution**: Reversal setups have only 30-41% historical success
3. **Momentum Confirmation**: Use 10:00am-12:00pm continuation as trend confirmation

---

## 5. Initial Balance Breaks

### Definition

**Initial Balance (IB)**: The high and low established between **9:30am - 10:30am ET**

This 1-hour range represents the market's initial price discovery phase after the opening bell.

### Statistical Data (10 years: 2014-2024)

#### Break Probability by Time

| Timeframe | IB Break Probability | Notes |
|-----------|---------------------|-------|
| **Before 4pm** (end of session) | **96%** | Almost always breaks one side |
| **Before 12pm** (lunch) | **83%** | High probability by midday |

#### Directional Break Probability

Which side of the IB is more likely to break? Based on IB close position:

| IB Close Position | Direction | Break Probability |
|------------------|-----------|-------------------|
| **Upper half of IB** | High breaks | **81%** |
| **Lower half of IB** | Low breaks | **74%** |

### Trading Applications

1. **Breakout Trading**:
   - Wait for IB formation (10:30am)
   - Identify IB close position (upper/lower half)
   - Trade breakout direction with 74-81% historical edge

2. **Risk Management**:
   - 96% probability suggests IB will be challenged by EOD
   - Position size accordingly for IB breakout trades

3. **Timing**:
   - 83% break by noon → Most opportunities occur before lunch
   - If IB hasn't broken by 12pm, probability increases for afternoon break

---

## 6. Net Change Standard Deviations

### Concept: "The Bread and Butter Edge"

Uses standard deviations of percentage net change over the sample period to determine **EXACT probability** of where price will close for any timeframe.

This is a **quantitative, math-derived approach** to pivot points and probability zones.

### Statistical Foundation (20 years: 2004-2024)

#### Daily Timeframe Example

**Standard Deviation**: ±1.376% from session open

**Probability Distribution:**

| SDEV Range | % Within Range | % Above Lower Bound | Trading Interpretation |
|-----------|----------------|---------------------|----------------------|
| **±0.5 SDEV** | 38.30% | 69.15% above -0.5 | Mean reversion zone |
| **±1.0 SDEV** | 68.27% | 84.13% above -1.0 | Normal distribution boundary |
| **±1.5 SDEV** | ~95% | ~97.5% above -1.5 | Outlier territory - trend day |
| **±2.0 SDEV** | ~99% | ~99.5% above -2.0 | Extreme outlier - strong trend |

#### Understanding the Numbers

For Daily timeframe with 1 SDEV = ±1.376%:

```
Session Open: 15,000
+0.5 SDEV: 15,103 (+0.688%)
+1.0 SDEV: 15,206 (+1.376%)
+1.5 SDEV: 15,309 (+2.064%)

-0.5 SDEV: 14,897 (-0.688%)
-1.0 SDEV: 14,794 (-1.376%)
-1.5 SDEV: 14,691 (-2.064%)
```

**Probabilities:**
- 68.27% chance price closes between 14,794 and 15,206
- 84.13% chance price closes above 14,794
- 15.87% chance price closes below 14,794

### The "Rubber Band Theory"

The further price extends from the mean (timeframe open), the more **reversion tension** builds:

1. **Within ±0.5 SDEV**: Normal fluctuation zone
2. **At ±1.0 SDEV**: Probability boundary - watch for reversal
3. **Beyond ±1.5 SDEV**: Strong reversion pressure
4. **Beyond ±2.0 SDEV**: Extreme - rubber band snap likely

**Trend Day Identifier**: When price extends past +1.0 SDEV **without pivoting or consolidating**, it signals a "rubber band snap" - trend day in progress.

### Multi-Timeframe Application

SDEV analysis applies to multiple timeframes:

| Timeframe | Application | Trading Use |
|-----------|-------------|-------------|
| **Monthly** | Long-term positioning | Portfolio allocation |
| **Weekly** | Swing trade targets | Multi-day positions |
| **Daily** | Intraday targets | Day trading pivots |
| **1-Hour** | Scalping zones | Quick mean reversion |

### Why It Works

Pivot points derived from SDEV are **quantitative and math-derived**, not arbitrary:

- Large institutional players likely use similar quantitative approaches
- Creates self-fulfilling prophecy effect at key SDEV levels
- Mathematical basis provides objective entry/exit points

### Trading Applications

1. **Take Profit Levels**: Use ±1.0 SDEV as primary targets
2. **Stop Loss Placement**: Place stops beyond ±1.5 SDEV (low probability zones)
3. **Mean Reversion**: Trade reversals at ±1.0 SDEV with 68% probability support
4. **Trend Confirmation**: Price beyond ±1.5 SDEV = trend day, trade with momentum
5. **Position Sizing**: Reduce size at SDEV extremes (higher risk zones)

---

## 7. Noon Curve

### Concept

Synthetic 8-hour candle (8:00am - 4:00pm) with 12:00pm (noon) as the mid-point, analyzing high/low formation probabilities.

### Statistical Data (20 years: 2004-2024)

#### High & Low Formation Probabilities

**Primary Finding:**

- **74.3%** of the time, high and low form on **OPPOSITE sides** of 12pm
- **25.7%** both form on the same side

**Interpretation:**
- If AM (8am-12pm) sets the low → Expect PM (12pm-4pm) to set the high
- If AM sets the high → Expect PM to set the low

### Quarterly Confluence (2-hour segments)

Breaking the 8-hour period into four 2-hour quarters:

| Quarter | Time Range | Characteristics | Probability |
|---------|-----------|----------------|-------------|
| **Q1** | 8am-10am | Sets AM high/low | 66-68% |
| **Q2** | 10am-12pm | Secondary formation | If Q1 low broken → PM low likely |
| **Q3** | 12pm-2pm | PM structure begins | - |
| **Q4** | 2pm-4pm | Final push/reversion | - |

#### Q1 Importance (8am-10am)

- **66-68% probability** that Q1 sets the AM session high or low
- Strong indicator for AM/PM structure

#### Q2 Significance (10am-12pm)

- If Q1 low is broken in Q2 → Increases probability of new PM low
- If Q1 high is broken in Q2 → Increases probability of new PM high

### Optimal Formation Times & Distances

Historical analysis shows **optimal times** for high/low formation and their typical **distance from 8am open**.

*Note: Specific distance metrics available in detailed NQ Stats database*

### Trading Applications

1. **Structure Mapping**:
   - Identify high/low formation before noon
   - Project opposite extreme for PM session (74.3% probability)

2. **Time-Based Entries**:
   - Q1 (8am-10am): Primary structure formation - wait for clarity
   - Q2 (10am-12pm): Confirmation phase - enter with AM bias
   - Q3 (12pm-2pm): Opposite extreme setup - prepare for reversal
   - Q4 (2pm-4pm): Final targets - manage exits

3. **Range Projection**:
   - AM low → PM high target
   - AM high → PM low target
   - Use SDEV levels for distance projection

---

## 8. ALN Sessions (Asian-London-New York)

### Session Definitions

| Session | Time Range (EST) | Characteristics |
|---------|-----------------|----------------|
| **Asian** | 8pm (prev day) - 2am | Lowest volatility, range-bound |
| **London** | 2am - 8am | Medium-high volatility, trend potential |
| **New York** | 8am - 4pm | Highest volatility and volume |

### Statistical Data (10 years: August 2015 - August 2025)

#### Pattern 1: London Engulfs Asia (Most Common)

When London High > Asia High AND London Low < Asia Low:

**NY Session Behavior:**
- **98%** NY breaks at least one side of London range
- **43%** NY engulfs entire overnight range (both Asia and London)

**Sequential Break Probabilities:**

| First Break | Second Break | Probability |
|------------|--------------|-------------|
| London High first | Then London Low | Only **44%** |
| London Low first | Then London High | Only **45%** |

**Key Insight:** If NY takes one side of London first, probability of taking the opposite side drops significantly.

#### Pattern 2: Asia Engulfs London (Rare - 7%)

When Asia High > London High AND Asia Low < London Low:

**NY Session Behavior:**
- **95%** NY breaks London High OR Low
- **91%** NY breaks Asia High OR Low
- **39%** NY engulfs entire overnight range

#### Pattern 3: London Partially Engulfs Upwards

When London High > Asia High BUT London Low stays within Asia range:

**NY Session Probabilities:**
- **79%** NY breaks London High
- **63%** NY breaks London Low
- **51%** NY breaks Asia Low

**If Opposite Side Taken First:**
- If NY takes London Low before High → Probability of taking High drops to ~48-50%

#### Pattern 4: London Partially Engulfs Downwards

When London Low < Asia Low BUT London High stays within Asia range:

**NY Session Probabilities:**
- **73%** NY breaks London Low
- **66%** NY breaks London High
- **54%** NY breaks Asia High

**If Opposite Side Taken First:**
- If NY takes London High before Low → Probability of taking Low drops to ~41-43%

### Order of Operations Impact

The **sequence** of level breaks significantly affects subsequent probabilities:

```
Standard Probability: 79% (London High break)
After opposite side taken: 48-50% (50% reduction)

Standard Probability: 73% (London Low break)
After opposite side taken: 41-43% (43% reduction)
```

### Trading Applications

1. **Session Relationship Mapping**:
   - Identify overnight pattern before NY open (8am EST)
   - Determine primary directional bias based on pattern type

2. **Sequential Probability Trading**:
   - **First Break**: Trade WITH highest probability direction
   - **Second Break**: Reduce position size (probability drops 40-50%)

3. **Pattern-Specific Strategies**:

   **London Engulfs Asia (98% edge):**
   - Aggressive: Trade London range breakouts early in NY session
   - Conservative: Wait for first break, fade second attempt

   **Partial Engulfs (73-79% edge):**
   - Trade direction of London expansion
   - Be cautious if opposite side taken first

4. **Risk Management**:
   - Pattern 1 (London Engulfs): Highest confidence trades
   - Pattern 2 (Asia Engulfs): Rare but significant - high alert
   - Patterns 3-4 (Partial): Good edge but watch order of operations

---

## Implementation Guide

### Highest Probability Setups (Ranked)

1. **RTH Opens Outside pRTH** - 83.29% edge
   - Application: Strong directional bias for entire session
   - Trade Type: Swing/day trades in breakout direction

2. **IB Close Position** - 74-81% edge
   - Application: Directional bias for IB breakout
   - Trade Type: Breakout trades after 10:30am

3. **Noon Curve Opposite Sides** - 74.3% edge
   - Application: AM/PM structure projection
   - Trade Type: Counter-structure trades at noon

4. **9am Hour Continuation** - 67-70% edge
   - Application: Intraday directional bias
   - Trade Type: Trend following throughout session

5. **Net Change SDEV Pivots** - 68-84% edge (depending on level)
   - Application: Mean reversion and trend identification
   - Trade Type: Reversal trades at SDEV levels

### Time-Based Trading Patterns

#### Intraday Time Map

```
8:00am - 9:00am   : Q1 structure formation (pre-market continuation)
9:00am - 9:30am   : 9am hour (bias formation) - WAIT for 9:30am open
9:30am - 10:00am  : Market open volatility, morning Judas setup
10:00am - 10:30am : IB formation completion
10:30am - 12:00pm : IB breakout window (83% by noon)
12:00pm           : Noon curve pivot - expect opposite extreme
12:00pm - 2:00pm  : Post-lunch continuation/reversal
2:00pm - 4:00pm   : Final push or reversion to mean
```

#### Hourly Segment Trading

**First 20 Minutes:**
- Highest retracement probability (up to 89%)
- Wick formation zone
- Best for reversal/retracement entries

**Middle 20 Minutes:**
- Expansion phase (~47% retracement probability)
- Trade breakouts and momentum

**Final 20 Minutes:**
- Secondary wick formation
- Prepare for next hour dynamics

### Integration Strategies

#### Multi-Timeframe Confirmation

Combine multiple NQ Stats for highest probability setups:

**Example: High Probability Long Setup**
1. ✅ RTH opens above pRTH high (83.29% won't go below pRTH low)
2. ✅ 9am hour closes green (70% NY session stays bullish)
3. ✅ IB closes in upper half (81% IB high will break)
4. ✅ Price at -1.0 SDEV (68% probability of reversion up)
5. ✅ AM set the low, expecting PM high (74.3% probability)

**Combined Probability**: Extremely high confidence setup

#### Risk Management Enhancement

**Position Sizing Based on Confluence:**
- 1 signal: Base position (1x)
- 2-3 signals: Increased position (1.5x)
- 4+ signals: Maximum position (2x)

**Dynamic Stop Loss:**
- Use SDEV levels as stop boundaries
- Place stops beyond -1.5 SDEV (outside 95% probability zone)
- Tighten stops during low probability hours

**Time-Based Position Management:**
- Reduce exposure during first 20 minutes of each hour (wick formation)
- Increase exposure during middle segment (expansion phase)
- Close or hedge positions before major time pivots (10am, 12pm, 2pm)

### Statistical Edge Philosophy

**Core Principles:**

1. **Quantitative Foundation**: All metrics based on 10-20 years of historical data
2. **Math-Derived Pivots**: SDEV levels provide objective, non-arbitrary entry/exit points
3. **Institutional Alignment**: Large players likely use similar quant approaches, creating self-fulfilling levels
4. **Probability-Based**: Trade with probabilities, not predictions
5. **Multiple Confirmation**: Higher confluence = higher probability = larger position

### Myth-Busting Summary

**Common Trading Myths vs. NQ Stats Reality:**

| Myth | NQ Stats Reality | Edge |
|------|-----------------|------|
| "10am reversal is common" | Continuation > Reversal | 59-68% continuation |
| "Morning Judas means fade" | Morning Judas = continuation | 64-70% continuation |
| "Trade against early move" | Trade WITH 9am hour close | 70% accuracy |
| "Ranges hold overnight" | IB breaks 96% by EOD | 96% break probability |
| "Noon is random" | Opposite extremes 74.3% | 74.3% structure |

---

## Configuration for Spectra Killer AI

### Recommended Integration Points

1. **SDEV-Based TP/SL Levels**
   ```python
   # Use timeframe-specific SDEV for dynamic targets
   tp_level = open_price + (1.0 * daily_sdev)
   sl_level = open_price - (1.5 * daily_sdev)
   ```

2. **Time-of-Day Filters**
   ```python
   # Reduce trading during low-probability windows
   if hour_segment == 'first_20_min':
       position_size *= 0.5  # Reduce size during wick formation
   ```

3. **9am Hour Bias Filter**
   ```python
   # Primary directional bias
   if hour_9am_close > hour_9am_open:
       daily_bias = 'BULLISH'  # 70% accuracy
       allowed_directions = ['LONG', 'CLOSE_SHORT']
   ```

4. **IB Breakout Logic**
   ```python
   # After 10:30am
   if time > '10:30' and IB_close_position == 'upper_half':
       breakout_bias = 'LONG'  # 81% probability
       target = IB_high + (IB_range * 0.5)
   ```

5. **Session-Based Adjustments**
   ```python
   # Integrate with existing SessionAnalyzer
   if london_engulfs_asia:
       ny_breakout_probability = 0.98
       position_multiplier = 1.5
   ```

### Data Requirements

To implement NQ Stats in Spectra Killer AI:

1. **Historical Calculation**:
   - Calculate instrument-specific SDEV values (20-year backtest)
   - Determine optimal time segments for target instrument (XAUUSD)

2. **Real-Time Tracking**:
   - Track session opens and ranges (RTH, pRTH, IB)
   - Monitor hourly opens and closes
   - Calculate current position relative to SDEV levels

3. **Probability Engine**:
   - Store probability lookup tables
   - Calculate real-time confluence scores
   - Generate probability-weighted signals

---

## References & Data Source

**Source**: nqstats.com
**Sample Periods**:
- Core metrics: 10 years (2015-2025)
- SDEV analysis: 20 years (2004-2024)
- Extended metrics: Various (2004-2025)

**Instruments Covered**: Nasdaq-100 E-Mini Futures (NQ)

**Applicability**: While derived from NQ data, these statistical principles can be adapted to other liquid futures and forex pairs (including XAUUSD) with proper instrument-specific calibration.

---

*Document Version: 1.0*
*Last Updated: November 2025*
*Integration Status: Ready for implementation in Spectra Killer AI v2.0*
