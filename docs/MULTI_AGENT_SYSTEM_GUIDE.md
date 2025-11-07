# Multi-Agent NQ Stats Trading System

## Overview

The Multi-Agent NQ Stats Trading System consists of **8 specialized agents**, each focusing on ONE specific NQ Stats methodology. Together, they provide comprehensive market coverage with diversified probability-based edges.

---

## 🤖 The 8 Specialized Agents

### 1. SDEV Mean Reversion Agent
**Edge**: 70-95% probability at extremes
**Active**: 24/7
**Strategy**: Fade extremes, trade back to mean
**Position Size**: 10% base

**How It Works:**
- Monitors price distance from mean (session open)
- Activates when price reaches ±1.5σ or beyond
- Higher confidence at greater extremes (±2.0σ = 95% probability)
- Stop: ±2.0σ, Target: Mean

**Example:**
```
Price at +1.8σ → SHORT signal
- Probability: 85% reversion
- Entry: Current price
- Stop: +2.0σ level
- Target: Session open (mean)
```

---

### 2. Hour Segment Agent
**Edge**: 89% retracement in first 20 minutes
**Active**: Minutes 0-20 of each hour
**Strategy**: Trade retracements after sweep
**Position Size**: 12% base

**How It Works:**
- Active only during first 20 minutes of any hour
- Detects sweep of previous hour high/low
- Trades retracement back to current hour open
- Skips 9:00-9:30am (pre-market open volatility)

**Example:**
```
10:00am - Previous hour high swept
→ LONG signal for retracement
- Probability: 89%
- Target: 10:00am open
```

---

### 3. RTH Break Agent
**Edge**: 83.29% directional bias
**Active**: 9:30am-4:00pm (RTH)
**Strategy**: Trade WITH directional bias
**Position Size**: 15% base (strong edge)

**How It Works:**
- Checks RTH open vs previous RTH range
- If opens ABOVE pRTH high → BULLISH (83.29% won't break low)
- If opens BELOW pRTH low → BEARISH (83.29% won't break high)
- Trades WITH the bias all session

**Example:**
```
RTH opens at 15,100 (pRTH: 14,950-15,050)
→ LONG signal (opened above pRTH)
- Probability: 83.29%
- Stop: pRTH low (14,950)
- Trade upside all day
```

---

### 4. 9am Hour Continuation Agent (STRONGEST!)
**Edge**: 70% NY session continuation
**Active**: After 10:00am
**Strategy**: Trade WITH 9am hour direction
**Position Size**: 20% base (DOUBLE - strongest edge!)

**How It Works:**
- Waits for 9am hour (9:00-10:00am) to close
- Determines direction: GREEN or RED
- Trades WITH that direction for rest of NY session
- 70% probability of continuation (highest edge!)

**Example:**
```
9am hour closes GREEN (10:00 > 9:00)
→ LONG signal for entire NY session
- Probability: 70%
- Confidence: HIGH
- Position: 20% (double normal)
```

---

### 5. Initial Balance Agent
**Edge**: 74-81% directional break
**Active**: After 10:30am
**Strategy**: Trade IB breakout direction
**Position Size**: 13% base

**How It Works:**
- Waits for IB formation (9:30-10:30am)
- Checks IB close position (upper/lower half)
- Upper half close → 81% IB high breaks
- Lower half close → 74% IB low breaks
- Trades breakout direction

**Example:**
```
IB: High 15,100, Low 15,000, Close 15,075
→ Upper half close
→ LONG signal (81% IB high breaks)
- Target: IB high + IB range
```

---

### 6. Noon Curve Agent
**Edge**: 74.3% opposite extremes
**Active**: After 12:00pm
**Strategy**: Trade for opposite extreme
**Position Size**: 11% base

**How It Works:**
- Identifies AM extreme (8am-12pm high or low)
- Expects PM to set opposite extreme
- If AM = low → PM = high (74.3%)
- If AM = high → PM = low (74.3%)

**Example:**
```
AM set the LOW at 14,950
→ LONG signal for PM
- Probability: 74.3% PM sets HIGH
- Target: Above AM high
```

---

### 7. Session Pattern Agent
**Edge**: 73-98% depending on pattern
**Active**: During NY session
**Strategy**: Trade highest probability session break
**Position Size**: 15% base

**How It Works:**
- Analyzes Asian/London/NY relationships
- London Engulfs Asia → 98% NY breaks London
- Partial engulfs → 73-79% edges
- Trades highest probability pattern

**Example:**
```
London engulfed Asia overnight
→ LONG/SHORT signal (direction TBD)
- Probability: 98% NY breaks London range
- Highest edge pattern!
```

---

### 8. Morning Judas Agent
**Edge**: 64-70% continuation
**Active**: After 10:00am
**Strategy**: Trade WITH morning momentum
**Position Size**: 10% base

**How It Works:**
- Checks 9:40am vs 9:30am (Judas formation)
- UP Judas → 64% continuation to 10am
- DOWN Judas → 70% continuation to 10am
- Trades WITH the continuation (NOT reversal!)

**Example:**
```
9:40am > 9:30am (UP Judas)
→ LONG signal (continuation)
- Probability: 64-70%
- Myth busted: Continuation > Reversal!
```

---

## 🎯 Coordination Modes

The Agent Coordinator orchestrates all 8 agents using one of three modes:

### Mode 1: Best Signal
**Strategy**: Take highest probability signal
**Use Case**: Simple, conservative, one trade at a time
**Best For**: Beginners, small accounts

**How It Works:**
1. Collect all agent signals
2. Sort by probability
3. Take top signal
4. Execute single trade

**Example:**
```
Signals:
- 9am Agent: LONG (70%)
- RTH Agent: LONG (83.29%)
- IB Agent: LONG (74%)

→ Take RTH Agent (highest: 83.29%)
```

---

### Mode 2: Confluence Voting (Recommended)
**Strategy**: Combine agreeing signals, scale size with agreement
**Use Case**: Balanced risk/reward, confluence-based sizing
**Best For**: Most traders, medium accounts

**How It Works:**
1. Collect all signals
2. Count LONG vs SHORT
3. Choose majority direction
4. Scale position size with confluence

**Scaling:**
- 1 agent: 1.0x base size
- 2 agents: 1.25x base size
- 3 agents: 1.5x base size
- 4+ agents: 2.0x base size (max)

**Example:**
```
Signals:
- 9am Agent: LONG (70%)
- RTH Agent: LONG (83.29%)
- IB Agent: LONG (74%)
- Noon Agent: SHORT (74.3%)

→ 3 LONG, 1 SHORT
→ Direction: LONG
→ Confluence: 3
→ Position Size: 1.5x base (50% larger!)
```

---

### Mode 3: Portfolio Diversification
**Strategy**: Trade multiple agents simultaneously
**Use Case**: Maximum edge capture, risk distribution
**Best For**: Advanced traders, larger accounts

**How It Works:**
1. Filter high-quality signals (>70% probability)
2. Take top N signals (configurable, max 3)
3. Trade each agent independently
4. Total exposure = sum of all positions

**Example:**
```
High-Quality Signals:
- 9am Agent: LONG (70%) - 20% size
- RTH Agent: LONG (83.29%) - 15% size
- Session Agent: SHORT (95%) - 15% size

→ Trade all 3 simultaneously
→ Total exposure: 50% of capital
→ Diversified across 3 different edges
```

---

## 📊 Position Sizing

### Individual Agent Sizes (Base)
| Agent | Edge | Base Size | Reasoning |
|-------|------|-----------|-----------|
| 9am Hour | 70% | **20%** | Strongest edge - DOUBLE |
| RTH Break | 83.29% | 15% | Very high probability |
| Session Pattern | 73-98% | 15% | Can be extremely high |
| IB Breakout | 74-81% | 13% | Strong directional |
| Hour Segment | 89% | 12% | High retracement prob |
| Noon Curve | 74.3% | 11% | Solid AM/PM edge |
| SDEV Mean Rev | 70-95% | 10% | Varies by distance |
| Morning Judas | 64-70% | 10% | Good but not strongest |

### With Confluence (Mode 2)
- **2 Agents Agree**: 1.25x = 12.5% to 25%
- **3 Agents Agree**: 1.5x = 15% to 30%
- **4+ Agents Agree**: 2.0x = 20% to 40%

### Portfolio Mode (Mode 3)
- Max 3 simultaneous positions
- Total capped at 50% of capital
- Individual agents keep their base sizes

---

## ⚙️ Configuration

Complete configuration in `config/multi_agent_config.yaml`:

### Key Settings

```yaml
# Coordination
coordination_mode: "confluence_voting"
min_confluence: 2
max_concurrent_positions: 3

# Individual Agent Thresholds
agents:
  9am_agent:
    min_probability: 0.65  # Lower (best edge!)
    position_size_pct: 0.20  # Higher (strongest!)

  rth_agent:
    min_probability: 0.80
    position_size_pct: 0.15

  # ... etc for all 8 agents

# Portfolio Management
portfolio:
  total_capital: 10000
  max_total_exposure: 0.50  # Max 50%
  reserve_cash_pct: 0.20    # Keep 20% cash

# Risk Management
risk_management:
  max_daily_loss_pct: 0.03  # Stop at -3%
  use_dynamic_stops: true    # SDEV-based
```

---

## 🚀 Quick Start

### 1. Run Demo
```bash
python demo_scripts/multi_agent_trading_demo.py
```

### 2. Basic Usage
```python
from spectra_killer_ai.agents import AgentCoordinator
import yaml

# Load config
with open('config/multi_agent_config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize coordinator
coordinator = AgentCoordinator(config)

# Analyze and coordinate
results = coordinator.analyze_and_coordinate(data)

# Get recommendation
recommendation = results['coordinated_recommendation']

if recommendation['action'] != 'WAIT':
    print(f"Trade {recommendation['action']}")
    print(f"Size: {recommendation['position_size']*100:.1f}%")
    print(f"Confluence: {recommendation.get('confluence', 0)}")
```

### 3. Change Coordination Mode
```python
config['coordination_mode'] = 'portfolio_diversification'
coordinator = AgentCoordinator(config)
```

---

## 📈 Performance Expectations

### Theoretical Performance (Based on Historical Probabilities)

**Mode 1: Best Signal**
- Win Rate: 70-85% (taking only best)
- Trades/Day: 1-2
- Risk: Low (single position)

**Mode 2: Confluence Voting**
- Win Rate: 75-80% (confluence boost)
- Trades/Day: 2-4
- Risk: Medium (larger positions with confluence)

**Mode 3: Portfolio Diversification**
- Win Rate: 70-75% (average across agents)
- Trades/Day: 3-6
- Risk: Medium-High (multiple positions)

### Example Portfolio (Confluence Voting)

```
Capital: $10,000
Mode: Confluence Voting
Min Confluence: 2

Day 1:
09:15 - 2 agents agree (RTH + IB) → LONG 12.5%
11:00 - 3 agents agree (9am + RTH + IB) → LONG 15%
14:30 - 1 agent only (Noon) → WAIT

Total Exposure: 27.5% (within 50% max)
Positions: 2 (within 3 max)
```

---

## 🎓 Strategy Guide

### When to Use Each Mode

**Best Signal**:
- ✅ Starting out with system
- ✅ Small account (<$5,000)
- ✅ Conservative risk tolerance
- ✅ Learning phase
- ❌ Missing opportunities

**Confluence Voting** (Recommended):
- ✅ Most balanced approach
- ✅ Medium account ($5,000-$50,000)
- ✅ Moderate risk tolerance
- ✅ Scales with agreement
- ✅ Best risk/reward ratio

**Portfolio Diversification**:
- ✅ Maximum edge capture
- ✅ Large account (>$50,000)
- ✅ Higher risk tolerance
- ✅ Advanced traders
- ❌ Requires more monitoring

---

## 🔧 Customization

### Adjust Agent Parameters

```yaml
agents:
  9am_agent:
    enabled: true              # Turn on/off
    min_probability: 0.65      # Lower threshold
    position_size_pct: 0.20    # Larger size
    use_aggressive_targets: true
```

### Adjust Coordination

```yaml
min_confluence: 3              # Require 3 agents (more conservative)
max_concurrent_positions: 2    # Limit to 2 (reduce risk)
```

### Adjust Risk

```yaml
portfolio:
  max_total_exposure: 0.30     # Reduce to 30% max
  reserve_cash_pct: 0.40       # Keep more cash

risk_management:
  max_daily_loss_pct: 0.02     # Tighter daily stop
```

---

## 📊 Monitoring

### Agent Performance Tracking

```python
# Get performance report
report = coordinator.get_performance_report()

for agent_name, perf in report['agent_performance'].items():
    print(f"{agent_name}:")
    print(f"  Win Rate: {perf['win_rate']:.1f}%")
    print(f"  P&L: ${perf['pnl']:.2f}")
    print(f"  Signals: {perf['signals_generated']}")
```

### Real-Time Status

```python
# Get current agent statuses
statuses = coordinator._get_all_agent_statuses()

for status in statuses:
    if status['status'] == 'active':
        print(f"🟢 {status['name']} - ACTIVE")
```

---

## 🎯 Best Practices

### 1. Start Conservative
- Begin with **Best Signal** mode
- Use smaller position sizes (50% of defaults)
- Paper trade for 1 month

### 2. Gradual Scaling
- Move to **Confluence Voting** after successful testing
- Increase sizes gradually (10% per week)
- Monitor win rates closely

### 3. Risk Management
- Never exceed 50% total exposure
- Keep 20% cash reserve minimum
- Use daily loss limits (-3%)

### 4. Agent Monitoring
- Track individual agent performance
- Disable underperforming agents
- Recalibrate probabilities monthly

### 5. Instrument Adaptation
- Calibrate for your instrument (XAUUSD)
- Validate probabilities on historical data
- Adjust thresholds as needed

---

## 🚨 Common Pitfalls

### ❌ Over-Leverage
**Problem**: Trading all 8 agents at max size simultaneously
**Solution**: Use max_total_exposure = 50% limit

### ❌ Ignoring Confluence
**Problem**: Taking every signal regardless of agreement
**Solution**: Use min_confluence = 2 in voting mode

### ❌ Not Calibrating
**Problem**: Using NQ probabilities on XAUUSD without adjustment
**Solution**: See XAUUSD_CALIBRATION_GUIDE.md

### ❌ Chasing Performance
**Problem**: Increasing sizes after wins
**Solution**: Stick to position sizing rules

### ❌ Disabling Best Agents
**Problem**: Turning off 9am agent because it "trades too much"
**Solution**: Reduce size, don't disable (strongest edge!)

---

## 📚 Additional Resources

- **NQ Stats Complete Guide**: `docs/NQ_STATS_COMPLETE_GUIDE.md`
- **Integration Guide**: `docs/NQ_STATS_INTEGRATION.md`
- **Calibration Guide**: `docs/XAUUSD_CALIBRATION_GUIDE.md`
- **Configuration**: `config/multi_agent_config.yaml`
- **Demo Script**: `demo_scripts/multi_agent_trading_demo.py`

---

## 🎉 Summary

The Multi-Agent NQ Stats Trading System provides:

✅ **8 Specialized Edges** - Each agent focuses on one proven methodology
✅ **Diversified Risk** - Not dependent on single strategy
✅ **Flexible Coordination** - 3 modes for different trading styles
✅ **Probability-Based** - All decisions backed by 10-20 years data
✅ **Scalable** - Works with small or large accounts
✅ **Production-Ready** - Complete configuration and monitoring

**The system is designed to capture the best NQ Stats edges while managing risk through diversification and intelligent coordination.**

---

*Built for Spectra Killer AI v2.0*
*Based on NQ Stats research (2004-2025)*
*Last Updated: November 2025*
