#!/usr/bin/env python3
"""
Multi-Agent NQ Stats Trading System Demo

Demonstrates 8 specialized trading agents working together:
- Each agent focuses on one NQ Stats methodology
- Agent Coordinator orchestrates all agents
- Shows different coordination modes
- Tracks agent performance

Usage:
    python demo_scripts/multi_agent_trading_demo.py --mode confluence_voting
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import yaml
import importlib.util

# Load modules directly
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load NQ Stats analyzer
base_path = Path(__file__).parent.parent / 'src' / 'spectra_killer_ai'
analyzer_path = base_path / 'strategies' / 'advanced' / 'nq_stats_analyzer.py'
nq_stats = load_module("nq_stats_analyzer", analyzer_path)

# Load agents
agents_path = base_path / 'agents' / 'nq_stats_agents.py'
agents_module = load_module("nq_stats_agents", agents_path)

# Load coordinator
coordinator_path = base_path / 'agents' / 'agent_coordinator.py'
coordinator_module = load_module("agent_coordinator", coordinator_path)

AgentCoordinator = coordinator_module.AgentCoordinator


def generate_sample_data(days=3):
    """Generate realistic sample data"""
    print("📊 Generating sample market data...")

    end_time = datetime.now(pytz.timezone('US/Eastern'))
    start_time = end_time - timedelta(days=days)
    date_range = pd.date_range(start=start_time, end=end_time, freq='5min')

    np.random.seed(42)
    base_price = 15000

    returns = np.random.normal(0, 0.001, len(date_range))
    prices = base_price * (1 + returns).cumprod()

    data = pd.DataFrame(index=date_range)
    data['open'] = prices
    data['high'] = prices * (1 + np.abs(np.random.normal(0, 0.0005, len(date_range))))
    data['low'] = prices * (1 - np.abs(np.random.normal(0, 0.0005, len(date_range))))
    data['close'] = prices * (1 + np.random.normal(0, 0.0003, len(date_range)))
    data['volume'] = np.random.randint(100, 1000, len(date_range))

    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    print(f"✅ Generated {len(data)} bars")
    return data


def load_config():
    """Load multi-agent configuration"""
    config_path = Path(__file__).parent.parent / 'config' / 'multi_agent_config.yaml'

    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default minimal config
        return {
            'coordination_mode': 'confluence_voting',
            'nq_stats': {'sdev_values': {'daily': 1.376, 'hourly': 0.34}},
            'agents': {},
            'min_confluence': 2
        }


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_agent_statuses(statuses):
    """Print agent status table"""
    print("\n📋 AGENT STATUS:")
    print(f"{'Agent':<30} {'Status':<12} {'Trades':<8} {'Win Rate':<10} {'P&L'}")
    print("-" * 80)

    for status in statuses:
        name = status['name']
        state = status['status']
        trades = status['total_trades']
        win_rate = status['win_rate']
        pnl = status['pnl']

        # Status emoji
        emoji = "🟢" if state == "active" else "⚪" if state == "monitoring" else "⏸️"

        print(f"{emoji} {name:<28} {state:<12} {trades:<8} {win_rate:>6.1f}%   ${pnl:>7.2f}")


def print_signals(signals):
    """Print active signals"""
    if not signals:
        print("\n⏸️  No active agent signals at this time")
        return

    print(f"\n🎯 ACTIVE AGENT SIGNALS ({len(signals)} signals):")
    print("-" * 80)

    for i, signal in enumerate(signals, 1):
        print(f"\n{i}. {signal.agent_name}")
        print(f"   Signal: {signal.signal_type}")
        print(f"   Probability: {signal.probability*100:.1f}%")
        print(f"   Confidence: {signal.confidence*100:.1f}%")
        print(f"   Entry: ${signal.entry_price:.2f}")
        print(f"   Stop Loss: ${signal.stop_loss:.2f}" if signal.stop_loss else "   Stop Loss: N/A")
        print(f"   Take Profit: ${signal.take_profit:.2f}" if signal.take_profit else "   Take Profit: N/A")
        print(f"   Size: {signal.position_size*100:.1f}% of capital")
        print(f"   Reasoning: {signal.reasoning}")


def print_coordination(recommendation):
    """Print coordinated recommendation"""
    print_header("COORDINATED RECOMMENDATION")

    action = recommendation.get('action', 'WAIT')
    mode = recommendation.get('mode', 'unknown')

    print(f"\n🎯 Mode: {mode}")
    print(f"🎯 Action: {action}")

    if action == 'WAIT':
        print(f"⏸️  Reason: {recommendation.get('reason', 'No clear setup')}")
        return

    if action == 'PORTFOLIO':
        # Portfolio mode
        positions = recommendation.get('positions', [])
        print(f"\n📊 Portfolio Positions: {len(positions)}")

        for i, pos in enumerate(positions, 1):
            print(f"\n  Position {i}: {pos['agent']}")
            print(f"     Direction: {pos['direction']}")
            print(f"     Probability: {pos['probability']*100:.1f}%")
            print(f"     Entry: ${pos['entry']:.2f}")
            print(f"     Size: {pos['size']*100:.1f}%")

        total_size = recommendation.get('total_position_size', 0)
        print(f"\n  Total Portfolio Size: {total_size*100:.1f}% of capital")

    else:
        # Single direction (best_signal or confluence_voting)
        print(f"\n💰 Entry Price: ${recommendation.get('entry_price', 0):.2f}")
        print(f"🛑 Stop Loss: ${recommendation.get('stop_loss', 0):.2f}")
        print(f"🎯 Take Profit: ${recommendation.get('take_profit', 0):.2f}")
        print(f"📊 Position Size: {recommendation.get('position_size', 0)*100:.1f}% of capital")

        if 'probability' in recommendation:
            print(f"📈 Probability: {recommendation['probability']*100:.1f}%")

        if 'confidence' in recommendation:
            print(f"🎲 Confidence: {recommendation['confidence']*100:.1f}%")

        if 'confluence' in recommendation:
            print(f"🤝 Confluence: {recommendation['confluence']} agents agree")

        if 'reasoning' in recommendation:
            print(f"\n💡 Reasoning: {recommendation['reasoning']}")

        if 'agreeing_agents' in recommendation:
            print(f"\n✅ Agreeing Agents:")
            for agent in recommendation['agreeing_agents']:
                print(f"   - {agent}")


def main():
    """Main demo function"""
    print_header("MULTI-AGENT NQ STATS TRADING SYSTEM DEMO")
    print("\n🤖 8 Specialized Trading Agents")
    print("📊 Probability-Based Decision Making")
    print("🎯 Multiple Coordination Modes")
    print("\n")

    # Load configuration
    print("⚙️  Loading configuration...")
    config = load_config()
    mode = config.get('coordination_mode', 'confluence_voting')
    print(f"✅ Coordination mode: {mode}")
    print(f"✅ Min confluence: {config.get('min_confluence', 2)}")
    print()

    # Generate data
    data = generate_sample_data(days=3)

    # Initialize coordinator
    print("🚀 Initializing Agent Coordinator...")
    coordinator = AgentCoordinator(config)
    print(f"✅ Initialized {len(coordinator.agents)} specialized agents\n")

    # List agents
    print("🤖 SPECIALIZED AGENTS:")
    for i, agent in enumerate(coordinator.agents, 1):
        print(f"   {i}. {agent.name}")
    print()

    # Run analysis
    current_time = datetime.now(pytz.timezone('US/Eastern'))
    print(f"🕐 Analysis Time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")

    print("🔍 Running multi-agent analysis...")
    results = coordinator.analyze_and_coordinate(data, current_time)
    print("✅ Analysis complete!\n")

    # Display results
    print_header("ANALYSIS RESULTS")

    # Agent statuses
    print_agent_statuses(results['agent_statuses'])

    # Active agents
    active = results['active_agents']
    print(f"\n🟢 Active Agents: {len(active)}")
    if active:
        for agent in active:
            print(f"   - {agent}")
    else:
        print("   (None - waiting for setup conditions)")

    # Individual agent signals
    signals = results['agent_signals']
    print_signals(signals)

    # Coordinated recommendation
    recommendation = results['coordinated_recommendation']
    print_coordination(recommendation)

    # Portfolio allocation
    print_header("PORTFOLIO ALLOCATION")
    allocation = results['portfolio_allocation']
    print(f"\n📊 Total Allocation: {allocation['total_allocation']*100:.1f}% of capital")
    print(f"📊 Active Agents: {allocation['agents_count']}")

    if allocation['allocations']:
        print("\n💰 Individual Allocations:")
        for alloc in allocation['allocations']:
            print(f"   {alloc['agent']:<30} {alloc['allocation']*100:>6.1f}%  (prob={alloc['probability']*100:.0f}%)")

    # Summary
    print_header("SUMMARY")
    print(f"✅ Analysis completed successfully")
    print(f"📊 {len(results['agent_signals'])} signals generated")
    print(f"🎯 Coordination mode: {mode}")
    print(f"💡 Recommendation: {recommendation.get('action', 'WAIT')}")

    if recommendation.get('action') != 'WAIT':
        if recommendation.get('action') == 'PORTFOLIO':
            print(f"📊 Portfolio size: {len(recommendation.get('positions', []))} positions")
        else:
            print(f"📊 Position size: {recommendation.get('position_size', 0)*100:.1f}%")

    # Performance report
    print_header("AGENT PERFORMANCE (Current Session)")
    perf_report = coordinator.get_performance_report()

    print(f"\n📊 Total Agents: {perf_report['total_agents']}")
    print(f"📊 Signals Generated: {sum(p['signals_generated'] for p in perf_report['agent_performance'].values())}")

    # Tips
    print_header("COORDINATION MODES")
    print("\n1. best_signal - Take highest probability signal")
    print("   • Simple and conservative")
    print("   • One trade at a time")
    print("   • Good for beginners")

    print("\n2. confluence_voting - Combine agreeing signals")
    print("   • Balanced approach")
    print("   • Scales size with agreement")
    print("   • Currently active mode ✅")

    print("\n3. portfolio_diversification - Trade multiple simultaneously")
    print("   • Advanced strategy")
    print("   • Maximum edge capture")
    print("   • Requires more capital")

    print_header("NEXT STEPS")
    print("\n1. Review agent signals and recommendations")
    print("2. Test different coordination modes")
    print("3. Backtest on historical data")
    print("4. Calibrate for your instrument (XAUUSD, etc.)")
    print("5. Forward test on demo account")
    print("6. Scale to live trading")

    print("\n🎉 Demo complete!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
