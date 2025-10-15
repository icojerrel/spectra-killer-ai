# Spectra Killer AI Trading Bot - Risk Management Integration Test
# Dit bestand test de integratie tussen alle risk management componenten

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spectra_killer_ai.core.risk_manager import RiskManager
from spectra_killer_ai.config import RISK_MANAGEMENT, LOGGING


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_risk_manager_initialization():
    """Test risk manager initialisatie."""
    print("\n" + "="*50)
    print("TEST 1: Risk Manager Initialisatie")
    print("="*50)
    
    try:
        # Creëer risk manager
        risk_manager = RiskManager()
        
        # Initialiseer
        success = await risk_manager.initialize()
        
        if success:
            print("[OK] Risk Manager succesvol geïnitialiseerd")
            print(f"   Account balance: ${risk_manager.account_balance:.2f}")
            print(f"   Trading enabled: {risk_manager.trading_enabled}")
            print(f"   Max risk per trade: {risk_manager.max_risk_per_trade:.2%}")
            return True
        else:
            print("[X] Risk Manager initialisatie mislukt")
            return False
            
    except Exception as e:
        print(f"[X] Fout in initialisatie test: {str(e)}")
        return False


async def test_trade_risk_assessment():
    """Test trade risico beoordeling."""
    print("\n" + "="*50)
    print("TEST 2: Trade Risico Beoordeling")
    print("="*50)
    
    try:
        # Creëer en initialiseer risk manager
        risk_manager = RiskManager()
        await risk_manager.initialize()
        
        # Test signalen
        test_signals = [
            {
                'signal': 'BUY',
                'confidence': 0.8,
                'price': 1950.0,
                'atr': 20.0
            },
            {
                'signal': 'SELL',
                'confidence': 0.6,
                'price': 1960.0,
                'atr': 25.0
            },
            {
                'signal': 'BUY',
                'confidence': 0.4,  # Lage confidence
                'price': 1955.0,
                'atr': 15.0
            }
        ]
        
        for i, signal in enumerate(test_signals, 1):
            print(f"\nTest signaal {i}: {signal['signal']} @ ${signal['price']:.2f} (confidence: {signal['confidence']:.2f})")
            
            # Beoordeel risico
            assessment = await risk_manager.assess_trade_risk(signal)
            
            print(f"   Resultaat: {'GOEDGEKEURD' if assessment.is_approved else 'AFGEKEURD'}")
            print(f"   Positiegrootte: {assessment.position_size:.2f} lots")
            print(f"   Risico: ${assessment.risk_amount:.2f} ({assessment.risk_percent:.2%})")
            print(f"   Stop Loss: ${assessment.stop_loss:.2f}")
            print(f"   Take Profit: ${assessment.take_profit:.2f}")
            print(f"   Risk/Reward Ratio: {assessment.risk_reward_ratio:.2f}:1")
            
            if assessment.reasons:
                print(f"   Redenen:")
                for reason in assessment.reasons:
                    print(f"     - {reason}")
        
        return True
        
    except Exception as e:
        print(f"[X] Fout in trade risico beoordeling test: {str(e)}")
        return False


async def test_position_sizing():
    """Test position sizing methodes."""
    print("\n" + "="*50)
    print("TEST 3: Position Sizing Methodes")
    print("="*50)
    
    try:
        from spectra_killer_ai.utils.risk_calculations import RiskCalculations
        
        risk_calcs = RiskCalculations()
        account_balance = 10000.0
        entry_price = 1950.0
        stop_loss = 1940.0
        
        print(f"Account balance: ${account_balance:.2f}")
        print(f"Entry price: ${entry_price:.2f}")
        print(f"Stop loss: ${stop_loss:.2f}")
        print(f"Risk per trade: 2% = ${account_balance * 0.02:.2f}")
        print()
        
        # Test verschillende methodes
        methods = ['fixed_fractional', 'volatility_based', 'kelly_criterion']
        
        for method in methods:
            print(f"Testing {method} method:")
            
            if method == 'fixed_fractional':
                position_size = risk_calcs.calculate_position_size_fixed_fractional(
                    account_balance, 0.02, entry_price, stop_loss
                )
            elif method == 'volatility_based':
                atr = 20.0
                position_size = risk_calcs.calculate_position_size_volatility_based(
                    account_balance, 0.02, entry_price, stop_loss, atr
                )
            elif method == 'kelly_criterion':
                win_rate = 0.6
                avg_win = 100.0
                avg_loss = 50.0
                position_size = risk_calcs.calculate_position_size_kelly_criterion(
                    win_rate, avg_win, avg_loss, account_balance, entry_price, stop_loss
                )
            
            print(f"   Position size: {position_size:.2f} lots")
            print(f"   Risk amount: ${position_size * abs(entry_price - stop_loss) * 100:.2f}")
            print()
        
        return True
        
    except Exception as e:
        print(f"[X] Fout in position sizing test: {str(e)}")
        return False


async def test_risk_monitoring():
    """Test risico monitoring."""
    print("\n" + "="*50)
    print("TEST 4: Risico Monitoring")
    print("="*50)
    
    try:
        # Creëer en initialiseer risk manager
        risk_manager = RiskManager()
        await risk_manager.initialize()
        
        # Simuleer posities
        test_positions = {
            'pos_1': {
                'symbol': 'XAUUSD',
                'type': 'BUY',
                'volume': 0.1,
                'entry_price': 1950.0,
                'current_price': 1955.0,
                'pnl': 50.0,
                'value': 19550.0,
                'risk_amount': 200.0,
                'open_time': datetime.now()
            },
            'pos_2': {
                'symbol': 'XAUUSD',
                'type': 'SELL',
                'volume': 0.05,
                'entry_price': 1960.0,
                'current_price': 1958.0,
                'pnl': 10.0,
                'value': 9790.0,
                'risk_amount': 100.0,
                'open_time': datetime.now()
            }
        }
        
        # Voeg posities toe
        for pos_id, position in test_positions.items():
            risk_manager.add_position(pos_id, position)
        
        # Update account equity
        risk_manager.update_account_equity(10160.0)  # +$160 P&L
        
        # Monitor portfolio risico
        portfolio_risk = await risk_manager.monitor_portfolio_risk()
        
        print("Portfolio Risico Monitoring Resultaten:")
        print(f"   Account balance: ${portfolio_risk.get('account_balance', 0):.2f}")
        print(f"   Account equity: ${portfolio_risk.get('account_equity', 0):.2f}")
        print(f"   Total risk exposure: ${portfolio_risk.get('total_risk_exposure', 0):.2f}")
        print(f"   Daily risk used: {portfolio_risk.get('daily_risk_used', 0):.2%}")
        print(f"   Open positions: {portfolio_risk.get('open_positions', 0)}")
        print(f"   Current drawdown: {portfolio_risk.get('current_drawdown', 0):.2f}%")
        print(f"   Active alerts: {portfolio_risk.get('active_alerts', 0)}")
        
        # Get dashboard data
        dashboard_data = risk_manager.get_dashboard_data()
        print(f"\nDashboard Data:")
        print(f"   Risk manager status: {'Initialised' if risk_manager.is_initialized else 'Not initialised'}")
        print(f"   Position sizing metrics: {len(dashboard_data.get('position_sizer', {}))} metrics")
        print(f"   Risk monitor data: {len(dashboard_data.get('risk_monitor', {}))} categories")
        
        return True
        
    except Exception as e:
        print(f"[X] Fout in risico monitoring test: {str(e)}")
        return False


async def test_risk_calculations():
    """Test risico berekeningen."""
    print("\n" + "="*50)
    print("TEST 5: Risico Berekeningen")
    print("="*50)
    
    try:
        from spectra_killer_ai.utils.risk_calculations import RiskCalculations
        import pandas as pd
        import numpy as np
        
        risk_calcs = RiskCalculations()
        
        # Test stop loss/take profit berekeningen
        entry_price = 1950.0
        
        print(f"Entry price: ${entry_price:.2f}")
        
        # Stop loss methodes
        sl_fixed = risk_calcs.calculate_stop_loss(entry_price, "fixed_pips", pips=50)
        sl_percentage = risk_calcs.calculate_stop_loss(entry_price, "percentage", percentage=1.0)
        sl_atr = risk_calcs.calculate_stop_loss(entry_price, "atr", atr=20.0)
        
        print(f"\nStop Loss berekeningen:")
        print(f"   Fixed pips (50): ${sl_fixed:.2f}")
        print(f"   Percentage (1%): ${sl_percentage:.2f}")
        print(f"   ATR-based (2x ATR): ${sl_atr:.2f}")
        
        # Take profit
        tp = risk_calcs.calculate_take_profit(entry_price, sl_fixed, 2.0)
        print(f"   Take profit (2:1 RR): ${tp:.2f}")
        
        # Risk metrics
        print(f"\nRisk Metrics:")
        
        # Genereer sample returns data
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))  # 100 days of returns
        
        sharpe = risk_calcs.calculate_sharpe_ratio(returns)
        sortino = risk_calcs.calculate_sortino_ratio(returns)
        
        print(f"   Sharpe ratio: {sharpe:.3f}")
        print(f"   Sortino ratio: {sortino:.3f}")
        
        # Equity curve voor drawdown
        equity_curve = pd.Series(10000 * (1 + returns).cumprod())
        drawdown_stats = risk_calcs.calculate_max_drawdown(equity_curve)
        
        print(f"   Max drawdown: ${drawdown_stats['max_drawdown']:.2f} ({drawdown_stats['max_drawdown_percent']:.2f}%)")
        
        # VaR en CVaR
        var_95 = risk_calcs.calculate_var(returns, 0.95)
        cvar_95 = risk_calcs.calculate_cvar(returns, 0.95)
        
        print(f"   VaR (95%): {var_95:.4f} ({var_95*100:.2f}%)")
        print(f"   CVaR (95%): {cvar_95:.4f} ({cvar_95*100:.2f}%)")
        
        # Risk/reward ratio validatie
        is_valid_rr = risk_calcs.validate_risk_reward_ratio(entry_price, sl_fixed, tp, 1.5)
        print(f"   Risk/Reward valid (min 1.5:1): {is_valid_rr}")
        
        return True
        
    except Exception as e:
        print(f"[X] Fout in risico berekeningen test: {str(e)}")
        return False


async def test_edge_cases():
    """Test edge cases en error handling."""
    print("\n" + "="*50)
    print("TEST 6: Edge Cases en Error Handling")
    print("="*50)
    
    try:
        risk_manager = RiskManager()
        await risk_manager.initialize()
        
        # Test 1: Ongeldig signaal
        print("Test 1: Ongeldig signaal")
        invalid_signal = {
            'signal': 'INVALID',
            'confidence': 1.5,  # Invalid confidence
            'price': -100  # Invalid price
        }
        
        assessment = await risk_manager.assess_trade_risk(invalid_signal)
        print(f"   Resultaat: {'GOEDGEKEURD' if assessment.is_approved else 'AFGEKEURD'}")
        print(f"   Redenen: {assessment.reasons}")
        
        # Test 2: Zero account balance
        print("\nTest 2: Zero account balance")
        risk_manager.update_account_balance(0.0)
        
        valid_signal = {
            'signal': 'BUY',
            'confidence': 0.8,
            'price': 1950.0
        }
        
        assessment = await risk_manager.assess_trade_risk(valid_signal)
        print(f"   Resultaat: {'GOEDGEKEURD' if assessment.is_approved else 'AFGEKEURD'}")
        print(f"   Redenen: {assessment.reasons}")
        
        # Test 3: Trading disabled
        print("\nTest 3: Trading disabled")
        risk_manager.update_account_balance(10000.0)  # Restore balance
        risk_manager.disable_trading()
        
        assessment = await risk_manager.assess_trade_risk(valid_signal)
        print(f"   Resultaat: {'GOEDGEKEURD' if assessment.is_approved else 'AFGEKEURD'}")
        print(f"   Redenen: {assessment.reasons}")
        
        # Test 4: Risk limieten overschreden
        print("\nTest 4: Risk limieten overschreden")
        risk_manager.enable_trading()
        
        # Simuleer hoge risico posities
        for i in range(10):  # Voeg veel posities toe
            risk_manager.daily_risk_used = 0.04  # 4% dagelijks risico gebruikt
            
        assessment = await risk_manager.assess_trade_risk(valid_signal)
        print(f"   Resultaat: {'GOEDGEKEURD' if assessment.is_approved else 'AFGEKEURD'}")
        print(f"   Redenen: {assessment.reasons}")
        
        return True
        
    except Exception as e:
        print(f"[X] Fout in edge cases test: {str(e)}")
        return False


async def main():
    """Hoofdfunctie voor alle tests."""
    print("Spectra Killer AI - Risk Management Integration Tests")
    print("=" * 60)
    
    # Voer alle tests uit
    tests = [
        ("Initialisatie", test_risk_manager_initialization),
        ("Trade Risico Beoordeling", test_trade_risk_assessment),
        ("Position Sizing", test_position_sizing),
        ("Risico Monitoring", test_risk_monitoring),
        ("Risico Berekeningen", test_risk_calculations),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[X] Test {test_name} mislukt met exception: {str(e)}")
            results.append((test_name, False))
    
    # Samenvatting
    print("\n" + "="*60)
    print("TEST SAMENVATTING")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "[OK] PASSED" if result else "[X] FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nTotaal: {passed}/{total} tests geslaagd ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n[PARTY] Alle tests succesvol! Risk management componenten zijn correct geïntegreerd.")
    else:
        print(f"\n[WARNING]  {total-passed} tests mislukt. Controleer de implementatie.")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(main())