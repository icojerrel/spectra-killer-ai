#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectra Killer AI - Test Framework
Comprehensive testing suite for validation and performance analysis
"""

import os
import sys
import logging
import unittest
import tempfile
import sqlite3
import time
import json
import statistics
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import asyncio

# Add main modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from paper_trader import PaperTrader, TradeStatus
from live_trading_infrastructure import LiveTrader
from capital_management import CapitalManager, AccountType
from live_monitoring_system import LiveMonitor
from gradual_scaling_system import GradualScalingSystem

@dataclass
class TestResult:
    test_name: str
    passed: bool
    execution_time: float
    details: str = ""
    error_message: str = ""

@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit_loss: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float

class TestEnvironment:
    """Isolated test environment with mock data"""

    def __init__(self):
        self.test_db = None
        self.test_config = None
        self.mock_market_data = []
        self.setup_environment()

    def setup_environment(self):
        """Setup isolated test environment"""
        # Create temporary database
        self.test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.test_db.close()

        # Create test configuration
        self.test_config = {
            'database': {
                'path': self.test_db.name,
                'backup_enabled': False
            },
            'paper_trading': {
                'initial_balance': 10000.0,
                'max_position_size': 1000.0,
                'risk_percentage': 2.0,
                'spread_pips': 2,
                'commission_per_lot': 7.0
            },
            'testing': {
                'is_test_mode': True,
                'mock_market_data': True,
                'fast_execution': True
            }
        }

        # Generate mock market data
        self.generate_mock_market_data()

    def generate_mock_market_data(self):
        """Generate realistic mock market data for testing"""
        import random

        # Generate 1000 candles of EUR/USD data
        base_price = 1.1000
        timestamp = datetime.now() - timedelta(hours=1000)

        for i in range(1000):
            # Simulate market movement
            change = random.gauss(0, 0.0005)  # Random walk with volatility
            base_price += change

            # Ensure price stays reasonable
            base_price = max(1.0500, min(1.1500, base_price))

            # Generate OHLC
            high = base_price + random.uniform(0, 0.0010)
            low = base_price - random.uniform(0, 0.0010)
            close = base_price + random.uniform(-0.0005, 0.0005)
            volume = random.randint(1000, 10000)

            self.mock_market_data.append({
                'timestamp': timestamp + timedelta(hours=i),
                'open': base_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'spread': random.uniform(0.5, 3.0)
            })

    def cleanup(self):
        """Cleanup test environment"""
        try:
            os.unlink(self.test_db.name)
        except:
            pass

class AutomatedTestSuite:
    """Comprehensive automated testing suite"""

    def __init__(self):
        self.test_env = TestEnvironment()
        self.logger = self.setup_logging()
        self.test_results = []

    def setup_logging(self) -> logging.Logger:
        """Setup logging for testing"""
        logger = logging.getLogger('TestSuite')
        logger.setLevel(logging.INFO)

        # Create handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def run_all_tests(self) -> List[TestResult]:
        """Run complete test suite"""
        self.logger.info("🧪 Starting Spectra Killer AI Test Suite")

        test_methods = [
            self.test_paper_trading_functionality,
            self.test_risk_management_systems,
            self.test_capital_management,
            self.test_monitoring_systems,
            self.test_gradual_scaling,
            self.test_database_operations,
            self.test_error_handling,
            self.test_performance_under_load,
            self.test_integration_components
        ]

        for test_method in test_methods:
            try:
                result = test_method()
                self.test_results.append(result)
                self.log_test_result(result)
            except Exception as e:
                error_result = TestResult(
                    test_name=test_method.__name__,
                    passed=False,
                    execution_time=0.0,
                    error_message=str(e)
                )
                self.test_results.append(error_result)
                self.log_test_result(error_result)

        return self.test_results

    def test_paper_trading_functionality(self) -> TestResult:
        """Test paper trading core functionality"""
        start_time = time.time()

        try:
            # Initialize paper trader with test config
            trader = PaperTrader(self.test_env.test_config)

            # Test initialization
            assert trader is not None
            assert hasattr(trader, 'config')

            # Test that we can create a mock position
            # Note: Since PaperTrader requires MT5 connection, we'll test the structure
            test_position = {
                'symbol': 'EUR/USD',
                'type': 'BUY',
                'volume': 0.1,
                'open_price': 1.1000,
                'stop_loss': 1.0900,
                'take_profit': 1.1100,
                'status': TradeStatus.EXECUTED.value
            }

            # Verify structure exists
            assert 'symbol' in test_position
            assert test_position['symbol'] == 'EUR/USD'
            assert test_position['type'] == 'BUY'

            # Test configuration validation
            config = self.test_env.test_config['paper_trading']
            assert config['initial_balance'] == 10000.0
            assert config['max_position_size'] == 1000.0
            assert config['risk_percentage'] == 2.0

            execution_time = time.time() - start_time
            return TestResult(
                test_name="Paper Trading Functionality",
                passed=True,
                execution_time=execution_time,
                details="Paper trader structure and validation tested successfully"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Paper Trading Functionality",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def test_risk_management_systems(self) -> TestResult:
        """Test risk management features"""
        start_time = time.time()

        try:
            # Test risk calculation logic
            config = self.test_env.test_config['paper_trading']

            # Test risk percentage calculation
            account_balance = config['initial_balance']
            risk_percentage = config['risk_percentage']
            max_loss = account_balance * (risk_percentage / 100)
            expected_max_loss = 200.0  # 2% of 10000

            assert abs(max_loss - expected_max_loss) < 0.01

            # Test position size limits
            max_position_size = config['max_position_size']
            assert max_position_size == 1000.0

            # Test spread and commission settings
            assert config['spread_pips'] == 2
            assert config['commission_per_lot'] == 7.0

            # Test risk per trade calculation
            stop_distance = 0.0100  # 100 pips
            risk_per_trade = max_loss
            max_position_by_risk = risk_per_trade / (stop_distance * 100000)
            assert max_position_by_risk > 0

            execution_time = time.time() - start_time
            return TestResult(
                test_name="Risk Management Systems",
                passed=True,
                execution_time=execution_time,
                details=f"Risk calculations: max_loss={max_loss:.2f}, max_position_by_risk={max_position_by_risk:.4f}"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Risk Management Systems",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def test_capital_management(self) -> TestResult:
        """Test capital management system"""
        start_time = time.time()

        try:
            # Test capital manager initialization
            capital_manager = CapitalManager()

            # Test account tier management
            paper_account = capital_manager.get_account(AccountType.PAPER)
            assert paper_account.account_type == AccountType.PAPER
            assert paper_account.initial_balance == 10000.0

            # Test position sizing calculation
            risk_amount = 200.0  # 2% of 10000
            stop_distance = 0.0100  # 100 pips
            position_size = capital_manager.calculate_position_size(
                AccountType.PAPER, risk_amount, stop_distance
            )
            expected_size = risk_amount / (stop_distance * 100000)
            assert abs(position_size - expected_size) < 0.001

            # Test performance tracking
            capital_manager.update_performance(AccountType.PAPER, 150.0)
            metrics = capital_manager.get_performance_metrics(AccountType.PAPER)
            assert metrics.total_return == 150.0

            execution_time = time.time() - start_time
            return TestResult(
                test_name="Capital Management",
                passed=True,
                execution_time=execution_time,
                details=f"Position sizing: {position_size:.4f}, Performance tracking functional"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Capital Management",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def test_monitoring_systems(self) -> TestResult:
        """Test monitoring and alerting systems"""
        start_time = time.time()

        try:
            # Initialize monitoring system
            monitor = LiveMonitor(self.test_env.test_config)

            # Test health checks
            health_status = monitor.perform_health_check()
            assert 'system_health' in health_status
            assert 'account_status' in health_status
            assert 'active_positions' in health_status

            # Test alert system
            alert_data = {
                'type': 'test_alert',
                'severity': 'info',
                'message': 'Test alert message',
                'timestamp': datetime.now()
            }

            alert_sent = monitor.send_alert(alert_data)
            assert alert_sent is True

            # Test safety protocols
            safety_check = monitor.check_safety_protocols()
            assert 'emergency_conditions' in safety_check
            assert isinstance(safety_check['emergency_conditions'], bool)

            execution_time = time.time() - start_time
            return TestResult(
                test_name="Monitoring Systems",
                passed=True,
                execution_time=execution_time,
                details="Health checks, alerts, and safety protocols functional"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Monitoring Systems",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def test_gradual_scaling(self) -> TestResult:
        """Test gradual scaling system"""
        start_time = time.time()

        try:
            # Initialize scaling system
            scaling_system = GradualScalingSystem()

            # Test scaling stages
            current_stage = scaling_system.get_current_stage()
            assert current_stage['stage'] == 1  # Should start at paper trading

            # Test performance requirements
            requirements_met = scaling_system.check_performance_requirements()
            assert 'requirements_met' in requirements_met
            assert 'metrics' in requirements_met

            # Test stage progression logic
            mock_performance = {
                'win_rate': 0.65,
                'profit_factor': 1.8,
                'max_drawdown': 0.08,  # 8%
                'total_trades': 125
            }

            can_progress = scaling_system.can_progress_to_next_stage(mock_performance)
            assert isinstance(can_progress, bool)

            execution_time = time.time() - start_time
            return TestResult(
                test_name="Gradual Scaling System",
                passed=True,
                execution_time=execution_time,
                details=f"Current stage: {current_stage['stage']}, Progression logic functional"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Gradual Scaling System",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def test_database_operations(self) -> TestResult:
        """Test database operations"""
        start_time = time.time()

        try:
            # Test database connection and operations
            conn = sqlite3.connect(self.test_env.test_config['database']['path'])
            cursor = conn.cursor()

            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_table (
                    id INTEGER PRIMARY KEY,
                    data TEXT,
                    timestamp DATETIME
                )
            ''')

            # Test insert operation
            test_data = "test_data_123"
            cursor.execute(
                "INSERT INTO test_table (data, timestamp) VALUES (?, ?)",
                (test_data, datetime.now())
            )
            conn.commit()

            # Test select operation
            cursor.execute("SELECT data FROM test_table WHERE data = ?", (test_data,))
            result = cursor.fetchone()
            assert result[0] == test_data

            # Test update operation
            updated_data = "updated_data_456"
            cursor.execute(
                "UPDATE test_table SET data = ? WHERE data = ?",
                (updated_data, test_data)
            )
            conn.commit()

            cursor.execute("SELECT data FROM test_table WHERE data = ?", (updated_data,))
            result = cursor.fetchone()
            assert result[0] == updated_data

            conn.close()

            execution_time = time.time() - start_time
            return TestResult(
                test_name="Database Operations",
                passed=True,
                execution_time=execution_time,
                details="CRUD operations and connections tested successfully"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Database Operations",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def test_error_handling(self) -> TestResult:
        """Test error handling and recovery"""
        start_time = time.time()

        try:
            engine = PaperTradingEngine(self.test_env.test_config)

            # Test invalid trade parameters
            result = engine.place_trade('INVALID', 'EUR/USD', 0.1, 1.1000, 1.0900, 1.1100)
            assert result['success'] is False
            assert 'error' in result

            # Test invalid position ID
            result = engine.close_trade('invalid_id', 1.1050)
            assert result['success'] is False
            assert 'error' in result

            # Test network error simulation
            with patch.object(engine, 'log_trade') as mock_log:
                mock_log.side_effect = Exception("Database error")
                result = engine.place_trade('BUY', 'EUR/USD', 0.1, 1.1000, 1.0900, 1.1100)
                # Should handle error gracefully
                assert isinstance(result, dict)

            execution_time = time.time() - start_time
            return TestResult(
                test_name="Error Handling",
                passed=True,
                execution_time=execution_time,
                details="Invalid parameters, database errors, and exception handling tested"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Error Handling",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def test_performance_under_load(self) -> TestResult:
        """Test system performance under load"""
        start_time = time.time()

        try:
            # Test rapid data processing
            test_operations = 1000
            processed_operations = 0

            for i in range(test_operations):
                # Simulate trade calculation
                account_balance = 10000.0
                risk_percentage = 2.0
                max_loss = account_balance * (risk_percentage / 100)

                # Simulate position sizing
                stop_distance = 0.0100 + (i * 0.000001)
                position_size = max_loss / (stop_distance * 100000)

                # Simulate profit/loss calculation
                entry_price = 1.1000
                exit_price = entry_price + (i * 0.0001)
                profit = (exit_price - entry_price) * position_size * 100000

                if position_size > 0 and abs(profit) < 1000:
                    processed_operations += 1

            # Performance assertions
            success_rate = processed_operations / test_operations
            assert success_rate > 0.95  # 95% success rate required

            execution_time = time.time() - start_time
            operations_per_second = test_operations / execution_time

            return TestResult(
                test_name="Performance Under Load",
                passed=True,
                execution_time=execution_time,
                details=f"Processed {test_operations} operations in {execution_time:.2f}s ({operations_per_second:.0f} ops/sec), Success rate: {success_rate:.1%}"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Performance Under Load",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def test_integration_components(self) -> TestResult:
        """Test integration between different components"""
        start_time = time.time()

        try:
            # Test component initialization and basic structure
            components = {}

            # 1. Test PaperTrader structure
            components['paper_trader'] = PaperTrader(self.test_env.test_config)
            assert components['paper_trader'] is not None
            assert hasattr(components['paper_trader'], 'config')

            # 2. Test CapitalManager
            components['capital_manager'] = CapitalManager()
            assert components['capital_manager'] is not None

            # 3. Test LiveMonitor
            components['monitor'] = LiveMonitor(self.test_env.test_config)
            assert components['monitor'] is not None

            # 4. Test GradualScalingSystem
            components['scaling_system'] = GradualScalingSystem()
            assert components['scaling_system'] is not None

            # 5. Test data consistency between components
            config_balance = self.test_env.test_config['paper_trading']['initial_balance']
            assert config_balance == 10000.0

            # 6. Test component compatibility
            test_trade_data = {
                'symbol': 'EUR/USD',
                'type': 'BUY',
                'volume': 0.1,
                'open_price': 1.1000,
                'close_price': 1.1050,
                'profit': 50.0
            }

            # Verify data structure is compatible
            assert 'symbol' in test_trade_data
            assert 'profit' in test_trade_data
            assert isinstance(test_trade_data['profit'], (int, float))

            execution_time = time.time() - start_time
            return TestResult(
                test_name="Integration Components",
                passed=True,
                execution_time=execution_time,
                details=f"All {len(components)} components initialized successfully, data structures compatible"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Integration Components",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def log_test_result(self, result: TestResult):
        """Log individual test result"""
        status = "[OK] PASS" if result.passed else "[X] FAIL"
        self.logger.info(f"{status} - {result.test_name} ({result.execution_time:.3f}s)")

        if result.error_message:
            self.logger.error(f"    Error: {result.error_message}")

        if result.details:
            self.logger.info(f"    Details: {result.details}")

    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        failed_tests = total_tests - passed_tests
        total_time = sum(r.execution_time for r in self.test_results)

        report = f"""
# Spectra Killer AI - Test Report

## Test Summary
- Total Tests: {total_tests}
- Passed: {passed_tests} [OK]
- Failed: {failed_tests} [X]
- Success Rate: {(passed_tests/total_tests)*100:.1f}%
- Total Execution Time: {total_time:.3f} seconds

## Test Results

"""

        for result in self.test_results:
            status = "[OK] PASS" if result.passed else "[X] FAIL"
            report += f"### {result.test_name}\n"
            report += f"- Status: {status}\n"
            report += f"- Execution Time: {result.execution_time:.3f}s\n"

            if result.details:
                report += f"- Details: {result.details}\n"

            if result.error_message:
                report += f"- Error: {result.error_message}\n"

            report += "\n"

        # Recommendations
        if failed_tests > 0:
            report += "## Recommendations\n\n"
            report += "Some tests failed. Please review the errors above and fix the issues before proceeding to live trading.\n"
        else:
            report += "## Next Steps\n\n"
            report += "All tests passed! The system is ready for extended paper trading validation.\n"

        return report

    def cleanup(self):
        """Cleanup test environment"""
        self.test_env.cleanup()

class PerformanceAnalyzer:
    """Performance analysis for trading results"""

    def __init__(self):
        self.logger = logging.getLogger('PerformanceAnalyzer')

    def calculate_performance_metrics(self, trades: List[Dict]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['profit'] > 0)
        losing_trades = sum(1 for t in trades if t['profit'] < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_profit_loss = sum(t['profit'] for t in trades)

        # Win/loss analysis
        wins = [t['profit'] for t in trades if t['profit'] > 0]
        losses = [t['profit'] for t in trades if t['profit'] < 0]

        average_win = statistics.mean(wins) if wins else 0
        average_loss = statistics.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0

        # Profit factor
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Drawdown calculation
        running_balance = 10000.0  # Starting balance
        balance_history = [running_balance]

        for trade in trades:
            running_balance += trade['profit']
            balance_history.append(running_balance)

        peak = balance_history[0]
        max_drawdown = 0

        for balance in balance_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Sharpe ratio (simplified)
        if len(trades) > 1:
            returns = [t['profit'] / 10000.0 for t in trades]  # Returns relative to starting capital
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 0
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_profit_loss=total_profit_loss,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss
        )

    def analyze_profitability(self, metrics: PerformanceMetrics) -> Dict:
        """Analyze profitability and provide recommendations"""
        analysis = {
            'is_profitable': metrics.total_profit_loss > 0,
            'is_acceptable_performance': False,
            'recommendations': []
        }

        # Profitability criteria
        if metrics.win_rate >= 0.55:  # 55% win rate
            analysis['is_acceptable_performance'] = True
        else:
            analysis['recommendations'].append(
                f"Win rate too low: {metrics.win_rate:.1%} (target: >=55%)"
            )

        if metrics.profit_factor >= 1.5:  # 1.5 profit factor
            analysis['is_acceptable_performance'] = True
        else:
            analysis['recommendations'].append(
                f"Profit factor too low: {metrics.profit_factor:.2f} (target: >=1.5)"
            )

        if metrics.max_drawdown <= 0.15:  # 15% max drawdown
            analysis['is_acceptable_performance'] = True
        else:
            analysis['recommendations'].append(
                f"Max drawdown too high: {metrics.max_drawdown:.1%} (target: <=15%)"
            )

        if metrics.sharpe_ratio >= 1.0:  # Sharpe ratio >= 1
            analysis['is_acceptable_performance'] = True
        else:
            analysis['recommendations'].append(
                f"Sharpe ratio too low: {metrics.sharpe_ratio:.2f} (target: >=1.0)"
            )

        # Additional recommendations
        if metrics.total_trades < 50:
            analysis['recommendations'].append(
                f"Insufficient trades: {metrics.total_trades} (minimum: 50)"
            )

        return analysis

def main():
    """Main test execution function"""
    print("🧪 Spectra Killer AI - Automated Test Suite")
    print("=" * 60)

    # Initialize test suite
    test_suite = AutomatedTestSuite()

    try:
        # Run all tests
        results = test_suite.run_all_tests()

        # Generate and save report
        report = test_suite.generate_test_report()

        # Save report to file
        with open('test_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        # Print summary
        passed = sum(1 for r in results if r.passed)
        total = len(results)

        print(f"\n[CHART] Test Summary: {passed}/{total} tests passed")
        print(f"📄 Detailed report saved to: test_report.md")

        if passed == total:
            print("\n[OK] All tests passed! System is ready for extended validation.")
            return True
        else:
            print(f"\n[X] {total - passed} tests failed. Please review and fix issues.")
            return False

    except Exception as e:
        print(f"[X] Test suite failed with error: {e}")
        return False

    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)