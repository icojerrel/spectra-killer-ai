#!/usr/bin/env python3
# Spectra Killer AI - AI Integration Test
# Quick test van de geïntegreerde AI systemen

import asyncio
import sys
import json
from datetime import datetime

sys.path.insert(0, '.')

async def test_background_agents():
    """Test Background Agents System"""
    print("Testing Background Agents...")

    try:
        from background_agents.agent_manager import AgentManager, AgentConfig, AgentType, LLMProviderType

        # Simple config for testing
        config = {
            "llm_providers": {
                "anthropic": {
                    "api_key": "test-key",
                    "model": "claude-3-sonnet-20240229"
                }
            }
        }

        manager = AgentManager(config)
        await manager.initialize()

        print("Background Agents initialized successfully")
        return True

    except Exception as e:
        print(f"Background Agents test failed: {e}")
        return False

async def test_ml_engine():
    """Test ML Trading Engine"""
    print("Testing ML Engine...")

    try:
        from ml.intelligent_trading_engine import IntelligentTradingEngine

        config = {
            "ml_config": {
                "model_type": "gradient_boosting",
                "lookback_period": 20
            }
        }

        engine = IntelligentTradingEngine(config)
        await engine.initialize()

        # Test prediction
        test_features = {
            "rsi": 45,
            "ema_9": 2005,
            "ema_21": 2000,
            "volume_ratio": 1.2,
            "atr": 15,
            "price_change": 0.5,
            "volatility": 0.8
        }

        prediction = await engine.predict(test_features)
        print(f"ML Prediction: {prediction.action} @ {prediction.confidence:.1%}")
        return True

    except Exception as e:
        print(f"ML Engine test failed: {e}")
        return False

async def test_mt5_connection():
    """Test MT5 Connection"""
    print("Testing MT5 Connection...")

    try:
        from mt5_connector import MT5Connector

        connector = MT5Connector(demo_mode=True)

        if connector.initialize():
            print("MT5 Connected successfully")

            # Test data retrieval
            data = connector.get_real_time_data("XAUUSD")
            if data:
                print(f"MT5 Data received: Price ${data.get('current_price', 'N/A')}")
                return True
            else:
                print("MT5 Connected but no data received")
                return False
        else:
            print("MT5 Connection failed")
            return False

    except Exception as e:
        print(f"MT5 test failed: {e}")
        return False

async def test_decision_optimizer():
    """Test Decision Optimizer"""
    print("Testing Decision Optimizer...")

    try:
        from decision_optimizer import DecisionOptimizer

        optimizer = DecisionOptimizer()

        # Test with sample data
        test_data = {
            "rsi": 35,
            "current_price": 2005,
            "ema_20": 2000,
            "ema_50": 1995,
            "atr": 12,
            "volume": 1000,
            "volume_sma": 800
        }

        decision = optimizer.get_adaptive_decision(test_data)
        print(f"Decision: {decision.get('decision', 'HOLD')} @ {decision.get('confidence', 0):.1%}")
        return True

    except Exception as e:
        print(f"Decision Optimizer test failed: {e}")
        return False

async def run_integration_test():
    """Run complete integration test"""
    print("Spectra Killer AI - Integration Test")
    print("=" * 50)

    start_time = datetime.now()

    # Test all components
    tests = [
        ("MT5 Connection", test_mt5_connection),
        ("Decision Optimizer", test_decision_optimizer),
        ("ML Engine", test_ml_engine),
        ("Background Agents", test_background_agents)
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"{test_name} crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("INTEGRATION TEST RESULTS")
    print("=" * 50)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        print(f"{test_name:.<30} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")
    print(f"Duration: {(datetime.now() - start_time).total_seconds():.1f} seconds")

    if passed == total:
        print("\nALL SYSTEMS GO - Ready for AI Trading!")
        print("\nNext steps:")
        print("1. Configure your API keys in ai_trading_config.json")
        print("2. Run: python integrated_ai_trading_system.py")
        print("3. Monitor the AI trading session")
    else:
        print("\nSome systems need attention before trading")
        print("Check the failed components above")

    return passed == total

if __name__ == "__main__":
    asyncio.run(run_integration_test())