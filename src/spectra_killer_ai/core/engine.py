"""
Spectra Trading Engine - Core trading system
Main orchestrator for all trading activities with comprehensive event handling
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Union, Callable
import json
from pathlib import Path

from .portfolio import Portfolio
from .risk_manager import RiskManager
from .position import Position, PositionType
from .events import EventBus, Event, EventType, event_bus
from ..data.sources.simulator import XAUUSDSimulator
from ..strategies.technical.technical_analyzer import TechnicalAnalyzer
from ..strategies.ml.cnn_trader import CNNTradingStrategy
from ..integrations.moon_dev import MoonDevIntegration
from ..utils.helpers import generate_id, format_currency
from ..utils.metrics import calculate_return, calculate_sharpe_ratio, calculate_max_drawdown

logger = logging.getLogger(__name__)


class SpectraTradingEngine:
    """
    Advanced trading engine with AI integration and comprehensive risk management
    
    Features:
    - Multi-strategy execution (Technical + AI)
    - Real-time risk monitoring
    - Event-driven architecture
    - Comprehensive logging and monitoring
    - Paper and live trading support
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize trading engine
        
        Args:
            config: Trading configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Core components
        self.portfolio = Portfolio(
            initial_balance=self.config['trading']['initial_balance']
        )
        self.risk_manager = RiskManager(self.config['risk_management'])
        
        # Data sources
        self.data_source = XAUUSDSimulator(self.config.get('data', {}))
        
        # Trading strategies
        self.technical_analyzer = TechnicalAnalyzer(self.config.get('technical', {}))
        self.cnn_strategy = None
        if self.config['ai'].get('enabled', False):
            self.cnn_strategy = CNNTradingStrategy(self.config['ai'])
        
        # Moon Dev integrations
        self.moon_dev_integration = None
        if self.config.get('moon_dev', {}).get('enabled', False):
            self.moon_dev_integration = MoonDevIntegration(self.config['moon_dev'])
        
        # Engine state
        self.is_running = False
        self.is_paused = False
        self.trading_mode = self.config['trading']['mode']  # 'paper', 'live', 'backtest'
        
        # Timing and tracking
        self.start_time = None
        self.last_analysis_time = None
        self.analysis_interval = self.config['trading'].get('analysis_interval', 60)  # seconds
        
        # Statistics
        self.stats = {
            'total_analyses': 0,
            'total_signals': 0,
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'errors': 0,
            'uptime_seconds': 0,
        }
        
        # Initialize event handlers
        self._setup_event_handlers()
        
        # Subscribe to market data updates
        self._market_data_callbacks: List[Callable] = []
        
        logger.info("SpectraTradingEngine initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'trading': {
                'symbol': 'XAUUSD',
                'timeframe': 'M5',
                'initial_balance': 10000,
                'mode': 'paper',  # 'paper', 'live', 'backtest'
                'analysis_interval': 60,
                'max_daily_trades': 10,
                'auto_trade': True,
            },
            'risk_management': {
                'max_position_size': 5000,
                'max_risk_per_trade': 0.02,
                'max_daily_loss': 0.05,
                'max_positions': 3,
                'max_portfolio_risk': 0.10,
            },
            'technical': {
                'rsi_period': 14,
                'rsi_overbought': 65,
                'rsi_oversold': 35,
                'ema_short': 9,
                'ema_long': 21,
                'bb_period': 20,
                'bb_std': 2,
            },
            'ai': {
                'enabled': False,
                'model_type': 'cnn_hybrid',
                'confidence_threshold': 0.65,
                'model_path': 'models/cnn_trader.pth',
            },
            'data': {
                'use_real_data': False,
                'data_path': 'data/xauusd.csv',
            }
        }
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for the engine"""
        event_bus.subscribe(EventType.SIGNAL_GENERATED, self._on_signal_generated)
        event_bus.subscribe(EventType.ERROR_OCCURRED, self._on_error_occurred)
        event_bus.subscribe(EventType.RISK_LIMIT_EXCEEDED, self._on_risk_limit_exceeded)
    
    async def start(self) -> None:
        """Start the trading engine"""
        if self.is_running:
            logger.warning("Trading engine is already running")
            return
        
        logger.info("Starting SpectraTradingEngine...")
        self.is_running = True
        self.is_paused = False
        self.start_time = datetime.now()
        
        # Reset daily stats if needed
        self.risk_manager.reset_daily_stats()
        self.portfolio.reset_daily_stats()
        
        try:
            # Load AI models if enabled
            if self.cnn_strategy:
                await self.cnn_strategy.load_model()
            
            # Main trading loop
            await self._main_trading_loop()
            
        except KeyboardInterrupt:
            logger.info("Trading engine stopped by user")
        except Exception as e:
            logger.error(f"Critical error in trading engine: {e}")
            self.stats['errors'] += 1
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the trading engine"""
        logger.info("Stopping SpectraTradingEngine...")
        self.is_running = False
        self.is_paused = False
        
        # Close all open positions in live mode
        if self.trading_mode == 'live' and self.portfolio.positions:
            await self._emergency_close_all_positions()
        
        # Final statistics
        if self.start_time:
            self.stats['uptime_seconds'] = (datetime.now() - self.start_time).total_seconds()
        
        # Save performance data
        await self._save_performance_data()
        
        logger.info("SpectraTradingEngine stopped")
    
    async def pause(self) -> None:
        """Pause trading engine"""
        logger.info("Pausing trading engine...")
        self.is_paused = True
    
    async def resume(self) -> None:
        """Resume trading engine"""
        logger.info("Resuming trading engine...")
        self.is_paused = False
    
    async def _main_trading_loop(self) -> None:
        """Main trading loop"""
        while self.is_running:
            try:
                if not self.is_paused:
                    await self._trading_cycle()
                else:
                    logger.debug("Trading engine is paused")
                
                # Wait for next cycle
                await asyncio.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _trading_cycle(self) -> None:
        """Execute one complete trading cycle"""
        try:
            # 1. Get market data
            market_data = await self._get_market_data()
            if not market_data:
                return
            
            # 2. Update portfolio with current prices
            self.portfolio.update_positions(market_data)
            
            # 3. Generate trading signals
            signals = await self._generate_signals(market_data)
            self.stats['total_signals'] += len(signals)
            
            # 4. Process signals and execute trades
            if signals and self.config['trading'].get('auto_trade', True):
                await self._process_signals(signals, market_data)
            
            # 5. Update risk metrics
            await self._update_risk_metrics(market_data)
            
            # 6. Take portfolio snapshot
            self.portfolio.take_snapshot(market_data)
            
            # 7. Log cycle completion
            await self._log_cycle_completion(signals)
            
            self.stats['total_analyses'] += 1
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            event_bus.publish(Event(
                type=EventType.ERROR_OCCURRED,
                data={'error': str(e), 'source': 'trading_cycle'},
                source="SpectraTradingEngine"
            ))
    
    async def _get_market_data(self) -> Optional[Dict[str, Decimal]]:
        """Get current market data"""
        try:
            symbol = self.config['trading']['symbol']
            
            if self.config['data'].get('use_real_data', False):
                # Would integrate with MT5 or other real data source
                logger.warning("Real data integration not implemented, using simulator")
            
            # Use simulator for now
            data = self.data_source.generate_data(timeframe='5M', periods=100)
            current_price = Decimal(str(data['close'].iloc[-1]))
            
            return {symbol: current_price}
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    async def _generate_signals(self, market_data: Dict[str, Decimal]) -> List[Dict]:
        """Generate trading signals from strategies"""
        signals = []
        symbol = self.config['trading']['symbol']
        
        try:
            # Get historical data for analysis
            historical_data = self.data_source.generate_data(timeframe='5M', periods=200)
            
            # Technical analysis signals
            technical_signal = await self.technical_analyzer.analyze(historical_data)
            if technical_signal:
                signals.append({
                    'source': 'technical',
                    'symbol': symbol,
                    'signal': technical_signal['signal'],
                    'confidence': technical_signal['confidence'],
                    'entry_price': market_data[symbol],
                    'metadata': technical_signal
                })
            
            # AI/CNN signals
            if self.cnn_strategy:
                ai_signal = await self.cnn_strategy.predict(historical_data)
                if ai_signal and ai_signal['confidence'] >= self.config['ai']['confidence_threshold']:
                    signals.append({
                        'source': 'ai',
                        'symbol': symbol,
                        'signal': ai_signal['signal'],
                        'confidence': ai_signal['confidence'],
                        'entry_price': market_data[symbol],
                        'metadata': ai_signal
                    })
            
            # Moon Dev swarm signals
            if self.moon_dev_integration:
                strategy_context = {
                    'timeframe': self.config['trading']['timeframe'],
                    'risk_tolerance': 'medium',
                    'symbol': symbol
                }
                market_context = {
                    'price': float(market_data[symbol]),
                    'rsi': technical_analysis.get('traditional_analysis', {}).get('rsi', {}).get('value', 50) if technical_analysis else 50,
                    'ema_short': technical_analysis.get('traditional_analysis', {}).get('ema', {}).get('short') if technical_analysis else None,
                    'ema_long': technical_analysis.get('traditional_analysis', {}).get('ema', {}).get('long') if technical_analysis else None
                }
                
                swarm_signal = await self.moon_dev_integration.get_enhanced_signal(market_context, strategy_context)
                if swarm_signal['confidence'] >= self.config.get('moon_dev', {}).get('confidence_threshold', 0.6):
                    signals.append({
                        'source': 'moon_dev_swarm',
                        'symbol': symbol,
                        'signal': swarm_signal['signal'],
                        'confidence': swarm_signal['confidence'],
                        'entry_price': market_data[symbol],
                        'metadata': swarm_signal
                    })
            
            # Combine signals if multiple
            if len(signals) > 1:
                combined_signal = await self._combine_signals(signals)
                return [combined_signal]
            elif signals:
                return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return []
    
    async def _combine_signals(self, signals: List[Dict]) -> Dict:
        """Combine multiple signals into one"""
        # Simple voting for now, could be made more sophisticated
        buy_votes = sum(1 for s in signals if s['signal'] == 'BUY')
        sell_votes = sum(1 for s in signals if s['signal'] == 'SELL')
        
        if buy_votes > sell_votes:
            final_signal = 'BUY'
        elif sell_votes > buy_votes:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        
        # Average confidence
        avg_confidence = sum(s['confidence'] for s in signals) / len(signals)
        
        # Use first signal's price and metadata
        base_signal = signals[0].copy()
        base_signal['signal'] = final_signal
        base_signal['confidence'] = avg_confidence
        base_signal['source'] = 'combined'
        
        return base_signal
    
    async def _process_signals(self, signals: List[Dict], market_data: Dict[str, Decimal]) -> None:
        """Process trading signals and execute trades"""
        for signal in signals:
            if signal['signal'] == 'HOLD':
                continue
            
            try:
                await self._execute_signal(signal, market_data)
            except Exception as e:
                logger.error(f"Error executing signal: {e}")
                self.stats['failed_trades'] += 1
    
    async def _execute_signal(self, signal: Dict, market_data: Dict[str, Decimal]) -> None:
        """Execute a trading signal"""
        symbol = signal['symbol']
        entry_price = signal['entry_price']
        confidence = signal['confidence']
        signal_type = signal['signal']
        
        # Check if we already have positions for this symbol
        existing_positions = self.portfolio.get_positions_by_symbol(symbol)
        
        # Risk check
        can_trade, reasons = self.risk_manager.can_open_position(
            portfolio_value=self.portfolio.get_equity(market_data),
            current_positions=len(self.portfolio.positions),
            daily_pnl=self.portfolio.daily_pnl,
            proposed_size=Decimal('1000'),  # Will be calculated properly
            proposed_risk=Decimal('100')
        )
        
        if not can_trade:
            logger.info(f"Cannot open position: {reasons}")
            return
        
        # Calculate position parameters
        position_size = self.risk_manager.calculate_position_size(
            account_balance=self.portfolio.balance,
            signal_confidence=confidence,
            current_price=entry_price,
            stop_loss=entry_price - Decimal('20')  # Placeholder
        )
        
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, signal_type)
        take_profit = self.risk_manager.calculate_take_profit(entry_price, signal_type, stop_loss)
        
        # Create position
        position = Position(
            position_id=generate_id('pos'),
            symbol=symbol,
            position_type=PositionType.LONG if signal_type == 'BUY' else PositionType.SHORT,
            entry_price=entry_price,
            quantity=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_id=f"{signal['source']}_strategy",
            confidence=Decimal(str(confidence))
        )
        
        # Open position
        if self.portfolio.open_position(position):
            self.stats['successful_trades'] += 1
            
            # Publish event
            event_bus.publish(Event(
                type=EventType.POSITION_OPENED,
                data={
                    'position_id': position.position_id,
                    'symbol': symbol,
                    'type': position.position_type.value,
                    'size': float(position_size),
                    'signal': signal
                },
                source="SpectraTradingEngine"
            ))
            
            logger.info(f"Opened {signal_type} position: {position}")
        else:
            self.stats['failed_trades'] += 1
    
    async def _update_risk_metrics(self, market_data: Dict[str, Decimal]) -> None:
        """Update risk metrics"""
        # Calculate current portfolio metrics
        equity = self.portfolio.get_equity(market_data)
        exposure = self.portfolio.get_exposure()
        
        # Update risk manager metrics
        self.risk_manager.current_metrics.portfolio_value = equity
        self.risk_manager.current_metrics.total_exposure = sum(exposure.values())
        self.risk_manager.current_metrics.daily_pnl = self.portfolio.daily_pnl
        self.risk_manager.current_metrics.open_positions_count = len(self.portfolio.positions)
        
        # Calculate risk score
        risk_score = self.risk_manager.calculate_risk_score(
            equity, [p.to_dict() for p in self.portfolio.get_open_positions()]
        )
        self.risk_manager.risk_score = risk_score
        self.risk_manager.update_risk_level(risk_score)
    
    async def _log_cycle_completion(self, signals: List[Dict]) -> None:
        """Log completion of trading cycle"""
        portfolio_stats = self.portfolio.get_performance_metrics()
        
        logger.info(f"Trading cycle completed | "
                   f"Balance: {format_currency(portfolio_stats['balance'])} | "
                   f"Positions: {portfolio_stats['open_positions']} | "
                   f"Daily P&L: {format_currency(portfolio_stats['daily_pnl'])} | "
                   f"Signals: {len(signals)} | "
                   f"Risk Score: {self.risk_manager.risk_score:.1f}")
    
    async def _emergency_close_all_positions(self) -> None:
        """Emergency close all positions"""
        logger.warning("Emergency: Closing all positions")
        
        # Get current market prices (simplified)
        market_data = await self._get_market_data()
        if not market_data:
            logger.error("Cannot get market data for emergency close")
            return
        
        for position in self.portfolio.get_open_positions():
            if position.symbol in market_data:
                self.portfolio.close_position(position.position_id, market_data[position.symbol])
    
    async def _save_performance_data(self) -> None:
        """Save performance data to file"""
        try:
            performance_data = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'stats': self.stats,
                'portfolio': self.portfolio.export_data(),
                'risk_report': self.risk_manager.get_risk_report(),
            }
            
            # Save to performance file
            perf_file = Path(f"performance/engine_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            perf_file.parent.mkdir(exist_ok=True)
            
            with open(perf_file, 'w') as f:
                json.dump(performance_data, f, indent=2, default=str)
            
            logger.info(f"Performance data saved to {perf_file}")
            
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    async def quick_analysis(self) -> Dict:
        """Quick market analysis"""
        try:
            # Generate sample data for analysis
            data = self.data_source.generate_data(timeframe='5M', periods=100)
            
            # Technical analysis
            technical_signal = await self.technical_analyzer.analyze(data)
            
            # AI analysis if available
            ai_signal = None
            if self.cnn_strategy:
                ai_signal = await self.cnn_strategy.predict(data)
            
            return {
                'symbol': self.config['trading']['symbol'],
                'current_price': float(data['close'].iloc[-1]),
                'technical_signal': technical_signal,
                'ai_signal': ai_signal,
                'timestamp': datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Error in quick analysis: {e}")
            return {'error': str(e)}
    
    async def start_paper_trading(self) -> None:
        """Start paper trading mode"""
        self.trading_mode = 'paper'
        logger.info("Starting paper trading mode")
        await self.start()
    
    def get_engine_status(self) -> Dict:
        """Get comprehensive engine status"""
        portfolio_stats = self.portfolio.get_performance_metrics()
        risk_report = self.risk_manager.get_risk_report()
        
        status = {
            'running': self.is_running,
            'paused': self.is_paused,
            'mode': self.trading_mode,
            'uptime_seconds': self.stats['uptime_seconds'],
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'portfolio': portfolio_stats,
            'risk': risk_report,
            'stats': self.stats,
            'config': {
                'symbol': self.config['trading']['symbol'],
                'timeframe': self.config['trading']['timeframe'],
                'analysis_interval': self.analysis_interval,
                'ai_enabled': self.config['ai']['enabled'],
            }
        }
        
        return status
    
    # Event handlers
    async def _on_signal_generated(self, event: Event) -> None:
        """Handle signal generated event"""
        logger.debug(f"Signal generated: {event.data}")
    
    async def _on_error_occurred(self, event: Event) -> None:
        """Handle error occurred event"""
        error_data = event.data
        logger.error(f"Engine error: {error_data.get('error', 'Unknown error')}")
        
        # Could implement error recovery logic here
        self.stats['errors'] += 1
    
    async def _on_risk_limit_exceeded(self, event: Event) -> None:
        """Handle risk limit exceeded event"""
        limit_name = event.data.get('limit_name')
        logger.warning(f"Risk limit exceeded: {limit_name}")
        
        # Could implement automatic risk mitigation
        if limit_name == 'max_daily_loss' and self.is_running:
            logger.critical("Daily loss limit exceeded - stopping trading")
            await self.stop()
    
    def __str__(self) -> str:
        """String representation"""
        status = "🟢 Running" if self.is_running else "🔴 Stopped"
        if self.is_paused:
            status = "🟡 Paused"
        
        return (f"SpectraEngine({status} | "
                f"Balance: {format_currency(self.portfolio.balance)} | "
                f"Positions: {len(self.portfolio.positions)} | "
                f"Risk Score: {self.risk_manager.risk_score:.1f})")
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return (f"SpectraTradingEngine(running={self.is_running}, "
                f"mode={self.trading_mode}, balance={self.portfolio.balance}, "
                f"positions={len(self.portfolio.positions)})")
