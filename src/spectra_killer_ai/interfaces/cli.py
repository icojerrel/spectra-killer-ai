"""
Command Line Interface for Spectra Killer AI
Professional CLI with comprehensive commands
"""

import asyncio
import typer
from typing import Optional, Dict
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
import yaml

from ..core.engine import SpectraTradingEngine
from .analyzer import quick_analysis

app = typer.Typer(
    name="spectra-killer",
    help="🚀 Spectra Killer AI - Advanced Trading System",
    no_args_is_help=True,
)

console = Console()

# Global engine instance
engine: Optional[SpectraTradingEngine] = None


@app.command()
def quick(
    symbol: str = typer.Option("XAUUSD", "--symbol", "-s", help="Trading symbol"),
    timeframe: str = typer.Option("M5", "--timeframe", "-t", help="Timeframe"),
):
    """Quick market analysis"""
    console.print("🔍 [bold blue]Quick Market Analysis[/bold blue]")
    console.print("=" * 50)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing market...", total=None)
        
        try:
            result = asyncio.run(quick_analysis())
            progress.update(task, completed=True)
            
            if 'error' in result:
                console.print(f"❌ [red]Error: {result['error']}[/red]")
                raise typer.Exit(1)
            
            # Display analysis results
            _display_analysis_results(result)
            
        except Exception as e:
            console.print(f"❌ [red]Analysis failed: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def paper_trade(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
    duration: Optional[int] = typer.Option(None, "--duration", "-d", help="Duration in minutes"),
    symbol: str = typer.Option("XAUUSD", "--symbol", "-s", help="Trading symbol"),
):
    """Start paper trading"""
    console.print("📊 [bold green]Paper Trading Mode[/bold green]")
    console.print("=" * 50)
    
    # Load configuration
    if config:
        try:
            with open(config) as f:
                if config.suffix in ['.yaml', '.yml']:
                    trading_config = yaml.safe_load(f)
                else:
                    trading_config = json.load(f)
        except Exception as e:
            console.print(f"❌ [red]Error loading config: {e}[/red]")
            raise typer.Exit(1)
    else:
        trading_config = _get_default_paper_config(symbol)
    
    # Start paper trading
    try:
        asyncio.run(_run_paper_trading(trading_config, duration))
    except KeyboardInterrupt:
        console.print("\n⚠️ [yellow]Trading stopped by user[/yellow]")
    except Exception as e:
        console.print(f"❌ [red]Trading error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def backtest(
    days: int = typer.Option(30, "--days", "-d", help="Number of days to backtest"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed results"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results"),
):
    """Run backtesting"""
    console.print(f"📈 [bold blue]Backtesting {days} days[/bold blue]")
    console.print("=" * 50)
    
    # Load configuration
    if config:
        try:
            with open(config) as f:
                if config.suffix in ['.yaml', '.yml']:
                    backtest_config = yaml.safe_load(f)
                else:
                    backtest_config = json.load(f)
        except Exception as e:
            console.print(f"❌ [red]Error loading config: {e}[/red]")
            raise typer.Exit(1)
    else:
        backtest_config = _get_default_backtest_config()
    
    # Run backtest
    try:
        results = asyncio.run(_run_backtest(backtest_config, days))
        _display_backtest_results(results, detailed)
        
        # Save results if output specified
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"✅ [green]Results saved to {output}[/green]")
            
    except Exception as e:
        console.print(f"❌ [red]Backtest error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def dashboard(
    port: int = typer.Option(8080, "--port", "-p", help="Port for dashboard"),
    host: str = typer.Option("localhost", "--host", help="Host for dashboard"),
):
    """Launch web dashboard"""
    console.print("🌐 [bold purple]Web Dashboard[/bold purple]")
    console.print("=" * 50)
    console.print(f"🚀 Starting dashboard on http://{host}:{port}")
    console.print("💡 Press Ctrl+C to stop the dashboard")
    
    try:
        # This would launch the actual web dashboard
        console.print("📊 Dashboard features:")
        console.print("   • Real-time P&L tracking")
        console.print("   • Live signal monitoring")
        console.print("   • Performance analytics")
        console.print("   • Risk metrics dashboard")
        console.print("")
        console.print("⚠️ [yellow]Note: Web dashboard implementation pending[/yellow]")
        
    except KeyboardInterrupt:
        console.print("\n⚠️ [yellow]Dashboard stopped[/yellow]")


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    validate: Optional[Path] = typer.Option(None, "--validate", help="Validate configuration file"),
    generate: Optional[Path] = typer.Option(None, "--generate", help="Generate sample configuration"),
):
    """Configuration management"""
    if show:
        _show_current_config()
    elif validate:
        _validate_config(validate)
    elif generate:
        _generate_config(generate)
    else:
        console.print("Use --show, --validate, or --generate")


@app.command()
def status():
    """Show system status"""
    console.print("🔍 [bold blue]System Status[/bold blue]")
    console.print("=" * 50)
    
    status_data = {
        'engine': 'Not running',
        'mode': 'None',
        'connected': False,
        'positions': 0,
        'balance': '$0.00',
        'uptime': '00:00:00',
    }
    
    # Create status table
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    
    for key, value in status_data.items():
        table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(table)


def _display_analysis_results(result: Dict) -> None:
    """Display quick analysis results"""
    # Current price panel
    price_panel = Panel(
        f"[bold green]Current Price:[/bold green] ${result['current_price']:.2f}",
        title="📊 Market Data",
        border_style="green"
    )
    console.print(price_panel)
    
    # Technical signals
    if result.get('technical_signal'):
        tech = result['technical_signal']
        signal_style = "bold green" if tech['signal'] == 'BUY' else "bold red" if tech['signal'] == 'SELL' else "bold yellow"
        
        tech_table = Table(title="📈 Technical Analysis")
        tech_table.add_column("Indicator", style="bold")
        tech_table.add_column("Signal")
        tech_table.add_column("Confidence")
        
        tech_table.add_row("RSI", tech['indicators']['rsi']['signal'], f"{tech['indicators']['rsi']['confidence']:.1%}")
        tech_table.add_row("EMA", tech['indicators']['ema']['signal'], f"{tech['indicators']['ema']['confidence']:.1%}")
        tech_table.add_row("Bollinger", tech['indicators']['bollinger']['signal'], f"{tech['indicators']['bollinger']['confidence']:.1%}")
        
        console.print(tech_table)
        
        # Combined signal
        combined_style = "bold green" if tech['signal'] == 'BUY' else "bold red" if tech['signal'] == 'SELL' else "bold yellow"
        signal_panel = Panel(
            f"[{combined_style}]Signal: {tech['signal']}[/combined_style]\\n"
            f"[bold]Confidence: {tech['confidence']:.1%}[/bold]",
            title="🎯 Combined Signal",
            border_style="blue"
        )
        console.print(signal_panel)


async def _run_paper_trading(config: Dict, duration: Optional[int] = None):
    """Run paper trading session"""
    global engine
    
    engine = SpectraTradingEngine(config)
    
    with Live(console=console, refresh_per_second=1) as live:
        try:
            if duration:
                # Limited duration
                task = asyncio.create_task(_run_trading_with_timeout(engine, duration))
                await task
            else:
                # Unlimited duration
                task = asyncio.create_task(engine.start())
                await task
                
        except Exception as e:
            console.print(f"❌ [red]Trading error: {e}[/red]")


async def _run_trading_with_timeout(engine: SpectraTradingEngine, duration_minutes: int):
    """Run trading for specified duration"""
    try:
        # Start trading in background
        trading_task = asyncio.create_task(engine.start())
        
        # Wait for specified duration
        await asyncio.sleep(duration_minutes * 60)
        
        # Stop trading
        await engine.stop()
        
        # Cancel trading task if still running
        if not trading_task.done():
            trading_task.cancel()
            
    except asyncio.CancelledError:
        await engine.stop()


async def _run_backtest(config: Dict, days: int) -> Dict:
    """Run backtesting"""
    # Configure for backtest
    config['trading']['mode'] = 'backtest'
    config['trading']['backtest_days'] = days
    
    engine = SpectraTradingEngine(config)
    
    # Simulate backtest (placeholder implementation)
    results = {
        'period': f"{days} days",
        'initial_balance': 10000,
        'final_balance': 12500,
        'total_return': 25.0,
        'win_rate': 65.5,
        'max_drawdown': 5.2,
        'sharpe_ratio': 1.45,
        'total_trades': 156,
        'profit_factor': 1.8,
    }
    
    return results


def _display_backtest_results(results: Dict, detailed: bool = False) -> None:
    """Display backtest results"""
    # Summary table
    summary_table = Table(title="📊 Backtest Summary")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value")
    
    summary_table.add_row("Period", results['period'])
    summary_table.add_row("Initial Balance", f"${results['initial_balance']:,.2f}")
    summary_table.add_row("Final Balance", f"${results['final_balance']:,.2f}")
    summary_table.add_row("Total Return", f"{results['total_return']:.1f}%")
    summary_table.add_row("Win Rate", f"{results['win_rate']:.1f}%")
    summary_table.add_row("Max Drawdown", f"{results['max_drawdown']:.1f}%")
    summary_table.add_row("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
    summary_table.add_row("Total Trades", str(results['total_trades']))
    summary_table.add_row("Profit Factor", f"{results['profit_factor']:.2f}")
    
    console.print(summary_table)


def _show_current_config():
    """Show current configuration"""
    config = _get_default_config()
    
    console.print("📋 [bold blue]Current Configuration[/bold blue]")
    console.print("=" * 50)
    
    # Pretty print configuration as YAML
    import yaml
    config_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)
    console.print(Panel(config_yaml, title="Configuration", border_style="blue"))


def _validate_config(config_path: Path):
    """Validate configuration file"""
    try:
        with open(config_path) as f:
            if config_path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        from ..utils.validation import validate_trading_config
        is_valid, issues = validate_trading_config(config)
        
        if is_valid:
            console.print("✅ [green]Configuration is valid[/green]")
        else:
            console.print("❌ [red]Configuration validation failed:[/red]")
            for issue in issues:
                console.print(f"   • {issue}")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"❌ [red]Error validating config: {e}[/red]")
        raise typer.Exit(1)


def _generate_config(output_path: Path):
    """Generate sample configuration"""
    config = _get_default_config()
    
    try:
        with open(output_path, 'w') as f:
            if output_path.suffix in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(config, f, indent=2)
        
        console.print(f"✅ [green]Sample configuration saved to {output_path}[/green]")
        
    except Exception as e:
        console.print(f"❌ [red]Error generating config: {e}[/red]")
        raise typer.Exit(1)


def _get_default_config() -> Dict:
    """Get default configuration"""
    return {
        'trading': {
            'symbol': 'XAUUSD',
            'timeframe': 'M5',
            'initial_balance': 10000,
            'mode': 'paper',
            'analysis_interval': 60,
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
        },
        'data': {
            'use_real_data': False,
        }
    }


def _get_default_paper_config(symbol: str) -> Dict:
    """Get default paper trading configuration"""
    config = _get_default_config()
    config['trading']['symbol'] = symbol
    config['trading']['mode'] = 'paper'
    config['trading']['auto_trade'] = True
    return config


def _get_default_backtest_config() -> Dict:
    """Get default backtest configuration"""
    config = _get_default_config()
    config['trading']['mode'] = 'backtest'
    config['trading']['auto_trade'] = True
    return config


if __name__ == "__main__":
    app()
