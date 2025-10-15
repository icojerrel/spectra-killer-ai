"""
Financial metrics calculations
"""

import math
from decimal import Decimal
from typing import List, Union
import numpy as np


def calculate_return(initial_value: Union[float, Decimal], 
                    final_value: Union[float, Decimal]) -> float:
    """
    Calculate percentage return
    
    Args:
        initial_value: Initial value
        final_value: Final value
        
    Returns:
        Percentage return (as decimal, e.g., 0.05 for 5%)
    """
    if initial_value == 0:
        return 0.0
    
    return (final_value - initial_value) / initial_value


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: List of periodic returns
        risk_free_rate: Risk-free rate (as decimal)
        
    Returns:
        Sharpe ratio
    """
    if not returns:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    if len(excess_returns) == 0 or excess_returns.std() == 0:
        return 0.0
    
    return excess_returns.mean() / excess_returns.std()


def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio (downside deviation)
    
    Args:
        returns: List of periodic returns
        risk_free_rate: Risk-free rate (as decimal)
        
    Returns:
        Sortino ratio
    """
    if not returns:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    # Calculate downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    downside_deviation = np.std(downside_returns)
    
    if downside_deviation == 0:
        return 0.0
    
    return excess_returns.mean() / downside_deviation


def calculate_max_drawdown(equity_curve: List[Union[float, Decimal]]) -> float:
    """
    Calculate maximum drawdown
    
    Args:
        equity_curve: List of portfolio values over time
        
    Returns:
        Maximum drawdown (as decimal, e.g., 0.05 for 5%)
    """
    if not equity_curve:
        return 0.0
    
    equity_array = np.array([float(v) for v in equity_curve])
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max
    
    return abs(drawdown.min())


def calculate_calmar_ratio(returns: List[float], max_drawdown: float) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown)
    
    Args:
        returns: List of periodic returns
        max_drawdown: Maximum drawdown (as decimal)
        
    Returns:
        Calmar ratio
    """
    if not returns or max_drawdown == 0:
        return 0.0
    
    annual_return = np.mean(returns) * 252  # Assuming daily returns
    return annual_return / abs(max_drawdown)


def calculate_profit_factor(winning_trades: List[float], losing_trades: List[float]) -> float:
    """
    Calculate profit factor
    
    Args:
        winning_trades: List of winning trade amounts
        losing_trades: List of losing trade amounts
        
    Returns:
        Profit factor
    """
    gross_profit = sum(winning_trades) if winning_trades else 0.0
    gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
    
    return gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0


def calculate_win_rate(winning_trades: int, total_trades: int) -> float:
    """
    Calculate win rate
    
    Args:
        winning_trades: Number of winning trades
        total_trades: Total number of trades
        
    Returns:
        Win rate (as decimal)
    """
    if total_trades == 0:
        return 0.0
    
    return winning_trades / total_trades


def calculate_average_win_loss(winning_trades: List[float], losing_trades: List[float]) -> tuple[float, float]:
    """
    Calculate average win and loss
    
    Args:
        winning_trades: List of winning trade amounts
        losing_trades: List of losing trade amounts
        
    Returns:
        Tuple of (average_win, average_loss)
    """
    avg_win = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0.0
    
    return avg_win, avg_loss


def calculate_var(returns: List[float], confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR)
    
    Args:
        returns: List of returns
        confidence_level: Confidence level (0.95 for 95% VaR)
        
    Returns:
        VaR value (as decimal)
    """
    if not returns:
        return 0.0
    
    returns_array = np.array(returns)
    var = np.percentile(returns_array, (1 - confidence_level) * 100)
    
    return abs(var)


def calculate_cvar(returns: List[float], confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR)
    
    Args:
        returns: List of returns
        confidence_level: Confidence level (0.95 for 95% CVaR)
        
    Returns:
        CVaR value (as decimal)
    """
    if not returns:
        return 0.0
    
    returns_array = np.array(returns)
    var_threshold = np.percentile(returns_array, (1 - confidence_level) * 100)
    
    # CVaR is the mean of returns below VaR threshold
    cvar_returns = returns_array[returns_array <= var_threshold]
    
    return abs(np.mean(cvar_returns)) if len(cvar_returns) > 0 else 0.0


def calculate_information_ratio(portfolio_returns: List[float], 
                               benchmark_returns: List[float]) -> float:
    """
    Calculate Information Ratio
    
    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Information ratio
    """
    if not portfolio_returns or not benchmark_returns or len(portfolio_returns) != len(benchmark_returns):
        return 0.0
    
    portfolio_array = np.array(portfolio_returns)
    benchmark_array = np.array(benchmark_returns)
    
    excess_returns = portfolio_array - benchmark_array
    
    if excess_returns.std() == 0:
        return 0.0
    
    return excess_returns.mean() / excess_returns.std()


def calculate_beta(portfolio_returns: List[float], market_returns: List[float]) -> float:
    """
    Calculate beta (systematic risk)
    
    Args:
        portfolio_returns: Portfolio returns
        market_returns: Market returns
        
    Returns:
        Beta value
    """
    if not portfolio_returns or not market_returns or len(portfolio_returns) != len(market_returns):
        return 0.0
    
    portfolio_array = np.array(portfolio_returns)
    market_array = np.array(market_returns)
    
    if market_array.var() == 0:
        return 0.0
    
    covariance = np.cov(portfolio_array, market_array)[0][1]
    return covariance / market_array.var()


def calculate_alpha(portfolio_returns: List[float], market_returns: List[float], 
                   risk_free_rate: float = 0.0) -> float:
    """
    Calculate alpha (excess return)
    
    Args:
        portfolio_returns: Portfolio returns
        market_returns: Market returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Alpha value
    """
    if not portfolio_returns or not market_returns:
        return 0.0
    
    portfolio_return = np.mean(portfolio_returns)
    market_return = np.mean(market_returns)
    beta = calculate_beta(portfolio_returns, market_returns)
    
    # CAPM: Expected Return = Risk Free Rate + Beta * (Market Return - Risk Free Rate)
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    
    return portfolio_return - expected_return
