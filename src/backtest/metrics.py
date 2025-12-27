"""
Performance Metrics Module

Calculates trading performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_trade: float
    num_trades: int
    exposure: float  # Time in market


def compute_metrics(
    nav: pd.Series,
    positions: pd.Series,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252
) -> PerformanceMetrics:
    """
    Compute comprehensive performance metrics.

    Args:
        nav: Net Asset Value series
        positions: Position series (-1 to 1)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        PerformanceMetrics object
    """
    # Returns
    returns = nav.pct_change().dropna()

    if len(returns) < 10:
        return PerformanceMetrics(
            total_return=0, annual_return=0, volatility=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
            calmar_ratio=0, win_rate=0, profit_factor=0,
            avg_trade=0, num_trades=0, exposure=0
        )

    # Total return
    total_return = nav.iloc[-1] / nav.iloc[0] - 1

    # Annualized return
    n_years = len(returns) / periods_per_year
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Volatility
    volatility = returns.std() * np.sqrt(periods_per_year)

    # Sharpe ratio
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else volatility
    sortino_ratio = excess_return / downside_std if downside_std > 0 else 0

    # Maximum drawdown
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_drawdown = drawdown.min()

    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Trade statistics
    position_changes = positions.diff().abs()
    trades = position_changes[position_changes > 0.1]
    num_trades = len(trades)

    # Win rate (daily)
    winning_days = (returns > 0).sum()
    win_rate = winning_days / len(returns) if len(returns) > 0 else 0

    # Profit factor
    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')

    # Average trade return
    avg_trade = returns.mean() * periods_per_year

    # Exposure (time in market)
    exposure = (positions.abs() > 0.1).mean()

    return PerformanceMetrics(
        total_return=total_return,
        annual_return=annual_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade=avg_trade,
        num_trades=num_trades,
        exposure=exposure
    )


def compute_yearly_metrics(
    nav: pd.Series,
    positions: pd.Series,
    risk_free_rate: float = 0.05
) -> Dict[int, PerformanceMetrics]:
    """
    Compute metrics for each calendar year.
    """
    yearly_metrics = {}

    for year in nav.index.year.unique():
        mask = nav.index.year == year
        yearly_nav = nav[mask]
        yearly_pos = positions[mask]

        if len(yearly_nav) > 20:
            metrics = compute_metrics(yearly_nav, yearly_pos, risk_free_rate)
            yearly_metrics[year] = metrics

    return yearly_metrics


def compute_benchmark_comparison(
    strategy_nav: pd.Series,
    benchmark_prices: pd.Series
) -> Dict[str, float]:
    """
    Compare strategy against benchmark.
    """
    # Align series
    common_idx = strategy_nav.index.intersection(benchmark_prices.index)
    strat = strategy_nav.loc[common_idx]
    bench = benchmark_prices.loc[common_idx]

    # Normalize benchmark
    bench_nav = bench / bench.iloc[0]

    strat_ret = strat.iloc[-1] / strat.iloc[0] - 1
    bench_ret = bench_nav.iloc[-1] - 1

    # Alpha and Beta
    strat_returns = strat.pct_change().dropna()
    bench_returns = bench_nav.pct_change().dropna()

    if len(strat_returns) > 10 and len(bench_returns) > 10:
        # Align returns
        common_ret_idx = strat_returns.index.intersection(bench_returns.index)
        sr = strat_returns.loc[common_ret_idx]
        br = bench_returns.loc[common_ret_idx]

        covariance = np.cov(sr, br)[0, 1]
        bench_var = br.var()
        beta = covariance / bench_var if bench_var > 0 else 1.0

        alpha = (sr.mean() - beta * br.mean()) * 252

        # Information ratio
        tracking_error = (sr - br).std() * np.sqrt(252)
        info_ratio = (sr.mean() - br.mean()) * 252 / tracking_error if tracking_error > 0 else 0
    else:
        beta = 1.0
        alpha = 0.0
        info_ratio = 0.0

    return {
        'strategy_return': strat_ret,
        'benchmark_return': bench_ret,
        'excess_return': strat_ret - bench_ret,
        'alpha': alpha,
        'beta': beta,
        'information_ratio': info_ratio
    }
