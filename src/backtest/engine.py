"""
Backtesting Engine

Simulates trading strategies on historical data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
import os
import torch

from ..core.kernel import SyntropyQuantKernel, RiskManager, KernelOutput
from ..core.gauge import GaugeFieldKernel, GaugeConfig
from ..data.features import FeatureBuilder
from .metrics import PerformanceMetrics, compute_metrics, compute_yearly_metrics


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 1_000_000
    transaction_cost: float = 0.001  # 10 bps
    slippage: float = 0.0005  # 5 bps
    max_position: float = 1.0
    risk_free_rate: float = 0.05
    warmup_period: int = 60
    kernel_type: str = "baseline"
    kernel_path: Optional[str] = None
    min_trade_threshold: float = 0.01
    min_hold_period: int = 1
    signal_window: int = 1
    confidence_threshold: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    symbol: str
    nav: pd.Series
    positions: pd.Series
    signals: pd.Series
    returns: pd.Series
    benchmark_returns: pd.Series
    metrics: PerformanceMetrics
    yearly_metrics: Dict[int, PerformanceMetrics]
    curvature_series: pd.Series
    regime_series: pd.Series
    trade_log: List[Dict] = field(default_factory=list)


class BacktestEngine:
    """
    Main backtesting engine for Syntropy Quant.

    Simulates the kernel on historical data with realistic
    transaction costs and slippage.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.feature_builder = FeatureBuilder()

    def _build_kernel(self, input_dim: int):
        """Instantiate kernel based on configuration."""
        kernel_type = (self.config.kernel_type or "baseline").lower()
        state = None

        if self.config.kernel_path:
            if not os.path.exists(self.config.kernel_path):
                print(f"Warning: model not found at {self.config.kernel_path}; using untrained kernel.")
            else:
                state = torch.load(self.config.kernel_path, map_location="cpu")

        if kernel_type == "gauge":
            gauge_config = GaugeConfig(input_dim=input_dim)
            if isinstance(state, dict):
                cfg = state.get("config", {})
                for key, value in cfg.items():
                    if key in GaugeConfig.__annotations__ and key != "input_dim":
                        setattr(gauge_config, key, value)
            kernel = GaugeFieldKernel(input_dim=input_dim, config=gauge_config)
        else:
            kernel = SyntropyQuantKernel(input_dim=input_dim)

        if state:
            if isinstance(state, dict) and "state_dict" in state:
                kernel.load_state_dict(state["state_dict"], strict=False)
            else:
                kernel.load_state_dict(state, strict=False)
            if isinstance(state, dict):
                mean = state.get("feature_mean")
                std = state.get("feature_std")
                if mean is not None and std is not None:
                    kernel.feature_mean = np.array(mean, dtype=np.float32)
                    kernel.feature_std = np.array(std, dtype=np.float32)
            kernel.eval()

        return kernel

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = 'UNKNOWN',
        verbose: bool = False
    ) -> BacktestResult:
        """
        Run backtest on a single asset.

        Args:
            df: OHLCV DataFrame
            symbol: Symbol name for reporting
            verbose: Show progress bar

        Returns:
            BacktestResult with all metrics
        """
        # Build features
        features = self.feature_builder.build_features(df)
        dt_series = self.feature_builder.get_dt_series(df)

        # Initialize kernel
        kernel = self._build_kernel(input_dim=features.shape[1])
        if hasattr(kernel, "feature_mean") and hasattr(kernel, "feature_std"):
            std = np.where(kernel.feature_std == 0, 1.0, kernel.feature_std)
            features = (features - kernel.feature_mean) / std
        risk_manager = RiskManager(max_position=self.config.max_position)

        # Initialize tracking
        prices = df['close'].values
        n = len(prices)

        nav = np.ones(n) * self.config.initial_capital
        positions = np.zeros(n)
        signals = np.zeros(n)
        curvatures = np.zeros(n)
        regimes = ['normal'] * n
        returns = np.zeros(n)

        # Current position and cash
        current_position = 0.0
        cash = self.config.initial_capital

        # Trade log
        trades = []

        # Run simulation
        signal_buffer: List[float] = []
        last_trade_idx: Optional[int] = None
        iterator = range(self.config.warmup_period, n)
        if verbose:
            iterator = tqdm(iterator, desc=f'Backtesting {symbol}')

        for t in iterator:
            # Get kernel output
            dt_value = 1.0 if self.config.kernel_type == "gauge" else dt_series[t]
            output = kernel.process_step(features[t], dt_value)

            # Compute realized volatility
            if t > 20:
                recent_returns = np.diff(np.log(prices[t-20:t]))
                realized_vol = np.std(recent_returns) * np.sqrt(252)
            else:
                realized_vol = 0.2

            # Signal smoothing and confidence gating
            effective_signal = output.signal * output.confidence
            if output.confidence < self.config.confidence_threshold:
                effective_signal = 0.0

            signal_buffer.append(effective_signal)
            if len(signal_buffer) > self.config.signal_window:
                signal_buffer.pop(0)
            smoothed_signal = float(np.mean(signal_buffer))

            # Risk-adjusted position
            pos_signal = risk_manager.compute_position_size(
                smoothed_signal,
                output.curvature,
                realized_vol
            )

            target_position = pos_signal.risk_adjusted_signal
            if output.regime == "crisis":
                target_position = 0.0

            # Execute trade if position change is significant
            position_delta = target_position - current_position

            if last_trade_idx is not None and (t - last_trade_idx) < self.config.min_hold_period:
                position_delta = 0.0
                target_position = current_position

            if abs(position_delta) > self.config.min_trade_threshold:  # Min trade threshold
                # Transaction costs
                trade_value = abs(position_delta) * prices[t] * nav[t-1] / prices[t-1]
                tc = trade_value * (self.config.transaction_cost + self.config.slippage)

                # Update position
                current_position = target_position
                cash -= tc
                last_trade_idx = t

                trades.append({
                    'time': df.index[t],
                    'price': prices[t],
                    'position_delta': position_delta,
                    'new_position': current_position,
                    'cost': tc,
                    'signal': output.signal,
                    'curvature': output.curvature
                })

            # Compute P&L
            if t > 0:
                price_return = prices[t] / prices[t-1] - 1
                pnl = current_position * price_return * nav[t-1]
                nav[t] = nav[t-1] + pnl
                returns[t] = nav[t] / nav[t-1] - 1
            else:
                nav[t] = nav[t-1]

            # Update risk manager
            risk_manager.update_nav(nav[t] / self.config.initial_capital)

            # Store tracking
            positions[t] = current_position
            signals[t] = smoothed_signal
            curvatures[t] = output.curvature
            regimes[t] = output.regime

        # Create result series
        nav_series = pd.Series(nav, index=df.index, name='NAV')
        pos_series = pd.Series(positions, index=df.index, name='Position')
        sig_series = pd.Series(signals, index=df.index, name='Signal')
        ret_series = pd.Series(returns, index=df.index, name='Returns')
        curv_series = pd.Series(curvatures, index=df.index, name='Curvature')
        reg_series = pd.Series(regimes, index=df.index, name='Regime')

        # Benchmark returns (buy and hold)
        log_prices = np.log(prices)
        benchmark_ret = pd.Series(
            np.diff(log_prices, prepend=log_prices[0]),
            index=df.index,
            name='Benchmark'
        )

        # Compute metrics
        metrics = compute_metrics(
            nav_series / self.config.initial_capital,
            pos_series,
            self.config.risk_free_rate
        )

        yearly = compute_yearly_metrics(
            nav_series / self.config.initial_capital,
            pos_series,
            self.config.risk_free_rate
        )

        return BacktestResult(
            symbol=symbol,
            nav=nav_series / self.config.initial_capital,
            positions=pos_series,
            signals=sig_series,
            returns=ret_series,
            benchmark_returns=benchmark_ret,
            metrics=metrics,
            yearly_metrics=yearly,
            curvature_series=curv_series,
            regime_series=reg_series,
            trade_log=trades
        )

    def run_multiple(
        self,
        data_dict: Dict[str, pd.DataFrame],
        verbose: bool = True
    ) -> Dict[str, BacktestResult]:
        """
        Run backtest on multiple assets.
        """
        results = {}

        for symbol, df in data_dict.items():
            if verbose:
                print(f"\nBacktesting {symbol}...")

            result = self.run(df, symbol, verbose=False)
            results[symbol] = result

            if verbose:
                m = result.metrics
                print(f"  Annual Return: {m.annual_return*100:.1f}%")
                print(f"  Sharpe Ratio:  {m.sharpe_ratio:.2f}")
                print(f"  Max Drawdown:  {m.max_drawdown*100:.1f}%")

        return results


def generate_report(
    results: Dict[str, BacktestResult],
    category_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Generate a summary report from backtest results.
    """
    if not results:
        return pd.DataFrame(columns=[
            'Symbol', 'Category', 'Strategy Annual %', 'Benchmark Annual %',
            'Excess Return %', 'Sharpe', 'Sortino', 'Max DD %',
            'Calmar', 'Win Rate %', 'Trades', 'Exposure %'
        ])

    rows = []

    for symbol, result in results.items():
        m = result.metrics
        category = category_map.get(symbol, 'Unknown') if category_map else 'Unknown'

        # Compute benchmark metrics
        bench_nav = np.exp(result.benchmark_returns.cumsum())
        bench_ret = bench_nav.iloc[-1] - 1
        n_years = len(result.benchmark_returns) / 252
        bench_annual = (1 + bench_ret) ** (1/n_years) - 1 if n_years > 0 else 0

        rows.append({
            'Symbol': symbol,
            'Category': category,
            'Strategy Annual %': m.annual_return * 100,
            'Benchmark Annual %': bench_annual * 100,
            'Excess Return %': (m.annual_return - bench_annual) * 100,
            'Sharpe': m.sharpe_ratio,
            'Sortino': m.sortino_ratio,
            'Max DD %': m.max_drawdown * 100,
            'Calmar': m.calmar_ratio,
            'Win Rate %': m.win_rate * 100,
            'Trades': m.num_trades,
            'Exposure %': m.exposure * 100
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('Category')

    return df


def generate_yearly_report(
    results: Dict[str, BacktestResult]
) -> pd.DataFrame:
    """
    Generate yearly breakdown report.
    """
    if not results:
        return pd.DataFrame(columns=['Symbol', 'Year', 'Return %', 'Sharpe', 'Max DD %', 'Win Rate %'])

    rows = []

    for symbol, result in results.items():
        for year, m in result.yearly_metrics.items():
            rows.append({
                'Symbol': symbol,
                'Year': year,
                'Return %': m.annual_return * 100,
                'Sharpe': m.sharpe_ratio,
                'Max DD %': m.max_drawdown * 100,
                'Win Rate %': m.win_rate * 100
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(['Symbol', 'Year'])

    return df


def generate_symbol_yearly_report(
    results: Dict[str, BacktestResult],
    category_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Generate yearly report per symbol with benchmark comparison.
    """
    if not results:
        return pd.DataFrame(columns=[
            'Symbol', 'Category', 'Year', 'Strategy Return %',
            'Benchmark Return %', 'Alpha %', 'Sharpe', 'Max DD %'
        ])

    rows = []

    for symbol, result in results.items():
        category = category_map.get(symbol, 'Unknown') if category_map else 'Unknown'
        bench_returns = result.benchmark_returns

        for year, m in result.yearly_metrics.items():
            bench_year = bench_returns[bench_returns.index.year == year]
            if bench_year.empty:
                continue

            bench_return = float(np.exp(bench_year.sum()) - 1)

            rows.append({
                'Symbol': symbol,
                'Category': category,
                'Year': year,
                'Strategy Return %': m.annual_return * 100,
                'Benchmark Return %': bench_return * 100,
                'Alpha %': (m.annual_return - bench_return) * 100,
                'Sharpe': m.sharpe_ratio,
                'Max DD %': m.max_drawdown * 100,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(['Category', 'Symbol', 'Year'])
    return df


def generate_category_yearly_report(
    symbol_yearly: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate yearly results by category.
    """
    if symbol_yearly.empty:
        return pd.DataFrame(columns=[
            'Category', 'Year', 'Strategy Return %', 'Benchmark Return %',
            'Alpha %', 'Sharpe', 'Max DD %'
        ])

    grouped = symbol_yearly.groupby(['Category', 'Year']).agg({
        'Strategy Return %': 'mean',
        'Benchmark Return %': 'mean',
        'Alpha %': 'mean',
        'Sharpe': 'mean',
        'Max DD %': 'mean',
    }).reset_index()

    return grouped.sort_values(['Category', 'Year'])
