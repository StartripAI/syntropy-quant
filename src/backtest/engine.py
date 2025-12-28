"""
Backtesting Engine

Simulates trading strategies on historical data.
"""

import copy
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
    multi_scale: bool = False
    scale_windows: Tuple[Tuple[int, int, int], ...] = ((3, 10, 30), (5, 20, 60), (10, 40, 120))
    momentum_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    momentum_gate: float = 0.15
    coupling_strength: float = 0.65


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

    def _compute_momentum_score(self, feature_row: np.ndarray) -> float:
        if feature_row.shape[0] <= 10:
            return 0.0
        trend = feature_row[10]
        macd_hist = feature_row[6]
        order_flow = feature_row[9]
        w_trend, w_macd, w_flow = self.config.momentum_weights
        score = w_trend * trend + w_macd * macd_hist + w_flow * order_flow
        return float(np.tanh(score))

    def _apply_curvature_momentum_gate(
        self,
        signal: float,
        momentum_score: float,
        curvature: float
    ) -> float:
        if signal == 0.0:
            return 0.0
        gate = self.config.momentum_gate
        if curvature <= gate:
            return signal
        curv_intensity = (curvature - gate) / max(gate, 1e-6)
        curv_intensity = float(np.clip(curv_intensity, 0.0, 1.0))
        momentum_strength = min(1.0, abs(momentum_score))
        alignment = np.sign(momentum_score) * np.sign(signal)
        penalty = self.config.coupling_strength * curv_intensity * (1.0 - momentum_strength)
        if alignment < 0:
            penalty += self.config.coupling_strength * curv_intensity
        adjusted = signal * (1.0 - penalty)
        return float(np.clip(adjusted, -1.0, 1.0))

    @staticmethod
    def _combine_regimes(regimes: List[str]) -> str:
        if "crisis" in regimes:
            return "crisis"
        if "high_vol" in regimes:
            return "high_vol"
        return "normal"

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
        # Build features (single or multi-scale)
        if self.config.multi_scale:
            builders = [FeatureBuilder(*window) for window in self.config.scale_windows]
        else:
            builders = [self.feature_builder]

        feature_sets = [builder.build_features(df) for builder in builders]
        dt_series = builders[0].get_dt_series(df)

        # Initialize kernels (one per scale)
        base_kernel = self._build_kernel(input_dim=feature_sets[0].shape[1])
        if self.config.multi_scale:
            kernels = [copy.deepcopy(base_kernel) for _ in feature_sets]
        else:
            kernels = [base_kernel]

        # Apply normalization if available
        if hasattr(base_kernel, "feature_mean") and hasattr(base_kernel, "feature_std"):
            std = np.where(base_kernel.feature_std == 0, 1.0, base_kernel.feature_std)
            feature_sets = [(features - base_kernel.feature_mean) / std for features in feature_sets]

        for kernel in kernels:
            kernel.eval()

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
            # Get kernel output (multi-scale aggregation)
            dt_value = 1.0 if self.config.kernel_type == "gauge" else dt_series[t]
            signals_per_scale: List[float] = []
            confidences: List[float] = []
            curvatures_step: List[float] = []
            regimes_step: List[str] = []
            momentum_scores: List[float] = []

            for features, kernel in zip(feature_sets, kernels):
                output = kernel.process_step(features[t], dt_value)
                signals_per_scale.append(output.signal)
                confidences.append(output.confidence)
                curvatures_step.append(output.curvature)
                regimes_step.append(output.regime)
                momentum_scores.append(self._compute_momentum_score(features[t]))

            avg_confidence = float(np.mean(confidences)) if confidences else 0.0
            if confidences and float(np.sum(confidences)) > 0:
                combined_signal = float(np.average(signals_per_scale, weights=np.clip(confidences, 1e-6, None)))
            else:
                combined_signal = float(np.mean(signals_per_scale)) if signals_per_scale else 0.0

            # Compute realized volatility
            if t > 20:
                recent_returns = np.diff(np.log(prices[t-20:t]))
                realized_vol = np.std(recent_returns) * np.sqrt(252)
            else:
                realized_vol = 0.2

            # Signal smoothing and confidence gating
            effective_signal = combined_signal * avg_confidence
            if avg_confidence < self.config.confidence_threshold:
                effective_signal = 0.0

            signal_buffer.append(effective_signal)
            if len(signal_buffer) > self.config.signal_window:
                signal_buffer.pop(0)
            smoothed_signal = float(np.mean(signal_buffer))

            curvature = float(np.max(curvatures_step)) if curvatures_step else 0.0
            regime = self._combine_regimes(regimes_step)
            momentum_score = float(np.mean(momentum_scores)) if momentum_scores else 0.0
            smoothed_signal = self._apply_curvature_momentum_gate(
                smoothed_signal,
                momentum_score,
                curvature
            )

            # Risk-adjusted position
            pos_signal = risk_manager.compute_position_size(
                smoothed_signal,
                curvature,
                realized_vol
            )

            target_position = pos_signal.risk_adjusted_signal
            if regime == "crisis":
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
                    'signal': smoothed_signal,
                    'curvature': curvature,
                    'momentum': momentum_score,
                    'regime': regime
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
            curvatures[t] = curvature
            regimes[t] = regime

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
