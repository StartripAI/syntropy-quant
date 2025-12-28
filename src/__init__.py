"""
Syntropy Quant - Physics-based Quantitative Trading Framework v4.0
The Poincare Patch: Fixed NaNs, Negative Damping, Semi-Implicit Integrator

Core innovations:
- DSU (Dissipative Symplectic Unit): Semi-implicit symplectic integrator with negative damping
- Gauge Field Kernel: Path-integral based future state estimation
- Epsilon-stabilized features: NaN-safe numerical operations
"""

from .core import (
    SyntropyQuantKernel,
    DissipativeSymplecticUnit,
    HamiltonianNetwork,
    RiskManager,
    KernelOutput,
    GaugeFieldKernel,
    GaugeConfig,
    RicciCurvatureFilter,
    SurpriseFilter,
)

from .data import DataFetcher, FeatureBuilder, AssetCategory, ASSET_UNIVERSE

# Optional backtest imports (may not be needed for training)
try:
    from .backtest import BacktestEngine, BacktestResult, PerformanceMetrics
    _backtest_available = True
except ImportError:
    _backtest_available = False

__version__ = '4.0.0'

__all__ = [
    # Core
    'SyntropyQuantKernel',
    'DissipativeSymplecticUnit',
    'HamiltonianNetwork',
    'RiskManager',
    'KernelOutput',
    'GaugeFieldKernel',
    'GaugeConfig',
    'RicciCurvatureFilter',
    'SurpriseFilter',

    # Data
    'DataFetcher',
    'FeatureBuilder',
    'AssetCategory',
    'ASSET_UNIVERSE',
]

if _backtest_available:
    __all__.extend(['BacktestEngine', 'BacktestResult', 'PerformanceMetrics'])
