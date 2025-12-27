"""
Syntropy Quant - Physics-based Quantitative Trading Framework

A unified framework that treats financial markets as dissipative
Hamiltonian dynamical systems operating on symplectic manifolds.
"""

from .core import (
    SyntropyQuantKernel,
    GaugeFieldKernel,
    DissipativeSymplecticUnit,
    HamiltonianNetwork,
    RicciCurvatureFilter,
    SurpriseFilter
)

from .backtest import BacktestEngine, BacktestResult, PerformanceMetrics
from .data import DataFetcher, AssetCategory, FeatureBuilder

__version__ = '1.0.0'

__all__ = [
    # Core
    'SyntropyQuantKernel',
    'GaugeFieldKernel',
    'DissipativeSymplecticUnit',
    'HamiltonianNetwork',
    'RicciCurvatureFilter',
    'SurpriseFilter',

    # Backtest
    'BacktestEngine',
    'BacktestResult',
    'PerformanceMetrics',

    # Data
    'DataFetcher',
    'AssetCategory',
    'FeatureBuilder',
]
