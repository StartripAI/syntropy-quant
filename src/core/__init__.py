"""
Syntropy Quant Core - Physics-based Quantitative Finance Kernel v4.0
"""

from .physics import DissipativeSymplecticUnit, HamiltonianNetwork
from .kernel import SyntropyQuantKernel, RiskManager, KernelOutput, PositionSignal
from .gauge import GaugeFieldKernel, GaugeConfig, GaugeKernelOutput
from .filters import RicciCurvatureFilter, SurpriseFilter

__all__ = [
    # Physics
    'DissipativeSymplecticUnit',
    'HamiltonianNetwork',

    # Kernel
    'SyntropyQuantKernel',
    'RiskManager',
    'KernelOutput',
    'PositionSignal',

    # Gauge
    'GaugeFieldKernel',
    'GaugeConfig',
    'GaugeKernelOutput',

    # Filters
    'RicciCurvatureFilter',
    'SurpriseFilter'
]
