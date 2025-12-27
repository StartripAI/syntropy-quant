"""
Syntropy Quant Core - Physics-based Quantitative Finance Kernel
"""

from .physics import DissipativeSymplecticUnit, HamiltonianNetwork
from .kernel import SyntropyQuantKernel
from .gauge import GaugeFieldKernel, GaugeConfig
from .filters import RicciCurvatureFilter, SurpriseFilter

__all__ = [
    'DissipativeSymplecticUnit',
    'HamiltonianNetwork',
    'SyntropyQuantKernel',
    'GaugeFieldKernel',
    'GaugeConfig',
    'RicciCurvatureFilter',
    'SurpriseFilter'
]
