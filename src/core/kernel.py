"""
Syntropy Quant Kernel

Main trading kernel that integrates:
1. Filter: Curvature-based singularity detection
2. Scaling: Multi-timeframe renormalization
3. Dynamics: Symplectic Hamiltonian evolution
4. Navigation: Geodesic path for optimal positioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from .physics import DissipativeSymplecticUnit, PhaseSpaceEncoder
from .filters import RicciCurvatureFilter, SurpriseFilter, VolatilityRegimeDetector


@dataclass
class KernelOutput:
    """Output from kernel processing"""
    signal: float          # -1 (short) to +1 (long)
    confidence: float      # 0 to 1
    curvature: float       # Market stress indicator
    regime: str            # Market regime
    energy: float          # System energy level
    damping: float         # Current damping coefficient


@dataclass
class PositionSignal:
    """Trading signal with risk adjustments"""
    raw_signal: float
    risk_adjusted_signal: float
    position_size: float
    stop_loss: float
    take_profit: float


class SyntropyQuantKernel(nn.Module):
    """
    Physics-based Quantitative Trading Kernel

    Core philosophy: Market is a dissipative symplectic dynamical system.

    The kernel maintains an internal state in phase space (q, p) and
    evolves it using physics-constrained dynamics.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        curvature_threshold: float = 2.0,
        risk_free_rate: float = 0.05
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.risk_free_rate = risk_free_rate

        # Pillar I: Filter - Curvature detection
        self.curvature_filter = RicciCurvatureFilter(
            hidden_dim,
            curvature_threshold
        )

        # Pillar II: Scaling - Encode to phase space
        self.encoder = PhaseSpaceEncoder(input_dim, hidden_dim // 2)

        # Pillar III: Dynamics - Symplectic evolution
        self.dynamics = DissipativeSymplecticUnit(hidden_dim)

        # Pillar IV: Navigation - Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 3)  # [Long, Neutral, Short]
        )

        # Confidence estimation
        self.confidence_head = nn.Linear(hidden_dim, 1)

        # State variables
        self.state_q = None
        self.state_p = None
        self.last_energy = None

        # Surprise filter for data selection
        self.surprise_filter = SurpriseFilter(threshold_k=2.0)

        # Volatility regime detector
        self.vol_detector = VolatilityRegimeDetector()

    def reset_state(self, batch_size: int = 1, device: str = 'cpu'):
        """Reset internal state"""
        self.state_q = torch.zeros(batch_size, self.hidden_dim // 2).to(device)
        self.state_p = torch.zeros(batch_size, self.hidden_dim // 2).to(device)
        self.last_energy = None

    def forward(
        self,
        x_market: torch.Tensor,
        volume_clock_dt: torch.Tensor
    ) -> KernelOutput:
        """
        Process market observation through the kernel.

        Args:
            x_market: Market features [batch, input_dim]
            volume_clock_dt: Time step (volume-weighted)

        Returns:
            KernelOutput with signal and diagnostics
        """
        batch_size = x_market.shape[0]
        device = x_market.device

        # Initialize state if needed
        if self.state_q is None:
            self.reset_state(batch_size, device)

        # 1. Encode observation to phase space
        obs_q, obs_p = self.encoder(x_market)

        # 2. Data assimilation (Kalman-like blending)
        # Trust internal momentum more than noisy observation
        alpha_q, alpha_p = 0.5, 0.2
        self.state_q = (1 - alpha_q) * self.state_q + alpha_q * obs_q
        self.state_p = (1 - alpha_p) * self.state_p + alpha_p * obs_p

        # 3. Evolve dynamics
        self.state_q, self.state_p = self.dynamics(
            self.state_q,
            self.state_p,
            volume_clock_dt
        )

        # Combine state
        state_combined = torch.cat([self.state_q, self.state_p], dim=-1)

        # 4. Compute energy
        energy = self.dynamics.compute_hamiltonian(self.state_q, self.state_p)
        self.last_energy = energy

        # 5. Curvature filter (singularity detection)
        curvature, regime = self.curvature_filter(state_combined, energy)

        # 6. Policy head - generate signal
        logits = self.policy_head(state_combined)
        probs = F.softmax(logits, dim=-1)

        # Net signal: Long probability - Short probability
        raw_signal = (probs[:, 0] - probs[:, 2]).mean().item()

        # 7. Confidence estimation
        confidence = torch.sigmoid(self.confidence_head(state_combined)).mean().item()

        # 8. Risk adjustment based on curvature
        curv_val = curvature.mean().item()
        if curv_val > 2.0:
            # High curvature = high risk = reduce exposure
            adjusted_signal = raw_signal * 0.1
        elif curv_val > 1.5:
            adjusted_signal = raw_signal * 0.5
        else:
            adjusted_signal = raw_signal

        return KernelOutput(
            signal=adjusted_signal,
            confidence=confidence,
            curvature=curv_val,
            regime=regime,
            energy=energy.mean().item(),
            damping=self.dynamics.get_damping()
        )

    def process_step(
        self,
        features: np.ndarray,
        dt: float = 1.0
    ) -> KernelOutput:
        """
        Process a single time step (numpy interface).

        Args:
            features: Market features as numpy array
            dt: Time step

        Returns:
            KernelOutput
        """
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        dt_tensor = torch.tensor([[dt]], dtype=torch.float32)

        with torch.no_grad():
            return self.forward(x, dt_tensor)


class RiskManager:
    """
    Risk management layer.

    Implements position sizing, stop-loss, and regime-based adjustments.
    """

    def __init__(
        self,
        max_position: float = 1.0,
        max_drawdown: float = 0.1,
        vol_target: float = 0.15
    ):
        self.max_position = max_position
        self.max_drawdown = max_drawdown
        self.vol_target = vol_target

        self.peak_nav = 1.0
        self.current_nav = 1.0
        self.returns_history = []

    def update_nav(self, nav: float):
        """Update NAV and track drawdown"""
        self.current_nav = nav
        self.peak_nav = max(self.peak_nav, nav)

    def current_drawdown(self) -> float:
        """Current drawdown from peak"""
        return 1.0 - self.current_nav / self.peak_nav

    def compute_position_size(
        self,
        signal: float,
        curvature: float,
        realized_vol: float
    ) -> PositionSignal:
        """
        Compute risk-adjusted position size.

        Uses volatility targeting and drawdown control.
        """
        # Volatility scaling
        if realized_vol > 0:
            vol_scalar = self.vol_target / realized_vol
        else:
            vol_scalar = 1.0
        vol_scalar = np.clip(vol_scalar, 0.1, 3.0)

        # Curvature penalty
        curv_scalar = max(0.1, 1.0 - 0.3 * max(0, curvature - 1.0))

        # Drawdown control
        dd = self.current_drawdown()
        if dd > self.max_drawdown * 0.5:
            dd_scalar = max(0.1, 1.0 - (dd / self.max_drawdown))
        else:
            dd_scalar = 1.0

        # Final position size
        raw_size = abs(signal) * self.max_position
        adjusted_size = raw_size * vol_scalar * curv_scalar * dd_scalar
        adjusted_size = np.clip(adjusted_size, 0, self.max_position)

        # Direction
        position = np.sign(signal) * adjusted_size

        # Stop-loss and take-profit based on volatility
        stop_loss = -2.0 * realized_vol  # 2 sigma stop
        take_profit = 3.0 * realized_vol  # 3 sigma target

        return PositionSignal(
            raw_signal=signal,
            risk_adjusted_signal=position,
            position_size=adjusted_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )


class MultiTimeframeKernel(nn.Module):
    """
    Multi-timeframe kernel using renormalization group principles.

    Aggregates signals across multiple time scales to identify
    scale-invariant patterns.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        timeframes: List[int] = [1, 5, 20, 60]
    ):
        super().__init__()
        self.timeframes = timeframes

        # Kernel for each timeframe
        self.kernels = nn.ModuleList([
            SyntropyQuantKernel(input_dim, hidden_dim)
            for _ in timeframes
        ])

        # Aggregation weights (learnable)
        self.weights = nn.Parameter(torch.ones(len(timeframes)) / len(timeframes))

        # Cross-scale interaction
        self.cross_scale = nn.Linear(len(timeframes), 1)

    def forward(
        self,
        x_multi: List[torch.Tensor],
        dt: torch.Tensor
    ) -> KernelOutput:
        """
        Process multi-timeframe data.

        Args:
            x_multi: List of features at different timeframes
            dt: Base time step
        """
        outputs = []
        for i, (kernel, x) in enumerate(zip(self.kernels, x_multi)):
            # Scale dt for each timeframe
            scaled_dt = dt * self.timeframes[i]
            out = kernel(x, scaled_dt)
            outputs.append(out)

        # Weighted aggregation
        weights = F.softmax(self.weights, dim=0)
        signals = torch.tensor([o.signal for o in outputs])
        aggregated_signal = (weights * signals).sum().item()

        # Cross-scale consistency check
        signal_std = signals.std().item()
        consistency = 1.0 / (1.0 + signal_std)  # High consistency if signals agree

        # Use highest curvature (most conservative)
        max_curvature = max(o.curvature for o in outputs)
        dominant_regime = outputs[0].regime  # Use shortest timeframe

        return KernelOutput(
            signal=aggregated_signal * consistency,
            confidence=consistency,
            curvature=max_curvature,
            regime=dominant_regime,
            energy=np.mean([o.energy for o in outputs]),
            damping=np.mean([o.damping for o in outputs])
        )
