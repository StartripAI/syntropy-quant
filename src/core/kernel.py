import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .physics import DissipativeSymplecticUnit


@dataclass
class KernelOutput:
    """Output from kernel processing step"""
    signal: float
    confidence: float
    curvature: float
    regime: str
    energy: float = 0.0
    damping: float = 0.0


@dataclass
class PositionSignal:
    """Risk-adjusted position signal"""
    raw_signal: float
    risk_adjusted_signal: float
    vol_scalar: float
    curvature_scalar: float


class RiskManager:
    """
    Risk Manager for position sizing and drawdown control.
    """
    def __init__(
        self,
        max_position: float = 1.0,
        max_drawdown: float = 0.2,
        vol_target: float = 0.15
    ):
        self.max_position = max_position
        self.max_drawdown = max_drawdown
        self.vol_target = vol_target
        self.peak_nav = 1.0
        self.current_nav = 1.0

    def update_nav(self, nav: float):
        """Update current NAV for drawdown tracking"""
        self.current_nav = nav
        self.peak_nav = max(self.peak_nav, nav)

    def compute_position_size(
        self,
        signal: float,
        curvature: float,
        realized_vol: float
    ) -> PositionSignal:
        """Compute risk-adjusted position size"""
        # Vol scaling
        vol_scalar = min(1.0, self.vol_target / (realized_vol + 1e-8))

        # Curvature penalty (reduce position in high curvature regimes)
        curvature_scalar = 1.0 / (1.0 + curvature)

        # Drawdown control
        current_dd = (self.peak_nav - self.current_nav) / self.peak_nav
        if current_dd > self.max_drawdown * 0.5:
            dd_scalar = max(0.1, 1.0 - current_dd / self.max_drawdown)
        else:
            dd_scalar = 1.0

        # Final position
        risk_adjusted = signal * vol_scalar * curvature_scalar * dd_scalar
        risk_adjusted = float(np.clip(risk_adjusted, -self.max_position, self.max_position))

        return PositionSignal(
            raw_signal=signal,
            risk_adjusted_signal=risk_adjusted,
            vol_scalar=vol_scalar,
            curvature_scalar=curvature_scalar
        )


class SyntropyQuantKernel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Physics
        self.dsu = DissipativeSymplecticUnit(hidden_dim)
        
        # Policy
        self.policy = nn.Linear(hidden_dim, 3)
        
        self.q = None
        self.p = None

    def forward(self, x, dt=1.0):
        batch = x.shape[0]
        if self.q is None or self.q.shape[0] != batch:
            self.q = torch.zeros(batch, self.hidden_dim//2).to(x.device)
            self.p = torch.zeros(batch, self.hidden_dim//2).to(x.device)
        else:
            self.q = self.q.detach()
            self.p = self.p.detach()
            
        # Embed & Assimilate
        feat = self.encoder(x)
        obs_q, obs_p = torch.chunk(feat, 2, dim=-1)
        
        self.q = 0.6 * self.q + 0.4 * obs_q
        self.p = 0.6 * self.p + 0.4 * obs_p
        
        # Physics Step
        self.q, self.p, gamma = self.dsu(self.q, self.p, dt)
        
        # Decision
        state = torch.cat([self.q, self.p], dim=-1)
        logits = self.policy(state)

        return logits, gamma

    def process_step(self, features: np.ndarray, dt: float = 1.0) -> KernelOutput:
        """
        Process a single step for backtest engine compatibility.

        Args:
            features: Feature array for current timestep
            dt: Time delta

        Returns:
            KernelOutput with signal, confidence, curvature, regime
        """
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits, gamma = self.forward(x, dt)
            probs = torch.softmax(logits, dim=1)

            # Signal: Long prob - Short prob
            signal = float(probs[0, 2] - probs[0, 0])

            # Confidence from probability distribution
            confidence = float(probs.max())

            # Curvature from gamma magnitude
            gamma_val = float(gamma.abs().mean())
            curvature = gamma_val

            # Regime classification based on gamma
            if gamma_val > 0.5:
                regime = 'high_vol'
            elif gamma_val < -0.3:
                regime = 'bubble'  # Negative damping = energy injection
            else:
                regime = 'normal'

            # Energy estimate (kinetic + potential proxy)
            energy = float(torch.norm(self.p).item() if self.p is not None else 0.0)

        return KernelOutput(
            signal=signal,
            confidence=confidence,
            curvature=curvature,
            regime=regime,
            energy=energy,
            damping=float(gamma.mean())
        )
