"""
Gauge Field Kernel

Implements a geometry-driven market model using:
- A learnable metric (risk geometry)
- A gauge potential (drift/flow field)
- Path-integral sampling (least-action expectation)
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GaugeConfig:
    """Configuration for the gauge field kernel."""
    input_dim: int
    metric_hidden: int = 64
    gauge_hidden: int = 64
    horizon: int = 5
    num_paths: int = 32
    dt: float = 1.0
    noise_scale: float = 0.6
    free_energy_scale: float = 0.12
    curvature_scale: float = 0.1
    signal_scale: float = 4.5
    signal_bias: float = 0.0
    regime_thresholds: Tuple[float, float] = (0.7, 1.4)


@dataclass
class GaugeKernelOutput:
    """Output from gauge kernel processing."""
    signal: float
    confidence: float
    curvature: float
    regime: str
    energy: float
    damping: float


class MarketManifold(nn.Module):
    """
    Learns the market geometry (metric) and gauge potential (flow field).
    """

    def __init__(self, state_dim: int, metric_hidden: int, gauge_hidden: int):
        super().__init__()
        self.dim = state_dim

        self.L_net = nn.Sequential(
            nn.Linear(state_dim, metric_hidden),
            nn.Tanh(),
            nn.Linear(metric_hidden, state_dim * state_dim),
        )

        self.A_net = nn.Sequential(
            nn.Linear(state_dim, gauge_hidden),
            nn.SiLU(),
            nn.Linear(gauge_hidden, state_dim),
        )

    def get_geometry(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        L = self.L_net(x).view(batch, self.dim, self.dim)
        g = torch.bmm(L, L.transpose(1, 2)) + 1e-5 * torch.eye(self.dim, device=x.device)
        g_inv = torch.linalg.inv(g)
        A = self.A_net(x)
        return g, g_inv, A


class PathIntegralEngine(nn.Module):
    """
    Approximates least-action paths using Monte Carlo sampling.
    """

    def __init__(self, config: GaugeConfig):
        super().__init__()
        self.config = config
        self.manifold = MarketManifold(
            config.input_dim,
            config.metric_hidden,
            config.gauge_hidden,
        )

    def onsager_machlup_action(
        self,
        x_t: torch.Tensor,
        x_next: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        g, _, A = self.manifold.get_geometry(x_t)
        velocity = (x_next - x_t) / max(dt, 1e-6)
        residual = velocity - A
        kinetic = 0.5 * torch.bmm(
            residual.unsqueeze(1),
            torch.bmm(g, residual.unsqueeze(-1)),
        ).squeeze()
        entropic = 0.01 * torch.norm(A, dim=1)
        return (kinetic + entropic) * dt

    def sample_future(
        self,
        x_current: torch.Tensor,
        horizon: Optional[int] = None,
        num_paths: Optional[int] = None,
        dt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        horizon = horizon if horizon is not None else self.config.horizon
        num_paths = num_paths if num_paths is not None else self.config.num_paths
        dt = dt if dt is not None else self.config.dt

        if num_paths <= 1:
            _, _, A = self.manifold.get_geometry(x_current)
            x_next = x_current + A * dt
            action = self.onsager_machlup_action(x_current, x_next, dt)
            return x_next, action

        batch, dim = x_current.shape
        device = x_current.device

        x = x_current.unsqueeze(1).repeat(1, num_paths, 1).view(-1, dim)
        actions = torch.zeros(batch * num_paths, device=device)

        for _ in range(horizon):
            g, _, A = self.manifold.get_geometry(x)

            drift = A
            jitter = 1e-4
            L = torch.linalg.cholesky(g + jitter * torch.eye(dim, device=device))
            eye = torch.eye(dim, device=device).expand(L.shape[0], dim, dim)
            L_inv = torch.linalg.solve_triangular(L, eye, upper=False)
            z = torch.randn_like(x).unsqueeze(-1)
            diffusion = torch.bmm(L_inv, z).squeeze(-1)

            x_next = x + drift * dt + diffusion * np.sqrt(dt) * self.config.noise_scale
            actions += self.onsager_machlup_action(x, x_next, dt)
            x = x_next

        x_final = x.view(batch, num_paths, dim)
        actions = actions.view(batch, num_paths)

        weights = F.softmax(-actions, dim=1).unsqueeze(-1)
        expected_state = torch.sum(x_final * weights, dim=1)

        partition = torch.sum(torch.exp(-actions), dim=1)
        free_energy = -torch.log(partition + 1e-8)

        return expected_state, free_energy


class GaugeFieldKernel(nn.Module):
    """
    Gauge Field Kernel.

    Produces trading signals from geometry-aware future state estimates.
    """

    def __init__(self, input_dim: int, config: Optional[GaugeConfig] = None):
        super().__init__()
        self.config = config or GaugeConfig(input_dim=input_dim)
        self.engine = PathIntegralEngine(self.config)

        self.policy_head = nn.Sequential(
            nn.Linear(self.config.input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        self.confidence_head = nn.Linear(self.config.input_dim, 1)

    def forward(self, x: torch.Tensor, dt: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_state, free_energy = self.engine.sample_future(x, dt=dt)
        logits = self.policy_head(expected_state)
        confidence = torch.sigmoid(self.confidence_head(expected_state)).squeeze(-1)
        return logits, free_energy, confidence

    def process_step(self, features: np.ndarray, dt: float = 1.0) -> GaugeKernelOutput:
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        logits, free_energy, confidence = self.forward(x, dt=dt)

        logit_delta = (logits[:, 2] - logits[:, 0]).mean() - self.config.signal_bias
        raw_signal = torch.tanh(logit_delta).item()

        risk = torch.sigmoid(free_energy * self.config.free_energy_scale).mean().item()
        scaled_signal = raw_signal * self.config.signal_scale
        adjusted_signal = float(np.clip(scaled_signal * (1.0 - risk), -1.0, 1.0))

        curvature = max(0.0, free_energy.mean().item() * self.config.curvature_scale)
        low_thr, high_thr = self.config.regime_thresholds
        if curvature > high_thr:
            regime = 'crisis'
        elif curvature > low_thr:
            regime = 'high_vol'
        else:
            regime = 'normal'

        return GaugeKernelOutput(
            signal=adjusted_signal,
            confidence=confidence.mean().item(),
            curvature=curvature,
            regime=regime,
            energy=free_energy.mean().item(),
            damping=0.0,
        )
