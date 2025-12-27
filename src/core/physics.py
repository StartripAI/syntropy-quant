"""
Physics Engine: Dissipative Symplectic Dynamics

Core principle: Market as a dissipative Hamiltonian system
- Position (q): Mispricing / Potential energy
- Momentum (p): Order flow / Kinetic energy
- Damping (gamma): Market efficiency friction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DissipativeSymplecticUnit(nn.Module):
    """
    Dissipative Symplectic Unit (DSU)

    Replaces LSTM/GRU with physics-constrained dynamics.
    Enforces approximate energy conservation with friction.

    Hamilton's equations with damping:
        dq/dt = dH/dp
        dp/dt = -dH/dq - gamma * p + noise
    """

    def __init__(self, hidden_dim: int, learnable_damping: bool = True):
        super().__init__()
        self.dim = hidden_dim // 2  # Split for q and p

        # Force field: -dH/dq (restoring force from potential)
        self.force_field = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.Tanh(),
            nn.Linear(self.dim * 2, self.dim)
        )

        # Velocity field: dH/dp (kinematic velocity from momentum)
        self.velocity_field = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.Tanh(),
            nn.Linear(self.dim * 2, self.dim)
        )

        # Learnable damping coefficient
        if learnable_damping:
            self.damping = nn.Parameter(torch.tensor([0.1]))
        else:
            self.register_buffer('damping', torch.tensor([0.1]))

        # Energy scaling for stability
        self.energy_scale = nn.Parameter(torch.tensor([1.0]))

    def compute_hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute total energy H = T(p) + V(q)"""
        kinetic = 0.5 * torch.sum(p ** 2, dim=-1, keepdim=True)
        potential = 0.5 * torch.sum(q ** 2, dim=-1, keepdim=True)
        return self.energy_scale * (kinetic + potential)

    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        dt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Symplectic Euler integration step.

        Uses symplectic integrator to preserve phase space structure.
        This prevents numerical drift in long trajectories.
        """
        # Ensure dt is properly shaped
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)

        # 1. Compute force from position (potential gradient)
        force = self.force_field(q)

        # 2. Update momentum with force and friction
        # dp = (F - gamma * p) * dt
        friction = torch.abs(self.damping) * p
        dp = (force - friction) * dt
        p_new = p + dp

        # 3. Compute velocity from new momentum
        velocity = self.velocity_field(p_new)

        # 4. Update position with new velocity
        # dq = v(p_new) * dt
        dq = velocity * dt
        q_new = q + dq

        return q_new, p_new

    def get_damping(self) -> float:
        """Return current damping coefficient"""
        return self.damping.item()


class HamiltonianNetwork(nn.Module):
    """
    Hamiltonian Neural Network

    Learns a Hamiltonian function H(q, p) and derives dynamics
    from its gradients. Guarantees energy conservation by construction.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim

        # Network to learn H(q, p)
        self.hamiltonian_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian value"""
        state = torch.cat([q, p], dim=-1)
        return self.hamiltonian_net(state)

    def time_derivative(
        self,
        q: torch.Tensor,
        p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Hamilton's equations:
        dq/dt = dH/dp
        dp/dt = -dH/dq
        """
        q.requires_grad_(True)
        p.requires_grad_(True)

        H = self.forward(q, p)

        dH_dq = torch.autograd.grad(H.sum(), q, create_graph=True)[0]
        dH_dp = torch.autograd.grad(H.sum(), p, create_graph=True)[0]

        dq_dt = dH_dp   # Hamilton's first equation
        dp_dt = -dH_dq  # Hamilton's second equation

        return dq_dt, dp_dt


class SymplecticIntegrator(nn.Module):
    """
    Higher-order symplectic integrator (Verlet/Leapfrog)

    More accurate than Euler for long-term predictions.
    """

    def __init__(self, hamiltonian_net: HamiltonianNetwork, damping: float = 0.1):
        super().__init__()
        self.H = hamiltonian_net
        self.damping = damping

    def leapfrog_step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Leapfrog (Stormer-Verlet) integration:
        1. Half step in p
        2. Full step in q
        3. Half step in p
        """
        # Half step in momentum
        _, dp_dt = self.H.time_derivative(q, p)
        p_half = p + 0.5 * dt * (dp_dt - self.damping * p)

        # Full step in position
        dq_dt, _ = self.H.time_derivative(q, p_half)
        q_new = q + dt * dq_dt

        # Half step in momentum with new position
        _, dp_dt_new = self.H.time_derivative(q_new, p_half)
        p_new = p_half + 0.5 * dt * (dp_dt_new - self.damping * p_half)

        return q_new, p_new


class PhaseSpaceEncoder(nn.Module):
    """
    Encodes market observations into phase space (q, p).

    Maps raw features to:
    - q: Mispricing coordinates (potential)
    - p: Order flow coordinates (momentum)
    """

    def __init__(self, input_dim: int, phase_dim: int):
        super().__init__()
        self.phase_dim = phase_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, phase_dim * 2)  # Output q and p
        )

        # Separate encoders for position and momentum
        self.q_transform = nn.Linear(phase_dim, phase_dim)
        self.p_transform = nn.Linear(phase_dim, phase_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observations to phase space"""
        features = self.encoder(x)
        q_raw, p_raw = torch.chunk(features, 2, dim=-1)

        q = self.q_transform(q_raw)
        p = self.p_transform(p_raw)

        return q, p
