import torch
import torch.nn as nn


class HamiltonianNetwork(nn.Module):
    """
    Hamiltonian Neural Network - learns energy function H(q, p).
    Used for computing physical quantities like energy and forces.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian H(q, p)"""
        state = torch.cat([q, p], dim=-1)
        return self.net(state).squeeze(-1)

    def compute_energy(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Alias for forward - compute total energy"""
        return self.forward(q, p)


class DissipativeSymplecticUnit(nn.Module):
    """
    USMK-Q Core v4.0: Semi-Implicit Symplectic Integrator.
    Crucial Fix: Allows Gamma < 0 (Negative Damping) to model Trends/Bubbles.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.dim = hidden_dim // 2
        
        # Force Field (-dH/dq)
        self.force_net = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2), nn.SiLU(),
            nn.Linear(self.dim * 2, self.dim)
        )
        
        # Velocity Field (dH/dp)
        self.velocity_net = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2), nn.SiLU(),
            nn.Linear(self.dim * 2, self.dim)
        )
        
        # Adaptive Damping Gamma(state)
        # Tanh output [-1, 1] * scale. 
        # Negative = Energy Injection (Trend). Positive = Dissipation (Mean Rev).
        self.damping_net = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.Tanh(),
            nn.Linear(32, 1), nn.Tanh()
        )
        self.damping_scale = nn.Parameter(torch.tensor([1.0]))

    def forward(self, q, p, dt):
        state = torch.cat([q, p], dim=-1)
        gamma = self.damping_net(state) * self.damping_scale
        
        # Semi-Implicit Symplectic Euler
        # 1. Update Momentum: p(t+1) = p(t) + F(q(t)) * dt - Friction
        force = self.force_net(q)
        friction = gamma * p
        p_new = p + (force - friction) * dt
        
        # 2. Update Position: q(t+1) = q(t) + v(p(t+1)) * dt
        # Using p_new ensures symplectic structure preservation
        velocity = self.velocity_net(p_new)
        q_new = q + velocity * dt
        
        return q_new, p_new, gamma
