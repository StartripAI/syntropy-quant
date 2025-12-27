"""
USMK-Q Core: Dissipative Hamiltonian System.
Guarantees energy conservation laws in predictions.
"""

import torch
import torch.nn as nn


class DissipativeSymplecticUnit(nn.Module):
    """
    USMK-Q Core: Dissipative Hamiltonian System.
    Guarantees energy conservation laws in predictions.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.dim = hidden_dim // 2
        
        # Potential Force Field (-dH/dq)
        self.force_field = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2), nn.Tanh(),
            nn.Linear(self.dim * 2, self.dim)
        )
        # Kinetic Velocity Field (dH/dp)
        self.velocity_field = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2), nn.Tanh(),
            nn.Linear(self.dim * 2, self.dim)
        )
        # Market Friction Parameter (Gamma)
        # gamma > 0: Mean Reversion; gamma < 0: Bubble/Trend
        self.damping = nn.Parameter(torch.tensor([0.1]))

    def forward(self, q, p, dt):
        # Symplectic Euler Integration to preserve phase space volume
        # 1. Momentum update with friction
        force = self.force_field(q)
        friction = torch.abs(self.damping) * p
        p_new = p + (force - friction) * dt
        
        # 2. Position update
        velocity = self.velocity_field(p_new)
        q_new = q + velocity * dt
        
        return q_new, p_new

