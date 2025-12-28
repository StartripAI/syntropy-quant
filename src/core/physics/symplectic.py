import torch
import torch.nn as nn

class SymplecticEuler(nn.Module):
    """
    Hamiltonian Dynamics for 'Laminar Flow' (Stable Markets).
    Strictly preserves Phase Space Volume (Liouville's Theorem).
    Best for: KO, JNJ, Index Range.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Separable Hamiltonian H(q,p) = T(p) + V(q)
        # Potential Force Field (-dV/dq)
        self.V_net = nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, dim))
        # Kinetic Velocity Field (dT/dp)
        self.T_net = nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, dim))

    def forward(self, q, p, dt=1.0):
        # Symplectic Euler Integration (Energy Stable)
        # 1. Kick: p(t+1) = p(t) - grad(V)(q(t)) * dt
        force = -self.V_net(q)
        p_next = p + force * dt
        
        # 2. Drift: q(t+1) = q(t) + grad(T)(p(t+1)) * dt
        velocity = self.T_net(p_next)
        q_next = q + velocity * dt
        
        return q_next, p_next
