import torch
import torch.nn as nn
import torch.nn.functional as F
from ..physics.symplectic import SymplecticEuler
from ..physics.gauge_field import YangMillsField
from ..geometry.berry_phase import BerryCurvature

class SyntropyQuantKernelV5(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, latent_dim=16):
        super().__init__()
        
        # 1. Manifold Embedding (Features -> Phase Space)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2) # Split into [q, p]
        )
        
        # 2. Dual Physics Engines
        self.symplectic = SymplecticEuler(latent_dim)   # Conservation
        self.gauge_field = YangMillsField(latent_dim * 2) # Expansion
        
        # 3. Geometry Observer
        self.berry = BerryCurvature(latent_dim * 2)
        
        # 4. Regime Gate (The Order Parameter)
        # Determines if we are in Solid (Symplectic) or Plasma (Gauge) state
        self.regime_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid() # 0 = Stable, 1 = Chaos
        )
        
        # 5. Policy Head
        self.policy = nn.Linear(latent_dim * 2, 3) # Long/Neutral/Short

    def forward(self, x, dt=1.0):
        # Embed
        features = self.encoder(x)
        half = features.shape[1] // 2
        q, p = features[:, :half], features[:, half:]
        state = torch.cat([q, p], dim=-1)
        
        # Calculate Regime Alpha (Phase Transition)
        alpha = self.regime_gate(state)
        
        # Path A: Symplectic Evolution
        q_sym, p_sym = self.symplectic(q, p, dt)
        state_sym = torch.cat([q_sym, p_sym], dim=-1)
        
        # Path B: Gauge Field Evolution
        state_gauge = self.gauge_field(state, dt)
        
        # Superposition (Quantum-Classical Mix)
        state_next = (1 - alpha) * state_sym + alpha * state_gauge
        
        # Geometric Check (Berry Phase)
        curvature = self.berry(state, state_next)
        
        # Decision
        logits = self.policy(state_next)
        
        return {
            "logits": logits,
            "regime": alpha,      # 0..1
            "curvature": curvature,
            "state": state_next
        }
