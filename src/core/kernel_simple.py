"""
Syntropy Quant Kernel - Simplified Version
Physics-based Quantitative Trading Framework
"""

import torch
import torch.nn as nn
from .physics_simple import DissipativeSymplecticUnit
from .filters_simple import RicciCurvatureFilter


class SyntropyQuantKernel(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.SiLU(),
            nn.Linear(64, hidden_dim)
        )
        self.dsu = DissipativeSymplecticUnit(hidden_dim)
        self.filter = RicciCurvatureFilter(hidden_dim)
        self.policy = nn.Linear(hidden_dim, 3) # Long/Neutral/Short
        
        self.q = None
        self.p = None

    def forward(self, x, dt):
        batch = x.shape[0]
        if self.q is None:
            self.q = torch.zeros(batch, self.hidden_dim//2).to(x.device)
            self.p = torch.zeros(batch, self.hidden_dim//2).to(x.device)
            
        # Embed market data into Phase Space
        feat = self.encoder(x)
        obs_q, obs_p = torch.chunk(feat, 2, dim=-1)
        
        # Data Assimilation (Kalman Filter style)
        self.q = 0.6 * self.q + 0.4 * obs_q
        self.p = 0.8 * self.p + 0.2 * obs_p
        
        # Physics Evolution
        self.q, self.p = self.dsu(self.q, self.p, dt)
        state = torch.cat([self.q, self.p], dim=-1)
        
        # Risk Check
        risk = self.filter(state)
        logits = self.policy(state)
        
        return logits, risk

