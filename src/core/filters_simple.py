"""
Detects market singularities.
High Curvature = Broken Manifold = Crash.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RicciCurvatureFilter(nn.Module):
    """
    Detects market singularities.
    High Curvature = Broken Manifold = Crash.
    """
    def __init__(self, dim):
        super().__init__()
        self.metric = nn.Linear(dim, 1)
    
    def forward(self, state):
        # R_ij approx via learned metric contraction
        curvature = F.softplus(self.metric(state))
        return curvature

