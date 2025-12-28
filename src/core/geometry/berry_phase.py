import torch
import torch.nn as nn

class BerryCurvature(nn.Module):
    """
    Calculates the Geometric Phase (Berry Phase) of market manifold.
    Non-zero Berry Phase implies 'Holonomy' -> Hidden Arbitrage / Risk.
    """
    def __init__(self, latent_dim):
        super().__init__()
        # The connection form A (Gauge Potential)
        self.connection = nn.Linear(latent_dim, latent_dim, bias=False)
        
    def forward(self, z, z_next):
        """
        Approximates Berry Curvature F_uv via discrete holonomy.
        High curvature = Market is twisting (Phase Transition).
        """
        # Tangent vector (Velocity on manifold)
        dz = z_next - z
        
        # Gauge field interaction A_mu * dx^mu
        A = self.connection(z)
        
        # Geometric Action
        # Represents the 'twisting' energy of the fiber bundle
        phase = torch.sum(A * dz, dim=-1, keepdim=True)
        
        # Curvature magnitude
        curvature = torch.abs(phase)
        return curvature
