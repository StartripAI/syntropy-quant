import torch
import torch.nn as nn

class SyntropyQuantKernel(nn.Module):
    """
    Syntropy Quant v4.0 - Physics Momentum Kernel
    """
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.policy = nn.Linear(hidden_dim, 3)
        self.regime_gate = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.encoder(x)
        logits = self.policy(h)
        regime_prob = torch.sigmoid(self.regime_gate(h))
        return logits, regime_prob

    def process_step(self, x):
        logits, regime_prob = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        
        # Result adapter
        from collections import namedtuple
        Result = namedtuple('Result', ['signal', 'regime', 'confidence'])
        
        signal = (probs[0, 2] - probs[0, 0]).item()
        regime_val = regime_prob.item()
        
        regime_label = 'stable'
        if regime_val > 0.7: regime_label = 'bubble'
        elif regime_val > 0.4: regime_label = 'transition'
        
        return Result(
            signal=signal,
            regime=regime_label,
            confidence=probs.max().item()
        )
