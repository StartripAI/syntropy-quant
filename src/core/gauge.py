import torch
import torch.nn as nn
from .hybrid.kernel_v5 import SyntropyQuantKernelV5

class GaugeConfig:
    def __init__(self, input_dim=14):
        self.input_dim = input_dim

class GaugeFieldKernel(SyntropyQuantKernelV5):
    """
    Syntropy Quant v5.0 - Gauge Field Kernel
    Adapts the base v5 kernel to the interface expected by trading_system.py
    """
    def __init__(self, input_dim=14, config=None):
        # trading_system.py passes input_dim=14
        super().__init__(input_dim=input_dim, hidden_dim=128, latent_dim=16)
        
    def process_step(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        res = self.forward(x)
        logits = res['logits']
        probs = torch.softmax(logits, dim=-1)
        
        # Result adapter
        from collections import namedtuple
        Result = namedtuple('Result', ['signal', 'regime', 'confidence', 'curvature'])
        
        signal = (probs[0, 2] - probs[0, 0]).item()
        regime_val = res['regime'].item()
        
        # Mapping numerical regime to labels used in trading_loop
        regime_label = 'stable'
        if regime_val > 0.8: regime_label = 'crisis'
        elif regime_val > 0.5: regime_label = 'high_vol'
        
        return Result(
            signal=signal,
            regime=regime_label,
            confidence=probs.max().item(),
            curvature=res['curvature'].item()
        )
