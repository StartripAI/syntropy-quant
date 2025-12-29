import torch
import torch.nn as nn
import numpy as np
from collections import deque

class RealTimeFeatureBuffer:
    """
    High-performance circular buffer for MFT features.
    Maintains a rolling window of market state in memory.
    """
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        
    def push(self, bar):
        # bar: [open, high, low, close, volume]
        self.buffer.append(bar)
        
    def get_features(self):
        if len(self.buffer) < self.window_size:
            return None
            
        data = np.array(self.buffer)
        close = data[:, 3]
        vol = data[:, 4]
        
        # Fast Momentum (Physics: Kinetic Energy)
        ret = np.diff(np.log(close + 1e-8), prepend=close[0])
        # Volatility (Physics: Temperature)
        temp = np.std(ret[-20:])
        # Order Flow (Physics: Pressure)
        force = (close[-1] - close[0]) / (np.max(data[:, 1]) - np.min(data[:, 2]) + 1e-8)
        
        # Hurst (Simplified for Speed)
        hurst = 0.5 # Default
        
        # Combined MFT Vector (Dim=14 for Gauge Kernel compatibility)
        feat = np.zeros(14)
        feat[0] = ret[-1]
        feat[1] = temp
        feat[2] = force
        feat[3] = hurst
        # ... Fill other dimensions with second-order stats
        
        return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
