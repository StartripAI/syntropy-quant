"""
Filtering Module: Singularity Detection and Surprise Filtering

Core principle: Free Energy Principle - only process surprising data
Detect market regime changes through curvature and entropy measures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class FilterResult:
    """Result of filtering operation"""
    keep: bool
    surprise: float
    curvature: float
    regime: str  # 'normal', 'high_vol', 'crisis', 'bubble'


class RicciCurvatureFilter(nn.Module):
    """
    Ricci Curvature Filter

    Detects singularities in the market manifold.
    High curvature = Structural instability (crisis, bubble)

    When curvature diverges, physical laws break down.
    Action: Reduce exposure or exit.
    """

    def __init__(self, hidden_dim: int, curvature_threshold: float = 2.0):
        super().__init__()
        self.threshold = curvature_threshold

        # Curvature estimation network
        self.curvature_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # Running statistics for normalization
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_var', torch.tensor(1.0))
        self.register_buffer('count', torch.tensor(0.0))

    def update_stats(self, curvature: torch.Tensor):
        """Update running statistics"""
        with torch.no_grad():
            batch_mean = curvature.mean()
            batch_var = curvature.var()

            # Exponential moving average
            alpha = min(0.1, 1.0 / (self.count + 1))
            self.running_mean = (1 - alpha) * self.running_mean + alpha * batch_mean
            self.running_var = (1 - alpha) * self.running_var + alpha * batch_var
            self.count += 1

    def forward(
        self,
        state: torch.Tensor,
        energy: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, str]:
        """
        Compute curvature and regime classification.

        Returns:
            curvature: Normalized curvature value
            regime: Market regime classification
        """
        # Base curvature from state
        raw_curvature = F.softplus(self.curvature_net(state))

        # Add energy contribution if available
        if energy is not None:
            raw_curvature = raw_curvature + 0.1 * energy

        # Normalize
        self.update_stats(raw_curvature)
        normalized = (raw_curvature - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)

        # Regime classification
        curv_val = normalized.mean().item()
        if curv_val > 3.0:
            regime = 'crisis'
        elif curv_val > 2.0:
            regime = 'high_vol'
        elif curv_val < -1.0:
            regime = 'bubble'
        else:
            regime = 'normal'

        return normalized, regime

    def should_exit(self, curvature: torch.Tensor) -> bool:
        """Check if curvature indicates need to exit"""
        return curvature.mean().item() > self.threshold


class SurpriseFilter:
    """
    Surprise Filter based on Free Energy Principle

    Only keeps data points that are "surprising" -
    i.e., significantly different from model predictions.

    Surprise S = -log P(observation | model)
    Keep if S > threshold
    """

    def __init__(
        self,
        threshold_k: float = 2.0,
        window_size: int = 100,
        min_samples: int = 20
    ):
        self.threshold_k = threshold_k
        self.window_size = window_size
        self.min_samples = min_samples

        # Statistics tracking
        self.values_buffer = []
        self.mean = 0.0
        self.std = 1.0

    def update_stats(self, value: float):
        """Update running statistics"""
        self.values_buffer.append(value)
        if len(self.values_buffer) > self.window_size:
            self.values_buffer.pop(0)

        if len(self.values_buffer) >= self.min_samples:
            arr = np.array(self.values_buffer)
            self.mean = np.mean(arr)
            self.std = np.std(arr) + 1e-8

    def compute_surprise(self, value: float) -> float:
        """
        Compute surprise score using Gaussian assumption.

        S = |value - mean| / std (standardized deviation)
        """
        if len(self.values_buffer) < self.min_samples:
            return 0.0

        z_score = abs(value - self.mean) / self.std
        return z_score

    def filter(self, value: float) -> FilterResult:
        """
        Filter a single value.

        Returns FilterResult with keep decision and metrics.
        """
        surprise = self.compute_surprise(value)
        keep = surprise > self.threshold_k

        # Update stats after filtering decision
        self.update_stats(value)

        return FilterResult(
            keep=keep,
            surprise=surprise,
            curvature=0.0,
            regime='normal'
        )

    def filter_batch(self, values: np.ndarray) -> Dict:
        """
        Filter a batch of values.

        Returns statistics about filtering.
        """
        results = []
        for v in values:
            results.append(self.filter(v))

        kept = sum(1 for r in results if r.keep)
        total = len(results)

        return {
            'kept': kept,
            'total': total,
            'filter_rate': 1.0 - kept / total if total > 0 else 0.0,
            'avg_surprise': np.mean([r.surprise for r in results]),
            'results': results
        }

    def reset(self):
        """Reset filter state"""
        self.values_buffer = []
        self.mean = 0.0
        self.std = 1.0


class VolatilityRegimeDetector:
    """
    Detects volatility regimes using rolling statistics.

    Regimes:
    - Low: vol < mean - 0.5*std
    - Normal: mean - 0.5*std <= vol <= mean + 1*std
    - High: vol > mean + 1*std
    - Extreme: vol > mean + 2*std
    """

    def __init__(self, window: int = 60, vol_window: int = 20):
        self.window = window
        self.vol_window = vol_window
        self.vol_history = []

    def update(self, returns: np.ndarray) -> str:
        """Update with new returns and return current regime"""
        # Compute realized volatility
        if len(returns) >= self.vol_window:
            vol = np.std(returns[-self.vol_window:]) * np.sqrt(252)
        else:
            vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.2

        self.vol_history.append(vol)
        if len(self.vol_history) > self.window:
            self.vol_history.pop(0)

        if len(self.vol_history) < 10:
            return 'normal'

        mean_vol = np.mean(self.vol_history)
        std_vol = np.std(self.vol_history) + 1e-8

        z = (vol - mean_vol) / std_vol

        if z > 2.0:
            return 'extreme'
        elif z > 1.0:
            return 'high'
        elif z < -0.5:
            return 'low'
        else:
            return 'normal'


class MarketPhaseDetector:
    """
    Detects market phase using price and momentum analysis.

    Phases:
    - accumulation: Low vol, sideways, building positions
    - markup: Trending up, increasing vol
    - distribution: High prices, divergences appearing
    - markdown: Trending down, high vol
    """

    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.prices = []
        self.volumes = []

    def update(self, price: float, volume: float) -> str:
        """Update and return current market phase"""
        self.prices.append(price)
        self.volumes.append(volume)

        if len(self.prices) > self.lookback:
            self.prices.pop(0)
            self.volumes.pop(0)

        if len(self.prices) < 20:
            return 'accumulation'

        prices = np.array(self.prices)
        volumes = np.array(self.volumes)

        # Trend analysis
        returns = np.diff(prices) / prices[:-1]
        trend = np.mean(returns[-20:])
        vol = np.std(returns[-20:])

        # Volume analysis
        vol_trend = np.mean(volumes[-10:]) / (np.mean(volumes[-30:-10]) + 1e-8)

        # Price position
        price_percentile = (prices[-1] - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-8)

        # Phase detection
        if trend > 0.001 and vol_trend > 1.1:
            return 'markup'
        elif trend < -0.001 and vol > 0.02:
            return 'markdown'
        elif price_percentile > 0.8 and vol_trend < 0.9:
            return 'distribution'
        else:
            return 'accumulation'
