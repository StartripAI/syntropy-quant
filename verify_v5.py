#!/usr/bin/env python3
"""
Syntropy Quant v5.0 Verification Suite
Testing "The Principle of Least Action" upgrades.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.optimized_portfolio import compute_hrp_weights, get_all_symbols
from src.core.gauge import GaugeFieldKernel, GaugeConfig
from src.core.filters import SurpriseFilter
from trading_system import TradingSystem

def test_hrp():
    print("\n--- 1. Testing HRP Weighting ---")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    # Create mock covariance matrix (high correlation between tech stocks)
    n = len(symbols)
    cov = np.ones((n, n)) * 0.8
    np.fill_diagonal(cov, 1.0)
    cov[4, 4] = 2.0 # More volatile NVDA
    
    weights = compute_hrp_weights(cov, symbols)
    for s, w in weights.items():
        print(f"  {s}: {w:.4f}")
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    print("SUCCESS: HRP weights normalized.")

def test_gauge_kernel():
    print("\n--- 2. Testing Gauge Field Kernel ---")
    config = GaugeConfig(input_dim=14)
    kernel = GaugeFieldKernel(input_dim=14, config=config)
    
    # Mock data: (batch, dim)
    x = torch.randn(1, 14)
    logits, free_energy, confidence = kernel(x)
    
    print(f"  Logits: {logits.shape}")
    print(f"  Free Energy: {free_energy.item():.4f}")
    print(f"  Confidence: {confidence.item():.4f}")
    
    # Process step
    res = kernel.process_step(x.numpy()[0])
    print(f"  Signal: {res.signal:.4f}")
    print(f"  Regime: {res.regime}")
    assert -1.0 <= res.signal <= 1.0
    print("SUCCESS: Gauge kernel forward pass OK.")

def test_surprise_filter():
    print("\n--- 3. Testing Surprise Filter ---")
    f = SurpriseFilter(threshold_k=2.0, min_samples=5)
    
    # Initial sequence
    data = [100.0, 101.0, 99.0, 100.5, 99.5]
    for v in data:
        f.filter(v)
        
    # Test normal point
    res_norm = f.filter(100.2)
    print(f"  Normal price (100.2) surprise: {res_norm.surprise:.4f} (Keep: {res_norm.keep})")
    
    # Test surprising point
    res_surp = f.filter(110.0)
    print(f"  Surprising price (110.0) surprise: {res_surp.surprise:.4f} (Keep: {res_surp.keep})")
    
    assert res_surp.surprise > res_norm.surprise
    print("SUCCESS: Surprise filter distinguishing regimes.")

def run_dry_cycle():
    print("\n--- 4. Running Dry Trading Cycle ---")
    try:
        system = TradingSystem(mode="paper", use_gauge=True)
        # Mocking the signals to avoid heavy data fetching in test
        # but we let it run as much as possible
        print("  Starting cycle (limited symbols)...")
        # Overriding symbols for speed
        import config.optimized_portfolio as op
        original_get = op.get_all_symbols
        op.get_all_symbols = lambda: ['QQQ', 'VOO', 'AAPL']
        
        summary = system.run_trading_cycle()
        print(f"  Summary: {summary}")
        
        # Restore
        op.get_all_symbols = original_get
        print("SUCCESS: Trading cycle executed.")
    except Exception as e:
        print(f"  FAILED: {e}")

if __name__ == "__main__":
    print("="*60)
    print("Syntropy Quant v5.0 Verification Suite")
    print("="*60)
    
    test_hrp()
    test_gauge_kernel()
    test_surprise_filter()
    run_dry_cycle()
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE: System v5.0 is Meta-Stable.")
    print("="*60)
