"""
Verify Syntropy v5.0 Regime Detection
Test if model can distinguish between Symplectic (stable) and Gauge (turbulent) regimes.
"""
import torch
import numpy as np
import pandas as pd
from src.core.hybrid.kernel_v5 import SyntropyQuantKernelV5
from src.data.features_v5 import FeatureBuilderV5

def generate_regime_test_samples(n_samples=100):
    """
    Generate test samples for two regimes:
    1. Stable: Mean-reverting, low vol
    2. Turbulent: Trending, high vol
    """
    samples_stable = []
    samples_turbulent = []
    
    # Generate stable samples (Symplectic regime)
    for _ in range(n_samples):
        # Low volatility, mean-reverting
        feat = np.array([
            np.random.normal(0, 0.005),      # Log return (small)
            np.random.uniform(0.008, 0.015),  # Vol (low)
            np.random.uniform(0.012, 0.02),   # Vol (medium)
            np.random.normal(0, 0.2),          # Flow (oscillating)
            np.random.uniform(0.3, 0.45),      # Hurst < 0.5 (mean rev)
            np.random.normal(0, 0.01),          # Potential (near 0)
            np.random.uniform(-0.5, 0.5),       # Phase (any)
        ])
        samples_stable.append(feat)
    
    # Generate turbulent samples (Gauge regime)
    for _ in range(n_samples):
        # High volatility, trending
        feat = np.array([
            np.random.normal(0.001, 0.02),   # Log return (trending)
            np.random.uniform(0.02, 0.05),   # Vol (high)
            np.random.uniform(0.03, 0.06),    # Vol (very high)
            np.random.normal(0.5, 0.8),      # Flow (strong direction)
            np.random.uniform(0.55, 0.8),     # Hurst > 0.5 (trend)
            np.random.uniform(-0.1, 0.2),     # Potential (drifting)
            np.random.uniform(-0.5, 0.5),       # Phase (any)
        ])
        samples_turbulent.append(feat)
    
    return np.array(samples_stable), np.array(samples_turbulent)

def verify_regime_separation():
    """
    Test if model can separate stable vs turbulent regimes.
    """
    print("\n" + "="*70)
    print("  SYNTROPY v5.0 REGIME VERIFICATION")
    print("="*70)
    
    # Load model
    model = SyntropyQuantKernelV5(input_dim=7, hidden_dim=128, latent_dim=16)
    try:
        model.load_state_dict(torch.load("models/syntropy_v5.pt", map_location='cpu'))
        print("✅ Model loaded\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    model.eval()
    
    # Generate test samples
    print("Generating test samples...")
    stable_samples, turbulent_samples = generate_regime_test_samples(n_samples=500)
    
    print(f"   Stable samples: {len(stable_samples)}")
    print(f"   Turbulent samples: {len(turbulent_samples)}")
    print()
    
    # Predict regimes
    with torch.no_grad():
        stable_regime = model(torch.tensor(stable_samples, dtype=torch.float32))['regime'].numpy()
        turbulent_regime = model(torch.tensor(turbulent_samples, dtype=torch.float32))['regime'].numpy()
    
    # Statistics
    stable_mean = stable_regime.mean() * 100
    stable_std = stable_regime.std() * 100
    turbulent_mean = turbulent_regime.mean() * 100
    turbulent_std = turbulent_regime.std() * 100
    
    print("📊 REGIME PREDICTION RESULTS:")
    print("-" * 70)
    print(f"{'Regime':<15} | {'Predicted Chaos%':<18} | {'Std Dev':<10}")
    print("-" * 70)
    print(f"{'Stable':<15} | {stable_mean:>18.1f}% | {stable_std:>10.2f}%")
    print(f"{'Turbulent':<15} | {turbulent_mean:>18.1f}% | {turbulent_std:>10.2f}%")
    print("-" * 70)
    
    # Separation quality
    separation = abs(turbulent_mean - stable_mean)
    
    print(f"\n🎯 SEPARATION ANALYSIS:")
    print(f"   Mean difference: {separation:.1f}%")
    
    if separation < 20:
        print(f"   ❌ POOR SEPARATION - Model cannot distinguish regimes")
        print(f"   💡 Recommendation: Adjust entropy regularization")
    elif separation < 50:
        print(f"   ⚠️  MODERATE SEPARATION - Model partially distinguishes regimes")
    else:
        print(f"   ✅ GOOD SEPARATION - Model effectively distinguishes regimes")
    
    print()
    print(f"💡 INTERPRETATION:")
    if stable_mean > 50 and turbulent_mean > 50:
        print(f"   Model predicts CHAOS for both regimes")
        print(f"   → Entropy regularization too weak or regime gate biased")
    elif stable_mean < 30 and turbulent_mean < 30:
        print(f"   Model predicts STABLE for both regimes")
        print(f"   → Entropy regularization too strong or regime gate suppressed")
    else:
        print(f"   Model shows regime sensitivity - GOOD!")
    
    print()

def test_adaptive_strategy():
    """
    Test adaptive strategy with different regimes.
    """
    print("="*70)
    print("  ADAPTIVE STRATEGY TEST")
    print("="*70)
    print()
    
    model = SyntropyQuantKernelV5(input_dim=7, hidden_dim=128, latent_dim=16)
    try:
        model.load_state_dict(torch.load("models/syntropy_v5.pt", map_location='cpu'))
    except:
        return
    model.eval()
    
    # Simulate 2 different market types
    print("Simulating 2 market types for 100 days each:")
    print()
    
    # Market 1: Stable (Symplectic)
    prices_stable = []
    returns_stable = []
    regime_stable = []
    
    price = 100.0
    for day in range(100):
        # Mean-reverting market
        pull = -0.5 * (price - 100) / 100
        ret = 0.0001 + pull + np.random.normal(0, 0.015)
        price = price * (1 + ret)
        prices_stable.append(price)
        returns_stable.append(ret)
        
        # Feature (simplified)
        feat = np.array([
            ret,
            0.015,
            0.015,
            np.random.normal(0, 0.2),
            0.4,  # Hurst < 0.5
            (price - 100) / 100,
            np.sin(day * 0.1)
        ])
        
        with torch.no_grad():
            out = model(torch.tensor(feat, dtype=torch.float32).unsqueeze(0))
            regime = out['regime'].item()
            regime_stable.append(regime)
    
    # Market 2: Turbulent (Gauge)
    prices_turb = []
    returns_turb = []
    regime_turb = []
    
    price = 100.0
    for day in range(100):
        # Trending market
        ret = 0.002 + np.random.normal(0, 0.03)
        price = price * (1 + ret)
        prices_turb.append(price)
        returns_turb.append(ret)
        
        # Feature (simplified)
        feat = np.array([
            ret,
            0.03,
            0.04,
            np.random.normal(0.5, 0.8),
            0.7,  # Hurst > 0.5
            (price - 100) / 100,
            np.sin(day * 0.1)
        ])
        
        with torch.no_grad():
            out = model(torch.tensor(feat, dtype=torch.float32).unsqueeze(0))
            regime = out['regime'].item()
            regime_turb.append(regime)
    
    # Results
    print(f"{'Market':<15} | {'Final Price':<12} | {'Total Ret':<10} | {'Avg Chaos%':<12}")
    print("-" * 70)
    
    stable_chaos = np.mean(regime_stable) * 100
    turb_chaos = np.mean(regime_turb) * 100
    
    print(f"{'Stable':<15} | {prices_stable[-1]:>10.2f} | {sum(returns_stable):>10.2%} | {stable_chaos:>12.1f}%")
    print(f"{'Turbulent':<15} | {prices_turb[-1]:>10.2f} | {sum(returns_turb):>10.2%} | {turb_chaos:>12.1f}%")
    print("-" * 70)
    print()
    
    if stable_chaos < turb_chaos - 30:
        print(f"✅ Model correctly identifies turbulent market as more chaotic")
    else:
        print(f"⚠️  Model may not be distinguishing regimes effectively")

if __name__ == "__main__":
    verify_regime_separation()
    test_adaptive_strategy()
