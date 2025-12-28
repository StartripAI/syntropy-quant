#!/usr/bin/env python3
"""
Syntropy Quant v4.0 vs v5.0 Comparison
Compare:
- v4.0: Syplectic Geometry (Energy-conserving)
- v5.0: Holographic Topological Field (Regime-adaptive)
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import argparse

# Import both versions
from src.core.physics import SyntropyQuantKernel  # v4.0 if exists
from src.core.kernel import SyntropyQuantKernel as SyntropyQuantKernelAlt
from src.core.hybrid.kernel_v5 import SyntropyQuantKernelV5  # v5.0
from src.data.features import FeatureBuilder
from src.data.features_v5 import FeatureBuilderV5

def generate_market_data(n_samples=2000):
    """
    Generate synthetic market with two regimes:
    1. Symplectic (stable, mean-reverting)
    2. Gauge (turbulent, trending)
    """
    np.random.seed(42)
    
    # Generate three types of markets
    markets = {
        "STABLE_INDEX": {"trend": 0.0000, "vol": 0.012, "shift": 0.05},
        "TRENDING_TECH": {"trend": 0.0015, "vol": 0.025, "shift": 0.35},
        "BUBBLE_CRYPTO": {"trend": 0.0030, "vol": 0.040, "shift": 0.50},
    }
    
    results = {}
    
    for name, params in markets.items():
        prices = []
        regimes = []
        current_regime = 0
        current_price = 100.0
        
        for day in range(n_samples):
            # Regime switch
            if np.random.random() < params["shift"]:
                current_regime = 1 - current_regime
            
            regimes.append(current_regime)
            
            # Generate returns
            if current_regime == 0:
                # Symplectic: Mean-reverting
                drift = 0.0001
                vol = params["vol"] * 0.7
                pull = -0.6 * (current_price - 100) / 100
                ret = drift + pull + np.random.normal(0, vol)
            else:
                # Gauge: Trending
                drift = params["trend"]
                vol = params["vol"] * 1.5
                ret = drift + np.random.normal(0, vol)
            
            current_price = current_price * (1 + ret)
            prices.append(current_price)
        
        # Build DataFrame
        dates = pd.date_range(start="2020-01-01", periods=n_samples)
        df = pd.DataFrame({
            "Open": [p * (1 + np.random.uniform(-0.005, 0.005)) for p in prices],
            "High": [p * (1 + abs(np.random.uniform(0, 0.015))) for p in prices],
            "Low": [p * (1 - abs(np.random.uniform(0, 0.015))) for p in prices],
            "Close": prices,
            "Volume": [1e6 * (1 + np.random.uniform(-0.3, 0.5)) for _ in range(n_samples)]
        }, index=dates)
        
        results[name] = {"df": df, "regimes": np.array(regimes), "params": params}
    
    return results

def backtest_v4(model, features, returns, market_type="") -> Dict:
    """
    Backtest using v4.0 (Symplectic Physics).
    Fixed threshold strategy.
    """
    with torch.no_grad():
        logits, gamma = model(features, dt=1.0)
        probs = torch.softmax(logits, dim=1).numpy()
    
    # Strategy: Long - Short prob
    signal = probs[:, 2] - probs[:, 0]
    signal = signal[:-1]
    
    # Fixed threshold
    thresh = 0.15
    pos = np.zeros_like(signal)
    pos[signal > thresh] = 1.0
    pos[signal < -thresh] = -0.5
    
    strat_ret = pos * returns[:len(pos)]
    
    cum = np.prod(1 + strat_ret) - 1
    ann = np.mean(strat_ret) * 252
    vol = np.std(strat_ret) * np.sqrt(252) + 1e-6
    sharpe = ann / vol
    
    eq = np.cumprod(1 + strat_ret)
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    mdd = np.min(dd)
    
    return {
        "return": cum,
        "sharpe": sharpe,
        "max_dd": mdd,
        "volatility": vol,
        "market_type": market_type
    }

def backtest_v5(model, features, returns, market_type="") -> Dict:
    """
    Backtest using v5.0 (Holographic Field).
    Adaptive regime-based strategy.
    """
    with torch.no_grad():
        out = model(features)
        logits = out["logits"].numpy()
        regimes = out["regime"].numpy().flatten()
    
    probs = softmax(logits, axis=1)
    sig = probs[:, 2] - probs[:, 0]
    sig = sig[:-1]
    
    # Adaptive strategy based on regime
    pos = np.zeros_like(sig)
    
    for i in range(len(sig)):
        r = regimes[i]
        s = sig[i]
        
        # Adaptive threshold
        thresh = 0.20 * (1.0 - r * 0.6)
        
        if s > thresh:
            pos[i] = 1.0
        elif s < -thresh:
            pos[i] = -1.0
    
    strat_ret = pos * returns[:len(pos)]
    
    cum = np.prod(1 + strat_ret) - 1
    ann = np.mean(strat_ret) * 252
    vol = np.std(strat_ret) * np.sqrt(252) + 1e-6
    sharpe = ann / vol
    
    eq = np.cumprod(1 + strat_ret)
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    mdd = np.min(dd)
    
    return {
        "return": cum,
        "sharpe": sharpe,
        "max_dd": mdd,
        "volatility": vol,
        "market_type": market_type,
        "avg_regime": np.mean(regimes) * 100
    }

def softmax(x, axis=1):
    """Softmax for numpy arrays."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def compare_versions():
    """
    Run comprehensive comparison between v4.0 and v5.0.
    """
    print("\n" + "="*80)
    print("  SYNTROPY QUANT v4.0 vs v5.0: THE GRAND COMPARISON")
    print("="*80)
    print()
    print("Theory:")
    print("  v4.0: Syplectic Geometry (Energy-conserving, Liouville's Theorem)")
    print("  v5.0: Holographic Topological Field (Regime-adaptive, Berry Phase)")
    print()
    
    # Generate market data
    print("Generating synthetic market data...")
    markets = generate_market_data(n_samples=2000)
    
    # Initialize feature builders
    builder_v4 = FeatureBuilder()
    builder_v5 = FeatureBuilderV5()
    
    # Build features for each market
    market_features = {}
    for name, data in markets.items():
        df = data["df"]
        feat_v4 = builder_v4.build(df)
        feat_v5 = builder_v5.build(df)
        
        returns = (df["Close"].values[1:] - df["Close"].values[:-1]) / df["Close"].values[:-1]
        
        if len(feat_v4) > 50:
            market_features[name] = {
                "feat_v4": feat_v4[50:],
                "feat_v5": feat_v5[50:],
                "returns": returns[50:],
                "true_regimes": data["regimes"][50:],
                "params": data["params"]
            }
    
    print(f"Processed {len(market_features)} markets\n")
    
    # Load models
    print("Loading models...")
    models = {}
    
    # Try v4.0
    try:
        model_v4 = SyntropyQuantKernel(input_dim=4, hidden_dim=64)
        # Try alternate imports
        try:
            model_v4_alt = SyntropyQuantKernelAlt(input_dim=4, hidden_dim=64)
            model_v4_alt.load_state_dict(torch.load("models/gauge_kernel.pt", map_location="cpu"))
            models["v4.0"] = model_v4_alt
            print("  ✅ v4.0 (GaugeKernel) loaded")
        except:
            try:
                model_v4.load_state_dict(torch.load("models/gauge_kernel.pt", map_location="cpu"))
                models["v4.0"] = model_v4
                print("  ✅ v4.0 (GaugeKernel) loaded")
            except Exception as e:
                print(f"  ⚠️  v4.0 not found: {e}")
    except Exception as e:
        print(f"  ⚠️  v4.0 not available: {e}")
    
    # Load v5.0
    try:
        model_v5 = SyntropyQuantKernelV5(input_dim=7, hidden_dim=128, latent_dim=16)
        model_v5.load_state_dict(torch.load("models/syntropy_v5_pure.pt", map_location="cpu"))
        models["v5.0"] = model_v5
        print("  ✅ v5.0 (Holographic Field) loaded")
    except Exception as e:
        print(f"  ❌ v5.0 not found: {e}")
        return
    
    print()
    
    # Run backtests
    results = {"v4.0": [], "v5.0": []}
    
    for name, data in market_features.items():
        params = data["params"]
        true_regimes = data["true_regimes"]
        
        print(f"\n{'='*60}")
        print(f"Market: {name}")
        print(f"  True Regime: {np.mean(true_regimes)*100:.1f}% (Gauge-dominant)")
        print(f"  Trend: {params['trend']*100:.2f}%/day, Vol: {params['vol']*100:.2f}%")
        print("-" * 60)
        
        # v4.0 backtest (if available)
        if "v4.0" in models:
            try:
                res_v4 = backtest_v4(
                    models["v4.0"],
                    data["feat_v4"][:, :4],  # Use first 4 features
                    data["returns"],
                    market_type=name
                )
                results["v4.0"].append(res_v4)
                print(f"  v4.0 (Symplectic):  {res_v4['return']*100:>7.1f}% | Sharpe: {res_v4['sharpe']:>5.2f} | DD: {res_v4['max_dd']*100:>5.1f}%")
            except Exception as e:
                print(f"  v4.0 error: {e}")
        
        # v5.0 backtest
        try:
            res_v5 = backtest_v5(
                models["v5.0"],
                data["feat_v5"],
                data["returns"],
                market_type=name
            )
            results["v5.0"].append(res_v5)
            print(f"  v5.0 (Holographic): {res_v5['return']*100:>7.1f}% | Sharpe: {res_v5['sharpe']:>5.2f} | DD: {res_v5['max_dd']*100:>5.1f}% | Regime: {res_v5['avg_regime']:>5.1f}%")
        except Exception as e:
            print(f"  v5.0 error: {e}")
    
    # Summary table
    print("\n" + "="*80)
    print("  SUMMARY COMPARISON")
    print("="*80)
    print()
    
    if "v4.0" in models and results["v4.0"]:
        v4_returns = [r["return"] for r in results["v4.0"]]
        v4_returns_pct = [r * 100 for r in v4_returns]
        
        print("v4.0 (Symplectic Physics - Energy Conserving):")
        print(f"  Mean Return:   {np.mean(v4_returns_pct):>7.1f}%")
        print(f"  Median Return: {np.median(v4_returns_pct):>7.1f}%")
        print(f"  Best Return:   {np.max(v4_returns_pct):>7.1f}%")
        print(f"  Worst Return:  {np.min(v4_returns_pct):>7.1f}%")
        print(f"  Mean Sharpe:   {np.mean([r['sharpe'] for r in results['v4.0']]):>6.2f}")
        print(f"  Mean MaxDD:    {np.mean([r['max_dd'] for r in results['v4.0']])*100:>7.1f}%")
        print()
    
    if "v5.0" in models and results["v5.0"]:
        v5_returns = [r["return"] for r in results["v5.0"]]
        v5_returns_pct = [r * 100 for r in v5_returns]
        v5_regimes = [r["avg_regime"] for r in results["v5.0"]]
        
        print("v5.0 (Holographic Field - Regime Adaptive):")
        print(f"  Mean Return:   {np.mean(v5_returns_pct):>7.1f}%")
        print(f"  Median Return: {np.median(v5_returns_pct):>7.1f}%")
        print(f"  Best Return:   {np.max(v5_returns_pct):>7.1f}%")
        print(f"  Worst Return:  {np.min(v5_returns_pct):>7.1f}%")
        print(f"  Mean Sharpe:   {np.mean([r['sharpe'] for r in results['v5.0']]):>6.2f}")
        print(f"  Mean MaxDD:    {np.mean([r['max_dd'] for r in results['v5.0']])*100:>7.1f}%")
        print(f"  Avg Regime:    {np.mean(v5_regimes):>7.1f}% (Chaos)")
        print()
    
    # Head-to-head comparison
    if "v4.0" in models and "v5.0" in models:
        print("="*80)
        print("  HEAD-TO-HEAD: MARKET BY MARKET")
        print("="*80)
        print()
        print(f"{'Market':<20} | {'v4.0 Return':<15} | {'v5.0 Return':<15} | {'v4.0 Sharpe':<14} | {'v5.0 Sharpe':<14} | {'v4.0 DD':<12} | {'v5.0 DD':<12} | {'v5.0 Regime':<15}")
        print("-" * 110)
        
        for i, name in enumerate(market_features.keys()):
            v4 = results["v4.0"][i] if i < len(results["v4.0"]) else None
            v5 = results["v5.0"][i] if i < len(results["v5.0"]) else None
            
            v4_ret = f"{v4['return']*100:>7.1f}%" if v4 else "N/A"
            v5_ret = f"{v5['return']*100:>7.1f}%" if v5 else "N/A"
            v4_sharpe = f"{v4['sharpe']:>5.2f}" if v4 else "N/A"
            v5_sharpe = f"{v5['sharpe']:>5.2f}" if v5 else "N/A"
            v4_dd = f"{v4['max_dd']*100:>5.1f}%" if v4 else "N/A"
            v5_dd = f"{v5['max_dd']*100:>5.1f}%" if v5 else "N/A"
            v5_regime = f"{v5['avg_regime']:>5.1f}%" if v5 else "N/A"
            
            print(f"{name:<20} | {v4_ret:<15} | {v5_ret:<15} | {v4_sharpe:<14} | {v5_sharpe:<14} | {v4_dd:<12} | {v5_dd:<12} | {v5_regime:<15}")
    
    # Final conclusion
    print("\n" + "="*80)
    print("  PHYSICS INTERPRETATION")
    print("="*80)
    print()
    
    if "v4.0" in models and "v5.0" in models:
        v4_mean = np.mean([r["return"] for r in results["v4.0"]]) * 100
        v5_mean = np.mean([r["return"] for r in results["v5.0"]]) * 100
        v4_sharpe = np.mean([r["sharpe"] for r in results["v4.0"]])
        v5_sharpe = np.mean([r["sharpe"] for r in results["v5.0"]])
        
        print(f"v4.0 (Symplectic): Energy conservation, Liouville volume preservation")
        print(f"  → Best for: Mean-reverting, stable markets")
        print(f"  → Performance: {v4_mean:>7.1f}% avg return, {v4_sharpe:>5.2f} avg Sharpe")
        print()
        print(f"v5.0 (Holographic): Regime switching, Berry phase, topological invariants")
        print(f"  → Best for: Turbulent, trending markets")
        print(f"  → Performance: {v5_mean:>7.1f}% avg return, {v5_sharpe:>5.2f} avg Sharpe")
        print()
        
        if v5_mean > v4_mean + 5:
            print("✅ v5.0 OUTPERFORMS v4.0 (Regime adaptation works!)")
        elif v4_mean > v5_mean + 5:
            print("✅ v4.0 OUTPERFORMS v5.0 (Symplectic stability wins!)")
        else:
            print("⚠️  Performance is similar - both have merits")
    
    print()

def main():
    compare_versions()

if __name__ == "__main__":
    main()
