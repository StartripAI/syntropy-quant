import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from src.core.hybrid.kernel_v5 import SyntropyQuantKernelV5
from src.data.features_v5 import FeatureBuilderV5

def physics_informed_loss(output, target, criterion):
    """
    CORRECTED Physics-informed loss function.
    Includes:
    1. Classification loss
    2. Entropy regularization (force clear regime choice)
    3. Berry phase minimization (least action)
    """
    # 1. Profit Objective
    cls_loss = criterion(output['logits'], target)
    
    # 2. Entropy Regularization (The Poincaré Constraint)
    regime = output['regime']
    # We want regime to be CLEAR (near 0 or near 1), NOT 0.5
    # Entropy = -sum(p * log(p)) is maximized when distribution is uniform
    # We want to MINIMIZE entropy to force clear decisions
    # For binary distribution: H = -[p*log(p) + (1-p)*log(1-p)]
    # When p=0 or p=1, H=0 (minimum)
    # When p=0.5, H=log(2) (maximum)
    
    # Simple approach: penalize being near 0.5
    # Distance from 0.5 squared: minimized when regime=0 or regime=1
    entropy_penalty = torch.mean((regime - 0.5)**2)
    
    # 3. Berry Phase Minimization (Least Action)
    curvature_loss = torch.mean(output['curvature'])
    
    # Total loss: classify + entropy_penalty + curvature
    # entropy_penalty encourages clear regime choice (away from 0.5)
    # Use STRONGER entropy weight (2.0) to force clear decisions
    return cls_loss + 2.0 * entropy_penalty + 0.01 * curvature_loss

def generate_synthetic_market(
    days=1000,
    trend_strength=0.0005,
    volatility=0.02,
    regime_shift=0.3  # Probability of regime switch
):
    """
    Generate synthetic market data with two regimes.
    """
    np.random.seed(42)
    prices = []
    regime_history = []
    
    current_regime = 0  # 0 = Symplectic, 1 = Gauge
    current_price = 100.0
    
    for day in range(days):
        # Check for regime switch
        if np.random.random() < regime_shift:
            current_regime = 1 - current_regime
        
        regime_history.append(current_regime)
        
        # Generate returns based on regime
        if current_regime == 0:
            # Symplectic: Mean-reverting
            drift = 0.0001
            vol = volatility * 0.8
            pull = -0.5 * (current_price - 100) / 100
            ret = drift + pull + np.random.normal(0, vol)
        else:
            # Gauge: Trending
            drift = trend_strength
            vol = volatility * 1.5
            ret = drift + np.random.normal(0, vol)
        
        current_price = current_price * (1 + ret)
        prices.append(current_price)
    
    # Generate OHLCV
    dates = pd.date_range(start='2016-01-01', periods=days)
    df = pd.DataFrame({
        'Open': [p * (1 + np.random.uniform(-0.005, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.uniform(0, 0.015))) for p in prices],
        'Low': [p * (1 - abs(np.random.uniform(0, 0.015))) for p in prices],
        'Close': prices,
        'Volume': [1e6 * (1 + np.random.uniform(-0.3, 0.5)) for _ in range(days)]
    }, index=dates)
    
    return df, np.array(regime_history)

def train(args):
    print("🌌 SYNTROPY v5.0 | Training Hybrid Kernel (FIXED LOSS)")
    print("   Generating synthetic market with regime switches...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    builder = FeatureBuilderV5()
    
    # Generate synthetic data for multiple "symbols" with different regimes
    X_list, Y_list, RegimeLabels = [], [], []
    
    configs = [
        ("SYMBOL_A", 0.0008, 0.018, 0.2),  # Strong trend
        ("SYMBOL_B", 0.0002, 0.015, 0.1),  # Weak trend
        ("SYMBOL_C", 0.0000, 0.012, 0.05),  # Range-bound
    ]
    
    for name, trend, vol, shift in configs:
        df, regimes = generate_synthetic_market(days=1500, trend_strength=trend, volatility=vol, regime_shift=shift)
        feat = builder.build(df)
        
        if len(feat) == 0:
            continue
        
        # Labels: 3-Day Future Return
        closes = df['Close'].values[50:]
        ret = (closes[3:] - closes[:-3]) / closes[:-3]
        
        # Align
        min_len = min(len(feat), len(ret), len(regimes[50:]))
        feat = feat[:min_len]
        ret = ret[:min_len]
        reg = regimes[50:50+min_len]
        
        y = np.ones(len(ret))
        y[ret > 0.003] = 2  # Long
        y[ret < -0.003] = 0  # Short
        
        X_list.append(feat)
        Y_list.append(torch.tensor(y, dtype=torch.long))
        RegimeLabels.append(reg)
        
        print(f"   Generated {name}: {len(feat)} samples, Gauge regime: {np.mean(reg)*100:.1f}%")
    
    if not X_list:
        print("❌ No data generated")
        return
    
    X = torch.cat(X_list).to(device)
    Y = torch.cat(Y_list).to(device)
    
    print(f"\n   Total samples: {len(X)}")
    print(f"   Feature dim: {X.shape[1]}")
    print(f"   Device: {device}")
    
    # Create model
    model = SyntropyQuantKernelV5(input_dim=7, hidden_dim=128, latent_dim=16).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {trainable_params:,} / {total_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    model.train()
    print("\n   Training with physics-informed loss...")
    print("   " + "="*60)
    
    for epoch in range(args.epochs):
        perm = torch.randperm(len(X))
        total_loss = 0
        avg_regime = 0
        avg_curvature = 0
        
        for i in range(0, len(X), 2048):
            idx = perm[i:i+2048]
            bx, by = X[idx], Y[idx]
            
            optimizer.zero_grad()
            out = model(bx)
            loss = physics_informed_loss(out, by, criterion)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            avg_regime += out['regime'].mean().item()
            avg_curvature += out['curvature'].mean().item()
        
        num_batches = len(X) / 2048
        if (epoch+1) % 5 == 0:
            reg_pct = (avg_regime / num_batches) * 100
            print(f"   Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | Chaos: {reg_pct:5.1f}% | Curvature: {avg_curvature/num_batches:.4f}")
    
    # Save model
    torch.save(model.state_dict(), args.save)
    print("\n   " + "="*60)
    print(f"✅ Model saved: {args.save}")
    
    # Quick validation
    print("\n   Quick validation...")
    model.eval()
    with torch.no_grad():
        test_sample = X[:100]
        out = model(test_sample)
        
        avg_regime = out['regime'].mean().item() * 100
        avg_curvature = out['curvature'].mean().item()
        
        print(f"   Validation regime: {avg_regime:.1f}% chaos")
        print(f"   Validation curvature: {avg_curvature:.4f}")
        
        # Check regime distribution
        reg_distribution = out['regime'].numpy().flatten()
        print(f"   Regime < 0.3: {np.sum(reg_distribution < 0.3)} samples")
        print(f"   Regime > 0.7: {np.sum(reg_distribution > 0.7)} samples")
        print(f"   Regime 0.3-0.7: {np.sum((reg_distribution >= 0.3) & (reg_distribution <= 0.7))} samples")
        
        # Calculate actual entropy
        reg_mean = np.mean(reg_distribution)
        if 0 < reg_mean < 1:
            entropy = -reg_mean * np.log(reg_mean) - (1-reg_mean) * np.log(1-reg_mean)
            print(f"   Binary Entropy: {entropy:.4f} (0=min, 0.693=max)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Syntropy Quant v5.0 with CORRECTED Loss')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--save', type=str, default='models/syntropy_v5.pt')
    args = parser.parse_args()
    
    train(args)

