import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from src.core.hybrid.kernel_v5 import SyntropyQuantKernelV5
from src.data.features_v5 import FeatureBuilderV5

def pure_profit_loss(output, target, criterion):
    """
    PURE profit-focused loss - NO entropy regularization.
    Let the regime gate learn naturally from market state.
    """
    # Only classification loss for profit prediction
    cls_loss = criterion(output['logits'], target)
    
    # Small curvature penalty (least action), but NO regime constraint
    curvature_loss = torch.mean(output['curvature'])
    
    return cls_loss + 0.005 * curvature_loss

def generate_synthetic_market(days=1000, trend_strength=0.0005, volatility=0.02, regime_shift=0.3):
    np.random.seed(42)
    prices = []
    regime_history = []
    
    current_regime = 0
    current_price = 100.0
    
    for day in range(days):
        if np.random.random() < regime_shift:
            current_regime = 1 - current_regime
        
        regime_history.append(current_regime)
        
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
    print("🌌 SYNTROPY v5.0 | PURE PROFIT TRAINING (No Regime Constraint)")
    print("   Letting regime gate learn naturally from market dynamics...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    builder = FeatureBuilderV5()
    
    # Generate synthetic data
    X_list, Y_list = [], []
    
    configs = [
        ("SYMBOL_A", 0.0008, 0.018, 0.2),
        ("SYMBOL_B", 0.0002, 0.015, 0.1),
        ("SYMBOL_C", 0.0000, 0.012, 0.05),
    ]
    
    for name, trend, vol, shift in configs:
        df, _ = generate_synthetic_market(days=1500, trend_strength=trend, volatility=vol, regime_shift=shift)
        feat = builder.build(df)
        
        if len(feat) == 0:
            continue
        
        closes = df['Close'].values[50:]
        ret = (closes[3:] - closes[:-3]) / closes[:-3]
        
        min_len = min(len(feat), len(ret))
        feat = feat[:min_len]
        ret = ret[:min_len]
        
        y = np.ones(len(ret))
        y[ret > 0.003] = 2
        y[ret < -0.003] = 0
        
        X_list.append(feat)
        Y_list.append(torch.tensor(y, dtype=torch.long))
    
    if not X_list:
        print("❌ No data generated")
        return
    
    X = torch.cat(X_list).to(device)
    Y = torch.cat(Y_list).to(device)
    
    print(f"   Total samples: {len(X)}")
    print(f"   Training with PURE profit loss (no regime supervision)...")
    print()
    
    model = SyntropyQuantKernelV5(input_dim=7, hidden_dim=128, latent_dim=16).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {trainable_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    model.train()
    print("=" * 60)
    
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
            loss = pure_profit_loss(out, by, criterion)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            avg_regime += out['regime'].mean().item()
            avg_curvature += out['curvature'].mean().item()
        
        num_batches = len(X) / 2048
        if (epoch+1) % 5 == 0:
            reg_pct = (avg_regime / num_batches) * 100
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | Chaos: {reg_pct:5.1f}% | Curvature: {avg_curvature/num_batches:.4f}")
    
    torch.save(model.state_dict(), args.save)
    print("\n" + "=" * 60)
    print(f"✅ Model saved: {args.save}")
    print()
    
    # Validation
    model.eval()
    print("Quick validation:")
    with torch.no_grad():
        test = X[:200]
        out = model(test)
        
        reg_dist = out['regime'].numpy().flatten()
        reg_mean = np.mean(reg_dist) * 100
        reg_std = np.std(reg_dist) * 100
        
        print(f"   Regime mean: {reg_mean:.1f}% (std: {reg_std:.1f}%)")
        print(f"   Curvature: {out['curvature'].mean().item():.4f}")
        print(f"   Regime < 0.3: {np.sum(reg_dist < 0.3)} samples")
        print(f"   Regime > 0.7: {np.sum(reg_dist > 0.7)} samples")
        print(f"   Regime 0.3-0.7: {np.sum((reg_dist >= 0.3) & (reg_dist <= 0.7))} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--save', type=str, default='models/syntropy_v5_pure.pt')
    args = parser.parse_args()
    
    train(args)

