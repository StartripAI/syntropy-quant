#!/bin/bash

# ====================================================
# SYNTROPY QUANT v5.0: THE POINCARE SUPREMACY
# Architecture: Symplectic-Gauge Hybrid Kernel
# Math: Fiber Bundles, Berry Phase, Hurst Fractal
# ====================================================

PROJECT_DIR="/Users/alfred/syntropy-quant"
echo ">>> [POINCARE] Initiating v5.0 Upgrade Sequence..."
echo ">>> [CLEANUP] Purging inferior mathematics..."

# 1. 重构目录结构 (Sacred Geometry)
rm -rf $PROJECT_DIR/src $PROJECT_DIR/models $PROJECT_DIR/data_cache $PROJECT_DIR/output
mkdir -p $PROJECT_DIR/src/core/{physics,geometry,hybrid}
mkdir -p $PROJECT_DIR/src/data
mkdir -p $PROJECT_DIR/models
mkdir -p $PROJECT_DIR/output

cd $PROJECT_DIR

# ----------------------------------------------------
# 2. 几何层: 贝里相位 (Topology)
# ----------------------------------------------------
cat << 'EOF' > src/core/geometry/berry_phase.py
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
EOF

# ----------------------------------------------------
# 3. 物理层: 辛动力学 (Symplectic - Ground State)
# ----------------------------------------------------
cat << 'EOF' > src/core/physics/symplectic.py
import torch
import torch.nn as nn

class SymplecticEuler(nn.Module):
    """
    Hamiltonian Dynamics for 'Laminar Flow' (Stable Markets).
    Strictly preserves Phase Space Volume (Liouville's Theorem).
    Best for: KO, JNJ, Index Range.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Separable Hamiltonian H(q,p) = T(p) + V(q)
        # Potential Force Field (-dV/dq)
        self.V_net = nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, dim))
        # Kinetic Velocity Field (dT/dp)
        self.T_net = nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, dim))

    def forward(self, q, p, dt=1.0):
        # Symplectic Euler Integration (Energy Stable)
        # 1. Kick: p(t+1) = p(t) - grad(V)(q(t)) * dt
        force = -self.V_net(q)
        p_next = p + force * dt
        
        # 2. Drift: q(t+1) = q(t) + grad(T)(p(t+1)) * dt
        velocity = self.T_net(p_next)
        q_next = q + velocity * dt
        
        return q_next, p_next
EOF

# ----------------------------------------------------
# 4. 物理层: 规范场 (Gauge Field - Excited State)
# ----------------------------------------------------
cat << 'EOF' > src/core/physics/gauge_field.py
import torch
import torch.nn as nn

class YangMillsField(nn.Module):
    """
    Non-Abelian Gauge Field for 'Turbulent Flow' (Bubbles/Crashes).
    Models market forces as fiber bundle curvature (Trend).
    Best for: NVDA, TSLA, Crypto.
    """
    def __init__(self, dim):
        super().__init__()
        # Covariant Derivative Operator D_u
        self.gauge_transform = nn.Sequential(
            nn.Linear(dim, 64), nn.Tanh(),
            nn.Linear(64, dim)
        )
        
    def forward(self, state, dt=1.0):
        # Evolution driven by Field Strength
        # Non-conservative dynamics (Energy Injection)
        drift = self.gauge_transform(state)
        
        # Stochastic Differential Equation (Langevin)
        diffusion = torch.randn_like(state) * 0.05
        
        state_next = state + drift * dt + diffusion
        return state_next
EOF

# ----------------------------------------------------
# 5. 混合内核 v5.0 (The Holonomic Brain)
# ----------------------------------------------------
cat << 'EOF' > src/core/hybrid/kernel_v5.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..physics.symplectic import SymplecticEuler
from ..physics.gauge_field import YangMillsField
from ..geometry.berry_phase import BerryCurvature

class SyntropyQuantKernelV5(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, latent_dim=16):
        super().__init__()
        
        # 1. Manifold Embedding (Features -> Phase Space)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2) # Split into [q, p]
        )
        
        # 2. Dual Physics Engines
        self.symplectic = SymplecticEuler(latent_dim)   # Conservation
        self.gauge_field = YangMillsField(latent_dim * 2) # Expansion
        
        # 3. Geometry Observer
        self.berry = BerryCurvature(latent_dim * 2)
        
        # 4. Regime Gate (The Order Parameter)
        # Determines if we are in Solid (Symplectic) or Plasma (Gauge) state
        self.regime_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid() # 0 = Stable, 1 = Chaos
        )
        
        # 5. Policy Head
        self.policy = nn.Linear(latent_dim * 2, 3) # Long/Neutral/Short

    def forward(self, x, dt=1.0):
        # Embed
        features = self.encoder(x)
        half = features.shape[1] // 2
        q, p = features[:, :half], features[:, half:]
        state = torch.cat([q, p], dim=-1)
        
        # Calculate Regime Alpha (Phase Transition)
        alpha = self.regime_gate(state)
        
        # Path A: Symplectic Evolution
        q_sym, p_sym = self.symplectic(q, p, dt)
        state_sym = torch.cat([q_sym, p_sym], dim=-1)
        
        # Path B: Gauge Field Evolution
        state_gauge = self.gauge_field(state, dt)
        
        # Superposition (Quantum-Classical Mix)
        state_next = (1 - alpha) * state_sym + alpha * state_gauge
        
        # Geometric Check (Berry Phase)
        curvature = self.berry(state, state_next)
        
        # Decision
        logits = self.policy(state_next)
        
        return {
            "logits": logits,
            "regime": alpha,      # 0..1
            "curvature": curvature,
            "state": state_next
        }
EOF

# ----------------------------------------------------
# 6. 特征工程 v5 (Fractal & Robust)
# ----------------------------------------------------
cat << 'EOF' > src/data/features_v5.py
import numpy as np
import pandas as pd
import torch

class FeatureBuilderV5:
    """
    v5.0 Features: Includes Hurst Exponent & Topological Invariants.
    Epsilon-stabilized to prevent Singularity (NaNs).
    """
    def build(self, df: pd.DataFrame) -> torch.Tensor:
        if len(df) < 60: return torch.empty(0)
        
        df = df.copy()
        eps = 1e-8
        
        # 1. Price Dynamics (Log Space)
        close = df['Close'].values
        ret = np.diff(np.log(close + eps), prepend=close[0])
        
        # 2. Energy (Volatility Surface)
        vol_5 = pd.Series(ret).rolling(5).std().fillna(0).values
        vol_20 = pd.Series(ret).rolling(20).std().fillna(0).values
        
        # 3. Order Flow (Momentum)
        hl_range = np.maximum(df['High'].values - df['Low'].values, eps)
        # Close Location Value
        clv = ((close - df['Low'].values) - (df['High'].values - close)) / hl_range
        # Volume Force
        flow = clv * np.log(df['Volume'].values + 1.0)
        # Z-Score
        flow_z = (flow - pd.Series(flow).rolling(20).mean()) / (pd.Series(flow).rolling(20).std() + eps)
        flow_z = flow_z.fillna(0).values
        
        # 4. Hurst Exponent (Fractal Dimension)
        # H > 0.5 (Trend), H < 0.5 (Mean Reversion)
        def rolling_hurst(series, window=30):
            h_vals = np.zeros(len(series)) + 0.5
            for i in range(window, len(series)):
                chunk = series[i-window:i]
                r = np.max(chunk) - np.min(chunk) + eps
                s = np.std(chunk) + eps
                # Simplified R/S analysis
                h_vals[i] = np.log(r/s) / np.log(window)
            return np.clip(h_vals, 0, 1)
            
        hurst = rolling_hurst(close)
        
        # 5. Relative Potential (Mean Reversion)
        ma50 = pd.Series(close).rolling(50).mean().fillna(close[0]).values
        pot = (close - ma50) / (ma50 + eps)
        
        # 6. Time Phase (Cyclicality)
        t = np.linspace(0, 4*np.pi, len(df))
        phase = np.sin(t)

        # Stack Features [Ret, Vol5, Vol20, Flow, Hurst, Potential, Phase]
        # Dim = 7
        feat = np.stack([ret, vol_5, vol_20, flow_z, hurst, pot, phase], axis=1)
        
        # Poincaré Cutoff (Remove Outliers)
        feat = np.clip(feat, -5.0, 5.0)
        
        return torch.tensor(feat[50:], dtype=torch.float32)
EOF

# ----------------------------------------------------
# 7. 数据获取 (Stable Fetcher)
# ----------------------------------------------------
cat << 'EOF' > src/data/fetcher.py
import yfinance as yf
import pandas as pd
import time

class DataFetcher:
    def fetch(self, symbol, start, end):
        print(f"Downloading {symbol}...", end=" ", flush=True)
        for _ in range(3):
            try:
                # Yahoo Finance is the only robust free source
                df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
                if len(df) > 100:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(method='ffill')
                    print("OK")
                    return df
            except:
                time.sleep(1)
        print("Failed")
        return pd.DataFrame()
EOF

# ----------------------------------------------------
# 8. 训练脚本 (Entropy Regularization)
# ----------------------------------------------------
cat << 'EOF' > train_v5.py
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from src.core.hybrid.kernel_v5 import SyntropyQuantKernelV5
from src.data.features_v5 import FeatureBuilderV5
from src.data.fetcher import DataFetcher

def physics_informed_loss(output, target, criterion):
    # 1. Profit Objective
    cls_loss = criterion(output['logits'], target)
    
    # 2. Entropy Regularization (The Poincaré Constraint)
    # Force the system to CHOOSE a regime (0 or 1).
    # We penalize being in the middle (0.5).
    regime = output['regime']
    # Maximize distance from 0.5 -> Minimize -(r-0.5)^2
    entropy_loss = -torch.mean((regime - 0.5)**2)
    
    # 3. Berry Phase Minimization (Least Action)
    # Penalize unnecessary geometric twisting
    curvature_loss = torch.mean(output['curvature'])
    
    return cls_loss + 0.1 * entropy_loss + 0.01 * curvature_loss

def train(args):
    print("🌌 SYNTROPY v5.0 | Training Hybrid Kernel...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fetcher = DataFetcher()
    builder = FeatureBuilderV5()
    symbols = args.symbols.split(',')
    
    X_list, Y_list = [], []
    
    for sym in symbols:
        df = fetcher.fetch(sym, "2016-01-01", "2023-01-01")
        if df.empty: continue
        feat = builder.build(df)
        if len(feat) == 0: continue
        
        # Labels: 3-Day Future Return
        closes = df['Close'].values[50:]
        ret = (closes[3:] - closes[:-3]) / closes[:-3]
        
        # Align
        min_len = min(len(feat), len(ret))
        feat = feat[:min_len]
        ret = ret[:min_len]
        
        y = np.ones(len(ret))
        y[ret > 0.003] = 2 # Long
        y[ret < -0.003] = 0 # Short
        
        X_list.append(feat)
        Y_list.append(torch.tensor(y, dtype=torch.long))
        
    if not X_list: return

    X = torch.cat(X_list).to(device)
    Y = torch.cat(Y_list).to(device)
    
    model = SyntropyQuantKernelV5(input_dim=7).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    model.train()
    print(f"   Training on {len(X)} phase space vectors...")
    
    for epoch in range(args.epochs):
        perm = torch.randperm(len(X))
        total_loss = 0
        avg_regime = 0
        
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
            
        if (epoch+1) % 5 == 0:
            reg_pct = (avg_regime / (len(X)/2048)) * 100
            print(f"   Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | Chaos Regime: {reg_pct:.1f}%")
            
    torch.save(model.state_dict(), args.save)
    print(f">>> Model Saved: {args.save}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save', type=str, default='models/syntropy_v5.pt')
    parser.add_argument('--symbols', type=str, default="SPY,QQQ,NVDA,AAPL")
    args = parser.parse_args()
    train(args)
EOF

# ----------------------------------------------------
# 9. 回测脚本 (Adaptive Physics)
# ----------------------------------------------------
cat << 'EOF' > run_backtest_v5.py
import torch
import pandas as pd
import numpy as np
from src.core.hybrid.kernel_v5 import SyntropyQuantKernelV5
from src.data.features_v5 import FeatureBuilderV5
from src.data.fetcher import DataFetcher

def run():
    print("\n===========================================================")
    print(" SYNTROPY QUANT v5.0 | HYBRID REGIME BACKTEST")
    print("===========================================================")
    
    model = SyntropyQuantKernelV5(input_dim=7)
    try:
        model.load_state_dict(torch.load("models/syntropy_v5.pt", map_location='cpu'))
    except:
        print("Model not found. Train first.")
        return
    model.eval()
    
    fetcher = DataFetcher()
    builder = FeatureBuilderV5()
    
    universe = {
        "INDEX": ["SPY", "QQQ", "IWM"],
        "TECH": ["NVDA", "MSFT", "META", "TSLA"],
        "PHARMA": ["LLY", "UNH"],
        "CONS": ["WMT", "KO", "MCD"]
    }
    
    print(f"{'Ticker':<8} | {'Ret%':<8} | {'Sharpe':<6} | {'MaxDD':<6} | {'Regime (Chaos%)':<15}")
    print("-" * 65)
    
    total_ret = []
    
    for cat, tickers in universe.items():
        for sym in tickers:
            df = fetcher.fetch(sym, "2023-01-01", "2025-12-31")
            if df.empty: continue
            feat = builder.build(df)
            if len(feat) == 0: continue
            
            closes = df['Close'].values[50:]
            returns = (closes[1:] - closes[:-1]) / closes[:-1]
            
            with torch.no_grad():
                out = model(feat)
                logits = out['logits']
                regimes = out['regime'].numpy().flatten()
            
            probs = torch.softmax(logits, dim=1).numpy()
            sig = probs[:, 2] - probs[:, 0]
            sig = sig[:-1]
            regimes = regimes[:-1]
            
            # Adaptive Strategy based on Physics Regime
            pos = np.zeros_like(sig)
            
            for i in range(len(sig)):
                r = regimes[i] # 0=Symplectic, 1=Gauge
                s = sig[i]
                
                # In Gauge Field (Chaos/Trend), we lower threshold to capture momentum
                # In Symplectic (Stable), we increase threshold to filter noise
                thresh = 0.20 * (1.0 - r * 0.6) 
                
                if s > thresh: pos[i] = 1.0
                elif s < -thresh: pos[i] = -1.0
            
            strat_ret = pos * returns[:len(pos)]
            cum = np.prod(1 + strat_ret) - 1
            ann = np.mean(strat_ret) * 252
            vol = np.std(strat_ret) * np.sqrt(252) + 1e-6
            sharpe = ann / vol
            
            eq = np.cumprod(1 + strat_ret)
            dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
            mdd = np.min(dd)
            
            avg_reg = np.mean(regimes) * 100
            reg_label = "GAUGE" if avg_reg > 50 else "SYMPL"
            
            print(f"{sym:<8} | {cum*100:>7.1f}% | {sharpe:>6.2f} | {mdd*100:>6.1f}% | {reg_label:<5} ({avg_reg:.0f}%)")
            total_ret.append(cum)

    print("-" * 65)
    print(f"Avg Portfolio Return: {np.mean(total_ret)*100:.1f}%")

if __name__ == "__main__":
    run()
EOF

# 10. 依赖配置
cat << 'EOF' > requirements.txt
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.0
EOF

chmod +x train_v5.py run_backtest_v5.py
echo "✅ v5.0 Architecture: Normalized."

