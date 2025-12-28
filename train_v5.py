
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from src.core.hybrid.kernel_v5 import SyntropyQuantKernelV5
from src.data.features_v5 import FeatureBuilderV5
from src.data.fetcher import DataFetcher

def physics_loss(output, target_cls, target_regime, criterion_cls, criterion_reg):
    loss_cls = criterion_cls(output['logits'], target_cls)
    loss_regime = criterion_reg(output['regime_logits'].squeeze(), target_regime)
    p = output['regime_prob']
    loss_higgs = 0.15 * torch.mean(p * (1.0 - p))
    return loss_cls + 1.0 * loss_regime + 0.15 * loss_higgs

def train(args):
    print(f"🌌 SYNTROPY v5.2 | Criticality Injection (Soft Higgs)"")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fetcher = DataFetcher()
    builder = FeatureBuilderV5()
    symbols = args.symbols.split(',')
    
    X_list, Y_list, Reg_list = [], [], []
    
    print(">>> Loading Phase Space (Relativistic Balancing)...")
    for sym in symbols:
        df = fetcher.fetch(sym, "2016-01-01", "2023-01-01")
        if df.empty: continue
        feat, phys_label = builder.build(df)
        if len(feat) == 0: continue
        
        closes = df['Close'].values[50:]
        ret = (closes[3:] - closes[:-3]) / closes[:-3]
        
        min_len = min(len(feat), len(ret))
        feat = feat[:min_len]
        ret = ret[:min_len]
        phys_label = phys_label[:min_len]
        
        y = np.ones(len(ret))
        y[ret > 0.005] = 2
        y[ret < -0.005] = 0
        
        X_list.append(feat)
        Y_list.append(torch.tensor(y, dtype=torch.long))
        Reg_list.append(phys_label)
        
    if not X_list: 
        print("No data loaded.")
        return

    X = torch.cat(X_list).to(device)
    Y = torch.cat(Y_list).to(device)
    R = torch.cat(Reg_list).to(device)
    
    model = SyntropyQuantKernelV5(input_dim=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    crit_cls = nn.CrossEntropyLoss()
    crit_reg = nn.BCEWithLogitsLoss()
    
    model.train()
    print(f"   Training on {len(X)} samples... Forcing Bifurcation...")
    
    for epoch in range(args.epochs):
        perm = torch.randperm(len(X))
        total_loss = 0
        
        for i in range(0, len(X), 2048):
            idx = perm[i:i+2048]
            bx, by, br = X[idx], Y[idx], R[idx]
            
            optimizer.zero_grad()
            out = model(bx)
            loss = physics_loss(out, by, br, crit_cls, crit_reg)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                full_regime = model(X[:5000])['regime_prob']
                gauge_pct = (full_regime > 0.5).float().mean().item() * 100
                print(f"   Epoch {epoch+1:02d} | Loss: {total_loss/2048:.4f} | Gauge State: {gauge_pct:.1f}% (Target: 20-40%)")
    
    torch.save(model.state_dict(), args.save)
    print(f">>> Model Saved: {args.save}")
    print(f"   Input Dim: 6 (Ret, Vol, Flow, Hurst, Pot, Trend, Regime_Label)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--save', type=str, default='models/syntropy_v5.pt')
    parser.add_argument('--symbols', type=str, default="SPY,QQQ,NVDA,AAPL,MSFT,GOOGL,AMZN,META,TSLA,LLY,WMT,KO,PEP")
    args = parser.parse_args()
    train(args)
