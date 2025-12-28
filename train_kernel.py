import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from src.data.fetcher import DataFetcher
from src.data.features import FeatureBuilder
from src.core.kernel import SyntropyQuantKernel
import numpy as np

def train(args):
    print(f">>> Training Physics Kernel (v4.0) | Symbols: {len(args.symbols.split(','))}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fetcher = DataFetcher()
    symbols = args.symbols.split(',')
    
    data_list, target_list = [], []
    
    for sym in symbols:
        df = fetcher.fetch(sym, "2015-01-01", "2022-12-31")
        if df.empty: continue
        
        feat = FeatureBuilder().build(df)
        if len(feat) == 0: continue
        
        # Label: Next Day Return
        closes = df['Close'].values[20:]
        if len(closes) > len(feat): closes = closes[:len(feat)]
        ret = (closes[1:] - closes[:-1]) / closes[:-1]
        
        # 3-Class Labels
        labels = np.ones(len(ret))
        labels[ret > 0.001] = 2 # Long
        labels[ret < -0.001] = 0 # Short
        
        data_list.append(feat[:-1])
        target_list.append(torch.tensor(labels, dtype=torch.long))
        
    if not data_list:
        print("No data available.")
        return

    X = torch.cat(data_list).to(device)
    Y = torch.cat(target_list).to(device)
    
    model = SyntropyQuantKernel(input_dim=4, hidden_dim=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    batch_size = 1024
    
    for epoch in range(args.epochs):
        perm = torch.randperm(len(X))
        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            idx = perm[i:i+batch_size]
            bx, by = X[idx], Y[idx]
            
            optimizer.zero_grad()
            logits, gamma = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} | Loss: {epoch_loss/(len(X)/batch_size):.4f}")
            
    torch.save(model.state_dict(), args.save)
    print(f"Saved: {args.save}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save', type=str, default='models/gauge_kernel.pt')
    parser.add_argument('--symbols', type=str, default='SPY')
    parser.add_argument('--providers', type=str, default='') 
    args = parser.parse_args()
    train(args)
