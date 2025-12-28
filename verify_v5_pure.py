"""
Verify Syntropy v5.0 Pure Model (No Regime Supervision)
"""
import torch
import numpy as np
from src.core.hybrid.kernel_v5 import SyntropyQuantKernelV5

def generate_regime_test_samples(n_samples=100):
    stable_samples = []
    turbulent_samples = []
    
    for _ in range(n_samples):
        feat = np.array([
            np.random.normal(0, 0.005),
            np.random.uniform(0.008, 0.015),
            np.random.uniform(0.012, 0.02),
            np.random.normal(0, 0.2),
            np.random.uniform(0.3, 0.45),
            np.random.normal(0, 0.01),
            np.random.uniform(-0.5, 0.5),
        ])
        stable_samples.append(feat)
    
    for _ in range(n_samples):
        feat = np.array([
            np.random.normal(0.001, 0.02),
            np.random.uniform(0.02, 0.05),
            np.random.uniform(0.03, 0.06),
            np.random.normal(0.5, 0.8),
            np.random.uniform(0.55, 0.8),
            np.random.uniform(-0.1, 0.2),
            np.random.uniform(-0.5, 0.5),
        ])
        turbulent_samples.append(feat)
    
    return np.array(stable_samples), np.array(turbulent_samples)

print("\n" + "="*70)
print("  SYNTROPY v5.0 PURE MODEL VERIFICATION")
print("="*70)

model = SyntropyQuantKernelV5(input_dim=7, hidden_dim=128, latent_dim=16)
try:
    model.load_state_dict(torch.load("models/syntropy_v5_pure.pt", map_location='cpu'))
    print("✅ Pure model loaded\n")
except Exception as e:
    print(f"❌ Failed to load: {e}")
    exit(1)

model.eval()

stable, turbulent = generate_regime_test_samples(n_samples=500)
print(f"Stable samples: {len(stable)}")
print(f"Turbulent samples: {len(turbulent)}\n")

with torch.no_grad():
    stable_reg = model(torch.tensor(stable, dtype=torch.float32))['regime'].numpy()
    turb_reg = model(torch.tensor(turbulent, dtype=torch.float32))['regime'].numpy()

stable_mean = stable_reg.mean() * 100
turb_mean = turb_reg.mean() * 100

print("📊 REGIME PREDICTION:")
print("-"*70)
print(f"{'Stable':<12} | {stable_mean:>10.1f}% | (std: {stable_reg.std()*100:.1f}%)")
print(f"{'Turbulent':<12} | {turb_mean:>10.1f}% | (std: {turb_reg.std()*100:.1f}%)")
print("-"*70)

sep = abs(turb_mean - stable_mean)
print(f"\nSeparation: {sep:.1f}%")

if sep > 30:
    print("✅ Good regime separation")
else:
    print("⚠️  Poor regime separation")

print()

