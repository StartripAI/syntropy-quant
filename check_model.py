#!/usr/bin/env python3
"""Check model architecture and weights"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Try to load and inspect the model
model_path = 'models/gauge_kernel.pt'
state_dict = torch.load(model_path, map_location='cpu')

print("=" * 70)
print("Model Keys in gauge_kernel.pt:")
print("=" * 70)
for key in sorted(state_dict.keys()):
    print(f"  {key}: {state_dict[key].shape}")

print()
print("=" * 70)
print("Analysis:")
print("=" * 70)

# Identify model type based on key patterns
has_manifold = any('manifold' in k for k in state_dict.keys())
has_engine = any('engine' in k for k in state_dict.keys())
has_policy_head = any('policy_head' in k for k in state_dict.keys())
has_confidence_head = any('confidence_head' in k for k in state_dict.keys())
has_encoder = any('encoder' in k for k in state_dict.keys())
has_dsu = any('dsu' in k for k in state_dict.keys())

if has_manifold and has_engine:
    print("✅ Model Type: GaugeFieldKernel")
    print("   - Has manifold (L_net, A_net)")
    print("   - Has engine")
    print("   - Has policy_head and confidence_head")
elif has_encoder and has_dsu:
    print("✅ Model Type: SyntropyQuantKernel (Physics Kernel)")
    print("   - Has encoder")
    print("   - Has DSU (Dissipative Symplectic Unit)")
else:
    print("⚠️  Unknown model architecture")

print()
print("Input dimension estimation:")
# Check first layer shape
first_layer_key = None
for key in state_dict.keys():
    if 'L_net' in key and key.endswith('weight'):
        first_layer_key = key
        break
    elif 'encoder' in key and key.endswith('weight'):
        first_layer_key = key
        break

if first_layer_key:
    shape = state_dict[first_layer_key].shape
    input_dim = shape[1]
    print(f"   First layer: {first_layer_key}")
    print(f"   Shape: {shape}")
    print(f"   Input dimension: {input_dim}")

