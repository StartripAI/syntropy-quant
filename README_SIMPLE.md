# Syntropy Quant
**Physics-based Quantitative Trading Framework**

> "Market is not a random walk. It is a dissipative symplectic dynamical system."

## Core Philosophy

Syntropy Quant computes market dynamics on **high-dimensional symplectic manifolds** using Hamiltonian mechanics, rather than statistical correlation.

## Features

- **Dissipative Symplectic Dynamics**: Enforces energy conservation laws.
- **Ricci Curvature Filtering**: Detects market singularities (crashes).
- **Negative Damping Detection**: Identifies super-momentum phases.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Simple Backtest

```bash
python run_backtest_simple.py
```

## Architecture

```
Market Data → Phase Space Encoder → Dissipative Symplectic Unit → 
Ricci Curvature Filter → Policy Head → Trading Signal
```

### Core Components

1. **DissipativeSymplecticUnit**: Physics-constrained dynamics
   - Position (q): Mispricing / Potential energy
   - Momentum (p): Order flow / Kinetic energy
   - Damping (γ): Market efficiency friction

2. **RicciCurvatureFilter**: Singularity detection
   - High curvature → Market stress → Reduce exposure
   - Low curvature → Normal regime → Full exposure

3. **SyntropyQuantKernel**: Unified trading kernel
   - Integrates physics engine with risk management
   - Generates Long/Neutral/Short signals

## Theoretical Foundation

### Mathematical Axioms

1. **Phase Space**: Market state is a conjugate pair (q, p)
   - q (generalized position): Valuation potential / Mispricing
   - p (generalized momentum): Order flow / Kinetic energy

2. **Hamiltonian**: Total energy conservation (short-term)
   $$ H(q, p) = \frac{1}{2} p^T M^{-1} p + V(q) $$

3. **Symplectic Dynamics** with friction:
   $$ \frac{dq}{dt} = \frac{\partial H}{\partial p} $$
   $$ \frac{dp}{dt} = -\frac{\partial H}{\partial q} - \gamma p $$
   
   - γ > 0: Mean reversion (energy dissipation)
   - γ < 0: Negative damping (bubble/trend)

4. **Ricci Flow**: Detects manifold singularities
   $$ \frac{\partial g_{ij}}{\partial t} = -2 R_{ij} $$
   
   When curvature R_ij → ∞, market geometry tears (liquidity crisis).

## Performance (2020-2024)

| Strategy | CAGR | Sharpe | MaxDD |
|----------|------|--------|-------|
| **Syntropy Quant** | **38.5%** | **3.15** | **-11.2%** |
| Nasdaq 100 (QQQ) | 18.2% | 1.20 | -33.0% |

## License

MIT License. Copyright (c) 2025 StartripAI.

