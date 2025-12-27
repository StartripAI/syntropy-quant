# Syntropy Quant

**Physics-based Quantitative Trading Framework**

A unified framework that treats financial markets as dissipative Hamiltonian dynamical systems operating on symplectic manifolds.

## Core Philosophy

Traditional quantitative finance searches for patterns in Euclidean space. Syntropy Quant computes market dynamics on **high-dimensional symplectic manifolds** using Hamiltonian mechanics.

| Question | Physics Principle | Solution |
|----------|------------------|----------|
| **What to keep?** | Free Energy Principle | Only process "surprising" data |
| **How to scale?** | Renormalization Group | Multi-timeframe aggregation |
| **When to compute?** | Symplectic Dynamics | Volume-clock integration |
| **Where to go?** | Least Action | Geodesic path optimization |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SYNTROPY QUANT                               │
├─────────────────────────────────────────────────────────────────────┤
│  Market Data Stream                                                  │
│         │                                                            │
│         ▼                                                            │
│   ┌─────────────────┐                                               │
│   │   PHASE SPACE   │  Map price/volume to (q, p)                   │
│   │    ENCODER      │  q = mispricing, p = order flow               │
│   └────────┬────────┘                                               │
│            │                                                         │
│            ▼                                                         │
│   ┌─────────────────┐                                               │
│   │   DISSIPATIVE   │  Symplectic evolution with friction           │
│   │  SYMPLECTIC     │  dq/dt = ∂H/∂p                                │
│   │     UNIT        │  dp/dt = -∂H/∂q - γp                          │
│   └────────┬────────┘                                               │
│            │                                                         │
│            ▼                                                         │
│   ┌─────────────────┐                                               │
│   │    CURVATURE    │  Detect manifold singularities                │
│   │     FILTER      │  High curvature → reduce exposure             │
│   └────────┬────────┘                                               │
│            │                                                         │
│            ▼                                                         │
│   ┌─────────────────┐                                               │
│   │   POLICY HEAD   │  Risk-adjusted position sizing                │
│   └─────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src import SyntropyQuantKernel, DataFetcher, BacktestEngine

# Initialize
fetcher = DataFetcher()
engine = BacktestEngine()

# Fetch data
data = fetcher.fetch('NVDA', '2020-01-01', '2024-12-31')

# Run backtest
result = engine.run(data, 'NVDA')

print(f"Annual Return: {result.metrics.annual_return*100:.1f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown*100:.1f}%")
```

## Run Full Backtest

```bash
python run_backtest.py
```

Tests across:
- **Indices**: IWM, VTI, RSP
- **Tech**: NVDA, AAPL, MSFT, GOOGL, AMZN, META, TSLA
- **Pharma**: LLY, UNH, JNJ, PFE, ABBV, MRK
- **Consumer**: WMT, PG, KO, PEP, COST, MCD

Outputs (CSV):
- `output/<kernel>/summary_by_category.csv`
- `output/<kernel>/yearly_by_category.csv`
- `output/<kernel>/yearly_by_symbol.csv`
- `output/<kernel>/comparison_by_symbol.csv`
- `output/<kernel>/external_benchmarks.csv`

Optional external benchmarks:
- Fill `benchmarks/quant_benchmarks.csv` to include external comparisons.
- If `Ticker` is provided, metrics are computed automatically.

## Core Components

### Dissipative Symplectic Unit (DSU)

Replaces LSTM/GRU with physics-constrained dynamics:

```python
from src.core import DissipativeSymplecticUnit

dsu = DissipativeSymplecticUnit(hidden_dim=64)

# Evolve state
q_new, p_new = dsu(q, p, dt)

# Check damping (market efficiency)
gamma = dsu.get_damping()
```

### Curvature Filter

Detects market regime singularities:

```python
from src.core import RicciCurvatureFilter

filter = RicciCurvatureFilter(hidden_dim=64, threshold=2.0)

curvature, regime = filter(state, energy)
# regime: 'normal', 'high_vol', 'crisis', 'bubble'

if filter.should_exit(curvature):
    position = 0  # Exit to cash
```

### Surprise Filter

Free Energy Principle - only process surprising data:

```python
from src.core import SurpriseFilter

filter = SurpriseFilter(threshold_k=2.0)

result = filter.filter(new_observation)
if result.keep:
    # Process this data point
    ...
```

### Gauge Field Kernel (Geometry + Path Integrals)

Optional kernel that models market geometry and drift as a gauge field.
It estimates least-action future states and scales risk by free energy.

```python
from src.core import GaugeFieldKernel, GaugeConfig

config = GaugeConfig(input_dim=12)
kernel = GaugeFieldKernel(input_dim=12, config=config)

logits, free_energy, confidence = kernel(torch.randn(4, 12))
```

## Training (Offline or Online)

Train the gauge kernel on historical data (defaults to a full multi-sector universe):

```bash
python train_kernel.py --epochs 10 --save models/gauge_kernel.pt
```

Optional: override provider priority:

```bash
python train_kernel.py --providers tiingo,polygon,yahoo,stooq
```

Online training mode (streaming updates):

```bash
python train_kernel.py --symbols AAPL --epochs 1 --online --save models/gauge_kernel.pt
```

Run backtest with a trained model:

```bash
python run_backtest.py --kernel gauge --model models/gauge_kernel.pt
```

Lower the trade threshold if you see zero trades:

```bash
python run_backtest.py --kernel gauge --model models/gauge_kernel.pt --min-trade-threshold 0.005
```

## Data Providers (Open + Private)

Default priority is: `tiingo` (if key) → `polygon` (if key) → `yahoo` → `stooq`.

Set keys via environment variables (never commit keys):

```bash
export TIINGO_API_KEY="..."
export POLYGON_API_KEY="..."
```

You can override provider order per run:

```bash
python run_backtest.py --kernel gauge --providers polygon,tiingo,yahoo,stooq
```

## Theoretical Foundations

1. **Hamiltonian Mechanics**: Market state evolves on phase space (position, momentum)
2. **Symplectic Geometry**: Phase space volume is preserved (energy conservation)
3. **Dissipation**: Transaction costs and information decay add friction
4. **Free Energy Principle**: Filter noise, keep only surprising signals
5. **Renormalization Group**: Scale-invariant pattern detection

## Performance Characteristics

| Metric | Target |
|--------|--------|
| Sharpe Ratio | > 2.0 |
| Max Drawdown | < 20% |
| Win Rate | > 50% |
| Calmar Ratio | > 1.5 |

## Risk Management

Built-in risk controls:
- Volatility targeting
- Drawdown-based position scaling
- Curvature-based regime detection
- Transaction cost modeling

## License

MIT License
