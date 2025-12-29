#!/bin/bash
# Syntropy Quant v5.0 Continuous Trading Starter
# This script starts the trading system in continuous mode (hourly cycles).

echo ">>> Initializing Syntropy Quant Trading System <<<"
echo ">>> Mode: PAPER"
echo ">>> Strategy: Hybrid Physics-Gauge (v4/v5)"
echo ">>> Frequency: HOURLY"

# Change to the project directory relative to the script location
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo ">>> Activating virtual environment..."
    source .venv/bin/activate
fi

# Run the trading system
python3 trading_system.py --mode paper --run continuous
