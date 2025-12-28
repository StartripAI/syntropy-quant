#!/bin/bash
# Syntropy Quant - Risk Check Runner
cd "$(dirname "$0")/.."
source .env 2>/dev/null || true

# Skip weekends
DAY=$(date +%u)
if [ "$DAY" -gt 5 ]; then
    exit 0
fi

python3 trading_system.py --risk-check >> logs/risk_$(date +%Y%m%d).log 2>&1
