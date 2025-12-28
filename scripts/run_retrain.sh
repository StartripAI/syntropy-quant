#!/bin/bash
# Syntropy Quant - Retrain Runner
cd "$(dirname "$0")/.."
source .env 2>/dev/null || true

# Skip weekends
DAY=$(date +%u)
if [ "$DAY" -gt 5 ]; then
    exit 0
fi

# Only retrain on Sundays or if forced
if [ "$DAY" -eq 7 ] || [ "$1" == "--force" ]; then
    python3 trading_system.py --retrain >> logs/retrain_$(date +%Y%m%d).log 2>&1
fi
