#!/bin/bash
# Syntropy Quant - Trading Runner
cd "$(dirname "$0")/.."
source .env 2>/dev/null || true

# Check if market is open (rough check)
DAY=$(date +%u)
if [ "$DAY" -gt 5 ]; then
    echo "Weekend - skipping"
    exit 0
fi

# Run trading cycle
python3 trading_system.py --mode paper --run once >> logs/trading_$(date +%Y%m%d).log 2>&1

# Send completion notification
python3 -c "
from monitoring.alerts import AlertManager
import json
with open('logs/result_$(date +%Y%m%d).json', 'r') as f:
    result = json.load(f)
AlertManager().send_system_alert(
    f\"Trading cycle complete: {result.get('orders_filled', 0)} orders filled\"
)
" 2>/dev/null || true
