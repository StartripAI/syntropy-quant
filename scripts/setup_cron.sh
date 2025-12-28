#!/bin/bash
# Syntropy Quant - Cron Job Setup
#
# This script sets up automated daily trading
#
# Usage:
#   ./scripts/setup_cron.sh install    # Install cron jobs
#   ./scripts/setup_cron.sh remove     # Remove cron jobs
#   ./scripts/setup_cron.sh status     # Check status

set -e

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_PATH=$(which python3)
LOG_DIR="$PROJECT_DIR/logs"
CRON_TAG="# syntropy-quant"

# Trading schedule (Eastern Time)
# Market opens at 9:30 AM ET, closes at 4:00 PM ET
TRADING_TIME="35 9"    # 9:35 AM (5 min after open)
RETRAIN_TIME="0 18"    # 6:00 PM (after market close)
RISK_CHECK="0 12"      # 12:00 PM (midday check)

echo "======================================"
echo "Syntropy Quant - Cron Job Setup"
echo "======================================"
echo "Project: $PROJECT_DIR"
echo "Python:  $PYTHON_PATH"
echo ""

# Ensure logs directory exists
mkdir -p "$LOG_DIR"

install_cron() {
    echo "Installing cron jobs..."

    # Create wrapper script for trading
    cat > "$PROJECT_DIR/scripts/run_trading.sh" << 'EOF'
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
EOF
    chmod +x "$PROJECT_DIR/scripts/run_trading.sh"

    # Create wrapper script for retraining
    cat > "$PROJECT_DIR/scripts/run_retrain.sh" << 'EOF'
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
EOF
    chmod +x "$PROJECT_DIR/scripts/run_retrain.sh"

    # Create wrapper script for risk check
    cat > "$PROJECT_DIR/scripts/run_risk_check.sh" << 'EOF'
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
EOF
    chmod +x "$PROJECT_DIR/scripts/run_risk_check.sh"

    # Build cron entries
    CRON_ENTRIES=$(cat << EOF
# Syntropy Quant Trading System $CRON_TAG
# Daily trading at market open (9:35 AM ET, Mon-Fri)
$TRADING_TIME * * 1-5 $PROJECT_DIR/scripts/run_trading.sh $CRON_TAG

# Midday risk check (12:00 PM ET, Mon-Fri)
$RISK_CHECK * * 1-5 $PROJECT_DIR/scripts/run_risk_check.sh $CRON_TAG

# Weekly model retrain (6:00 PM ET, Sunday)
$RETRAIN_TIME * * 0 $PROJECT_DIR/scripts/run_retrain.sh $CRON_TAG
EOF
)

    # Get existing crontab (excluding our entries)
    EXISTING=$(crontab -l 2>/dev/null | grep -v "$CRON_TAG" || true)

    # Install new crontab
    echo "$EXISTING" | cat - <(echo "$CRON_ENTRIES") | crontab -

    echo "Cron jobs installed!"
    echo ""
    echo "Schedule:"
    echo "  - Daily Trading: 9:35 AM ET (Mon-Fri)"
    echo "  - Risk Check:    12:00 PM ET (Mon-Fri)"
    echo "  - Model Retrain: 6:00 PM ET (Sunday)"
    echo ""
    echo "Logs: $LOG_DIR"
}

remove_cron() {
    echo "Removing Syntropy Quant cron jobs..."

    # Remove our entries
    crontab -l 2>/dev/null | grep -v "$CRON_TAG" | crontab - 2>/dev/null || true

    echo "Cron jobs removed."
}

show_status() {
    echo "Current cron jobs:"
    echo ""
    crontab -l 2>/dev/null | grep -A1 "syntropy" || echo "No Syntropy Quant cron jobs found"
    echo ""
    echo "Recent logs:"
    ls -lt "$LOG_DIR"/*.log 2>/dev/null | head -5 || echo "No logs found"
}

# Main
case "${1:-}" in
    install)
        install_cron
        ;;
    remove)
        remove_cron
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {install|remove|status}"
        echo ""
        echo "Commands:"
        echo "  install  - Install cron jobs for automated trading"
        echo "  remove   - Remove all Syntropy Quant cron jobs"
        echo "  status   - Show current cron jobs and recent logs"
        ;;
esac
