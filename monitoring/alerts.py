#!/usr/bin/env python3
"""
Syntropy Quant - Alerting System

Supports:
- Telegram notifications
- Discord webhooks
- Email alerts (optional)

Usage:
    from monitoring.alerts import AlertManager

    alerts = AlertManager()
    alerts.send_trade_alert("BUY 100 AAPL @ $150.00")
    alerts.send_risk_alert("High drawdown detected: -15%")
"""

import os
import json
import logging
import requests
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum
from pathlib import Path


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = None
    data: Dict = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.data is None:
            self.data = {}


class TelegramNotifier:
    """Telegram Bot notifications"""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        self.logger = logging.getLogger('TelegramNotifier')

    def send(self, alert: Alert) -> bool:
        """Send alert via Telegram"""
        try:
            # Format message
            emoji = {
                AlertLevel.INFO: "",
                AlertLevel.WARNING: "",
                AlertLevel.ERROR: "",
                AlertLevel.CRITICAL: "",
            }.get(alert.level, "")

            message = f"{emoji} *{alert.title}*\n\n"
            message += f"{alert.message}\n\n"
            message += f"_{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_"

            # Send
            response = requests.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "Markdown"
                },
                timeout=10
            )

            if response.status_code == 200:
                return True
            else:
                self.logger.error(f"Telegram error: {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"Telegram send failed: {e}")
            return False


class DiscordNotifier:
    """Discord Webhook notifications"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.logger = logging.getLogger('DiscordNotifier')

    def send(self, alert: Alert) -> bool:
        """Send alert via Discord webhook"""
        try:
            # Color based on level
            color = {
                AlertLevel.INFO: 0x3498db,      # Blue
                AlertLevel.WARNING: 0xf39c12,   # Orange
                AlertLevel.ERROR: 0xe74c3c,     # Red
                AlertLevel.CRITICAL: 0x9b59b6,  # Purple
            }.get(alert.level, 0x95a5a6)

            # Create embed
            embed = {
                "title": alert.title,
                "description": alert.message,
                "color": color,
                "timestamp": alert.timestamp.isoformat(),
                "footer": {"text": "Syntropy Quant Trading System"}
            }

            # Add fields from data
            if alert.data:
                fields = []
                for key, value in alert.data.items():
                    fields.append({
                        "name": key,
                        "value": str(value),
                        "inline": True
                    })
                embed["fields"] = fields[:25]  # Discord limit

            response = requests.post(
                self.webhook_url,
                json={"embeds": [embed]},
                timeout=10
            )

            return response.status_code == 204

        except Exception as e:
            self.logger.error(f"Discord send failed: {e}")
            return False


class AlertManager:
    """
    Unified alerting manager

    Sends alerts to all configured channels.
    """

    def __init__(self):
        self.logger = logging.getLogger('AlertManager')
        self._load_config()
        self._init_notifiers()

    def _load_config(self):
        """Load configuration from environment"""
        # Try loading .env file
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        if value:  # Only set if not empty
                            os.environ[key] = value

        self.telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        self.discord_webhook = os.environ.get('DISCORD_WEBHOOK_URL')

    def _init_notifiers(self):
        """Initialize available notifiers"""
        self.notifiers = []

        if self.telegram_token and self.telegram_chat_id:
            self.notifiers.append(
                TelegramNotifier(self.telegram_token, self.telegram_chat_id)
            )
            self.logger.info("Telegram notifications enabled")

        if self.discord_webhook:
            self.notifiers.append(
                DiscordNotifier(self.discord_webhook)
            )
            self.logger.info("Discord notifications enabled")

        if not self.notifiers:
            self.logger.warning("No notification channels configured")

    def send(self, alert: Alert) -> bool:
        """Send alert to all configured channels"""
        if not self.notifiers:
            self.logger.debug(f"Alert (no channels): {alert.title}")
            return False

        success = False
        for notifier in self.notifiers:
            if notifier.send(alert):
                success = True

        return success

    # Convenience methods

    def send_trade_alert(self, message: str, data: Dict = None):
        """Send trade execution alert"""
        alert = Alert(
            level=AlertLevel.INFO,
            title="Trade Executed",
            message=message,
            data=data or {}
        )
        return self.send(alert)

    def send_signal_alert(self, symbol: str, signal: float, action: str):
        """Send trading signal alert"""
        alert = Alert(
            level=AlertLevel.INFO,
            title=f"Signal: {symbol}",
            message=f"Action: {action}\nSignal Strength: {signal:.3f}",
            data={"Symbol": symbol, "Signal": f"{signal:.3f}", "Action": action}
        )
        return self.send(alert)

    def send_risk_alert(self, message: str, level: AlertLevel = AlertLevel.WARNING):
        """Send risk warning alert"""
        alert = Alert(
            level=level,
            title="Risk Alert",
            message=message
        )
        return self.send(alert)

    def send_system_alert(self, message: str, level: AlertLevel = AlertLevel.INFO):
        """Send system status alert"""
        alert = Alert(
            level=level,
            title="System Status",
            message=message
        )
        return self.send(alert)

    def send_daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        trades: int,
        top_performers: List[tuple]
    ):
        """Send daily performance summary"""
        pnl_pct = (daily_pnl / equity) * 100 if equity > 0 else 0

        message = f"Equity: ${equity:,.2f}\n"
        message += f"Daily P&L: ${daily_pnl:+,.2f} ({pnl_pct:+.2f}%)\n"
        message += f"Trades: {trades}\n\n"

        if top_performers:
            message += "Top Performers:\n"
            for sym, pnl in top_performers[:5]:
                message += f"  {sym}: {pnl:+.2f}%\n"

        alert = Alert(
            level=AlertLevel.INFO,
            title="Daily Summary",
            message=message,
            data={
                "Equity": f"${equity:,.2f}",
                "P&L": f"${daily_pnl:+,.2f}",
                "Trades": trades
            }
        )
        return self.send(alert)

    def send_emergency_alert(self, message: str):
        """Send critical emergency alert"""
        alert = Alert(
            level=AlertLevel.CRITICAL,
            title="EMERGENCY",
            message=message
        )
        return self.send(alert)

    def send_retrain_alert(self, version: str, metrics: Dict):
        """Send model retrain completion alert"""
        message = f"New model version: {version}\n\n"
        if metrics:
            for key, value in metrics.items():
                message += f"{key}: {value}\n"

        alert = Alert(
            level=AlertLevel.INFO,
            title="Model Retrained",
            message=message,
            data=metrics
        )
        return self.send(alert)


def setup_telegram():
    """Interactive setup for Telegram bot"""
    print("=" * 60)
    print("Telegram Bot Setup")
    print("=" * 60)
    print()
    print("1. Open Telegram and search for @BotFather")
    print("2. Send /newbot and follow instructions")
    print("3. Copy the bot token")
    print()
    print("To get your chat ID:")
    print("1. Send a message to your bot")
    print("2. Visit: https://api.telegram.org/bot<TOKEN>/getUpdates")
    print("3. Find 'chat':{'id': YOUR_ID}")
    print()

    token = input("Enter bot token: ").strip()
    chat_id = input("Enter chat ID: ").strip()

    if token and chat_id:
        # Test
        notifier = TelegramNotifier(token, chat_id)
        alert = Alert(
            level=AlertLevel.INFO,
            title="Test Alert",
            message="Syntropy Quant alerting is working!"
        )

        if notifier.send(alert):
            print("Success! Check your Telegram.")

            # Save to .env
            env_path = Path(__file__).parent.parent / '.env'
            with open(env_path, 'a') as f:
                f.write(f"\nTELEGRAM_BOT_TOKEN={token}\n")
                f.write(f"TELEGRAM_CHAT_ID={chat_id}\n")
            print(f"Saved to {env_path}")
        else:
            print("Failed to send test message. Check your credentials.")


def setup_discord():
    """Interactive setup for Discord webhook"""
    print("=" * 60)
    print("Discord Webhook Setup")
    print("=" * 60)
    print()
    print("1. In Discord, go to Server Settings > Integrations > Webhooks")
    print("2. Click 'New Webhook'")
    print("3. Set name and channel, then copy webhook URL")
    print()

    webhook_url = input("Enter webhook URL: ").strip()

    if webhook_url:
        # Test
        notifier = DiscordNotifier(webhook_url)
        alert = Alert(
            level=AlertLevel.INFO,
            title="Test Alert",
            message="Syntropy Quant alerting is working!"
        )

        if notifier.send(alert):
            print("Success! Check your Discord channel.")

            # Save to .env
            env_path = Path(__file__).parent.parent / '.env'
            with open(env_path, 'a') as f:
                f.write(f"\nDISCORD_WEBHOOK_URL={webhook_url}\n")
            print(f"Saved to {env_path}")
        else:
            print("Failed to send test message. Check your webhook URL.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Alert System Setup')
    parser.add_argument('--telegram', action='store_true', help='Setup Telegram')
    parser.add_argument('--discord', action='store_true', help='Setup Discord')
    parser.add_argument('--test', action='store_true', help='Test all channels')

    args = parser.parse_args()

    if args.telegram:
        setup_telegram()
    elif args.discord:
        setup_discord()
    elif args.test:
        manager = AlertManager()
        manager.send_system_alert("Test message from Syntropy Quant")
        print("Test alert sent to all configured channels")
    else:
        print("Syntropy Quant - Alerting System")
        print()
        print("Setup options:")
        print("  python monitoring/alerts.py --telegram   # Setup Telegram")
        print("  python monitoring/alerts.py --discord    # Setup Discord")
        print("  python monitoring/alerts.py --test       # Test all channels")
