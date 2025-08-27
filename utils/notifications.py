"""
Notification Manager - Handles all notifications (Telegram, Email, etc.)
Provides real-time alerts and status updates for the trading bot.
"""

import asyncio
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass

from config.config import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class NotificationMessage:
    """Notification message structure"""
    message: str
    priority: str  # 'low', 'medium', 'high', 'critical'
    category: str  # 'trade', 'system', 'error', 'performance'
    timestamp: datetime
    data: Optional[Dict] = None

class TelegramNotifier:
    """Telegram notification handler"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_delay = 1.0  # Seconds between messages
        self.last_message_time = datetime.utcnow() - timedelta(seconds=2)
        
    async def initialize(self):
        """Initialize Telegram connection"""
        try:
            self.session = aiohttp.ClientSession()
            
            # Test connection
            await self.test_connection()
            logger.info("✅ Telegram notifier initialized")
            
        except Exception as e:
            logger.error(f"❌ Telegram initialization failed: {e}")
            raise
    
    async def test_connection(self):
        """Test Telegram bot connection"""
        try:
            url = f"{self.base_url}/getMe"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    bot_info = data.get('result', {})
                    logger.info(f"🤖 Telegram bot connected: {bot_info.get('first_name')}")
                else:
                    raise Exception(f"Telegram API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Telegram connection test failed: {e}")
            raise
    
    async def send_message(self, message: str, parse_mode: str = "HTML",
                          disable_notification: bool = False):
        """Send message via Telegram"""
        try:
            # Rate limiting
            time_since_last = (datetime.utcnow() - self.last_message_time).total_seconds()
            if time_since_last < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - time_since_last)
            
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_notification': disable_notification
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    self.last_message_time = datetime.utcnow()
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Telegram send failed: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Telegram message failed: {e}")
            return False
    
    async def send_photo(self, photo_path: str, caption: str = ""):
        """Send photo via Telegram"""
        try:
            url = f"{self.base_url}/sendPhoto"
            
            with open(photo_path, 'rb') as photo_file:
                data = aiohttp.FormData()
                data.add_field('chat_id', self.chat_id)
                data.add_field('photo', photo_file)
                if caption:
                    data.add_field('caption', caption)
                
                async with self.session.post(url, data=data) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"❌ Telegram photo send failed: {e}")
            return False
    
    async def close(self):
        """Close Telegram session"""
        if self.session:
            await self.session.close()

class NotificationManager:
    """Main notification management system"""
    
    def __init__(self):
        self.telegram_notifier: Optional[TelegramNotifier] = None
        self.enabled = config.notifications.enable_notifications
        
        # Message queue and batching
        self.message_queue: List[NotificationMessage] = []
        self.batch_size = 5
        self.batch_interval = 30  # seconds
        
        # Notification settings
        self.notification_settings = {
            'trade_entry': True,
            'trade_exit': True,
            'high_pnl': True,
            'risk_alerts': True,
            'system_errors': True,
            'daily_summary': True,
            'performance_updates': False,
            'debug_messages': False
        }
        
        # Rate limiting
        self.message_counts = {
            'hourly': 0,
            'daily': 0
        }
        self.rate_limits = {
            'hourly': 50,
            'daily': 200
        }
        
        self.last_reset = {
            'hourly': datetime.utcnow(),
            'daily': datetime.utcnow()
        }
    
    async def initialize(self):
        """Initialize notification manager"""
        try:
            if not self.enabled:
                logger.info("📱 Notifications disabled")
                return
            
            # Initialize Telegram if configured
            if config.notifications.telegram_token and config.notifications.telegram_chat_id:
                self.telegram_notifier = TelegramNotifier(
                    config.notifications.telegram_token,
                    config.notifications.telegram_chat_id
                )
                await self.telegram_notifier.initialize()
            
            # Start batch processing
            asyncio.create_task(self.batch_processor())
            
            logger.info("📱 Notification manager initialized")
            
        except Exception as e:
            logger.error(f"❌ Notification manager initialization failed: {e}")
            self.enabled = False
    
    async def send_notification(self, message: str, priority: str = 'medium',
                              category: str = 'system', data: Optional[Dict] = None,
                              immediate: bool = False):
        """Send notification (queued or immediate)"""
        try:
            if not self.enabled:
                return
            
            # Check rate limits
            if not self.check_rate_limits():
                logger.debug("📱 Rate limit reached, skipping notification")
                return
            
            notification = NotificationMessage(
                message=message,
                priority=priority,
                category=category,
                timestamp=datetime.utcnow(),
                data=data
            )
            
            if immediate or priority == 'critical':
                await self.send_immediate(notification)
            else:
                self.message_queue.append(notification)
                
        except Exception as e:
            logger.error(f"❌ Notification send failed: {e}")
    
    async def send_immediate(self, notification: NotificationMessage):
        """Send notification immediately"""
        try:
            formatted_message = self.format_message(notification)
            
            if self.telegram_notifier:
                success = await self.telegram_notifier.send_message(
                    formatted_message,
                    disable_notification=(notification.priority == 'low')
                )
                
                if success:
                    self.increment_message_count()
                    logger.debug(f"📱 Immediate notification sent: {notification.category}")
                    
        except Exception as e:
            logger.error(f"❌ Immediate notification failed: {e}")
    
    async def batch_processor(self):
        """Process queued notifications in batches"""
        try:
            while True:
                await asyncio.sleep(self.batch_interval)
                
                if not self.message_queue:
                    continue
                
                # Group messages by category
                batched_messages = self.group_messages_by_category()
                
                for category, messages in batched_messages.items():
                    if self.notification_settings.get(category, True):
                        batch_message = self.create_batch_message(category, messages)
                        
                        if self.telegram_notifier:
                            await self.telegram_notifier.send_message(batch_message)
                            self.increment_message_count()
                
                self.message_queue.clear()
                
        except Exception as e:
            logger.error(f"❌ Batch processor error: {e}")
    
    def group_messages_by_category(self) -> Dict[str, List[NotificationMessage]]:
        """Group queued messages by category"""
        grouped = {}
        
        for message in self.message_queue:
            category = message.category
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(message)
        
        return grouped
    
    def create_batch_message(self, category: str, messages: List[NotificationMessage]) -> str:
        """Create batched message for multiple notifications"""
        if len(messages) == 1:
            return self.format_message(messages[0])
        
        category_emojis = {
            'trade': '💰',
            'system': '⚙️',
            'performance': '📊',
            'error': '❌'
        }
        
        emoji = category_emojis.get(category, '📱')
        header = f"{emoji} <b>{category.title()} Updates ({len(messages)})</b>\n\n"
        
        batch_content = []
        for msg in messages[-5:]:  # Show last 5 messages
            time_str = msg.timestamp.strftime("%H:%M")
            batch_content.append(f"• <code>{time_str}</code> {msg.message}")
        
        if len(messages) > 5:
            batch_content.append(f"... and {len(messages) - 5} more")
        
        return header + "\n".join(batch_content)
    
    def format_message(self, notification: NotificationMessage) -> str:
        """Format notification message"""
        priority_emojis = {
            'low': '🔵',
            'medium': '🟡',
            'high': '🟠',
            'critical': '🔴'
        }
        
        category_emojis = {
            'trade': '💰',
            'system': '⚙️',
            'performance': '📊',
            'error': '❌',
            'risk': '⚠️'
        }
        
        priority_emoji = priority_emojis.get(notification.priority, '📱')
        category_emoji = category_emojis.get(notification.category, '📱')
        
        timestamp = notification.timestamp.strftime("%H:%M:%S")
        
        return (f"{priority_emoji} {category_emoji} <b>{notification.category.title()}</b>\n"
                f"<code>{timestamp}</code> {notification.message}")
    
    def check_rate_limits(self) -> bool:
        """Check if rate limits allow sending message"""
        current_time = datetime.utcnow()
        
        # Reset hourly counter
        if (current_time - self.last_reset['hourly']).total_seconds() >= 3600:
            self.message_counts['hourly'] = 0
            self.last_reset['hourly'] = current_time
        
        # Reset daily counter
        if (current_time - self.last_reset['daily']).total_seconds() >= 86400:
            self.message_counts['daily'] = 0
            self.last_reset['daily'] = current_time
        
        # Check limits
        return (self.message_counts['hourly'] < self.rate_limits['hourly'] and
                self.message_counts['daily'] < self.rate_limits['daily'])
    
    def increment_message_count(self):
        """Increment message counters"""
        self.message_counts['hourly'] += 1
        self.message_counts['daily'] += 1
    
    # Specialized notification methods
    async def send_startup_message(self):
        """Send bot startup notification"""
        message = (f"🚀 <b>Trading Bot Started</b>\n\n"
                  f"📊 Mode: <code>{config.trading.mode.upper()}</code>\n"
                  f"💰 Capital: <code>${config.trading.initial_capital:,.2f}</code>\n"
                  f"🎯 Max Risk: <code>{config.trading.risk_per_trade:.1%}</code>\n"
                  f"⏰ Started: <code>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</code>")
        
        await self.send_notification(message, priority='high', category='system', immediate=True)
    
    async def send_shutdown_message(self):
        """Send bot shutdown notification"""
        message = (f"🛑 <b>Trading Bot Shutdown</b>\n\n"
                  f"⏰ Stopped: <code>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</code>\n"
                  f"📊 Session completed")
        
        await self.send_notification(message, priority='high', category='system', immediate=True)
    
    async def send_trade_notification(self, trade_type: str, symbol: str, side: str,
                                    amount: float, price: float, strategy: str):
        """Send trade notification"""
        side_emoji = "🟢" if side == "long" else "🔴"
        action_emoji = "🔵" if trade_type == "entry" else "🔄"
        
        message = (f"{action_emoji} <b>{trade_type.upper()}</b> {side_emoji}\n\n"
                  f"📈 Symbol: <code>{symbol}</code>\n"
                  f"💰 Amount: <code>{amount:.6f}</code>\n"
                  f"💵 Price: <code>${price:.4f}</code>\n"
                  f"🎯 Strategy: <code>{strategy}</code>\n"
                  f"💲 Value: <code>${amount * price:.2f}</code>")
        
        await self.send_notification(message, priority='medium', category='trade')
    
    async def send_pnl_notification(self, symbol: str, pnl: float, pnl_pct: float,
                                  trade_type: str = "exit"):
        """Send PnL notification"""
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        profit_loss = "PROFIT" if pnl >= 0 else "LOSS"
        
        message = (f"{pnl_emoji} <b>{profit_loss}</b>\n\n"
                  f"📈 Symbol: <code>{symbol}</code>\n"
                  f"💰 PnL: <code>${pnl:.2f}</code>\n"
                  f"📊 Return: <code>{pnl_pct:+.2f}%</code>")
        
        # High priority for significant gains/losses
        priority = 'high' if abs(pnl_pct) > 5 else 'medium'
        await self.send_notification(message, priority=priority, category='trade')
    
    async def send_portfolio_update(self, total_balance: float, daily_pnl: float,
                                  daily_pnl_pct: float, positions_count: int):
        """Send portfolio update notification"""
        balance_emoji = "💰" if daily_pnl >= 0 else "💸"
        trend_emoji = "📈" if daily_pnl >= 0 else "📉"
        
        message = (f"{balance_emoji} <b>Portfolio Update</b> {trend_emoji}\n\n"
                  f"💵 Total Balance: <code>${total_balance:.2f}</code>\n"
                  f"📊 Daily PnL: <code>${daily_pnl:.2f} ({daily_pnl_pct:+.2f}%)</code>\n"
                  f"📍 Active Positions: <code>{positions_count}</code>")
        
        await self.send_notification(message, priority='medium', category='performance')
    
    async def send_risk_alert(self, alert_type: str, symbol: str, details: str,
                            severity: str = 'medium'):
        """Send risk management alert"""
        severity_emojis = {
            'low': '🟡',
            'medium': '🟠',
            'high': '🔴',
            'critical': '🚨'
        }
        
        severity_emoji = severity_emojis.get(severity, '⚠️')
        
        message = (f"{severity_emoji} <b>RISK ALERT</b>\n\n"
                  f"📈 Symbol: <code>{symbol}</code>\n"
                  f"⚠️ Type: <code>{alert_type}</code>\n"
                  f"📋 Details: {details}")
        
        priority = 'critical' if severity == 'critical' else 'high'
        await self.send_notification(message, priority=priority, category='risk', immediate=True)
    
    async def send_error_notification(self, error_info: Dict[str, Any]):
        """Send error notification"""
        message = (f"🚨 <b>SYSTEM ERROR</b>\n\n"
                  f"🔧 Component: <code>{error_info.get('component', 'Unknown')}</code>\n"
                  f"❌ Error: <code>{error_info.get('error', 'Unknown error')}</code>\n"
                  f"⚡ Impact: {error_info.get('impact', 'Unknown')}\n"
                  f"🔢 Error #: <code>{error_info.get('error_number', 0)}</code>")
        
        await self.send_notification(message, priority='critical', category='error', immediate=True)
    
    async def send_strategy_performance(self, strategy: str, win_rate: float,
                                      total_pnl: float, trades_count: int):
        """Send strategy performance notification"""
        performance_emoji = "🎯" if win_rate >= 0.6 else "🎲"
        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
        
        message = (f"{performance_emoji} <b>Strategy Performance</b>\n\n"
                  f"🎯 Strategy: <code>{strategy}</code>\n"
                  f"📊 Win Rate: <code>{win_rate:.1%}</code>\n"
                  f"{pnl_emoji} Total PnL: <code>${total_pnl:.2f}</code>\n"
                  f"🔢 Trades: <code>{trades_count}</code>")
        
        await self.send_notification(message, priority='low', category='performance')
    
    async def send_ml_update(self, model_name: str, accuracy: float,
                           improvement: float = None):
        """Send ML model update notification"""
        brain_emoji = "🧠" if accuracy >= 0.7 else "🤖"
        
        message = (f"{brain_emoji} <b>ML Model Update</b>\n\n"
                  f"🔬 Model: <code>{model_name}</code>\n"
                  f"🎯 Accuracy: <code>{accuracy:.1%}</code>")
        
        if improvement:
            trend_emoji = "📈" if improvement > 0 else "📉"
            message += f"\n{trend_emoji} Change: <code>{improvement:+.1%}</code>"
        
        await self.send_notification(message, priority='low', category='system')
    
    async def send_market_analysis(self, symbol: str, analysis_type: str,
                                 signal_strength: float, details: str):
        """Send market analysis notification"""
        strength_emojis = {
            'weak': '🤔',
            'moderate': '📊',
            'strong': '🔥'
        }
        
        if signal_strength >= 0.8:
            strength = 'strong'
        elif signal_strength >= 0.6:
            strength = 'moderate'
        else:
            strength = 'weak'
        
        strength_emoji = strength_emojis[strength]
        
        message = (f"{strength_emoji} <b>Market Analysis</b>\n\n"
                  f"📈 Symbol: <code>{symbol}</code>\n"
                  f"🔍 Type: <code>{analysis_type}</code>\n"
                  f"💪 Strength: <code>{signal_strength:.1%}</code>\n"
                  f"📋 Details: {details}")
        
        priority = 'medium' if signal_strength >= 0.7 else 'low'
        await self.send_notification(message, priority=priority, category='system')
    
    async def send_daily_summary(self, summary_data: Dict[str, Any]):
        """Send daily performance summary"""
        daily_pnl = summary_data.get('daily_pnl', 0)
        total_trades = summary_data.get('total_trades', 0)
        win_rate = summary_data.get('win_rate', 0)
        balance = summary_data.get('total_balance', 0)
        
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        
        message = (f"📊 <b>Daily Summary</b> {pnl_emoji}\n\n"
                  f"💰 Balance: <code>${balance:.2f}</code>\n"
                  f"📊 Daily PnL: <code>${daily_pnl:.2f}</code>\n"
                  f"🎯 Win Rate: <code>{win_rate:.1%}</code>\n"
                  f"🔢 Total Trades: <code>{total_trades}</code>\n"
                  f"📅 Date: <code>{datetime.utcnow().strftime('%Y-%m-%d')}</code>")
        
        await self.send_notification(message, priority='medium', category='performance')
    
    async def send_weekly_report(self, report_data: Dict[str, Any]):
        """Send weekly performance report"""
        weekly_return = report_data.get('weekly_return_pct', 0)
        best_strategy = report_data.get('best_strategy', 'N/A')
        total_trades = report_data.get('total_trades', 0)
        
        return_emoji = "🚀" if weekly_return >= 5 else "📈" if weekly_return >= 0 else "📉"
        
        message = (f"📈 <b>Weekly Report</b> {return_emoji}\n\n"
                  f"📊 Weekly Return: <code>{weekly_return:+.2f}%</code>\n"
                  f"🏆 Best Strategy: <code>{best_strategy}</code>\n"
                  f"🔢 Total Trades: <code>{total_trades}</code>\n"
                  f"📅 Week: <code>{datetime.utcnow().strftime('W%W %Y')}</code>")
        
        # Add performance chart if available
        chart_path = report_data.get('chart_path')
        if chart_path and self.telegram_notifier:
            await self.telegram_notifier.send_photo(chart_path, message)
        else:
            await self.send_notification(message, priority='medium', category='performance')
    
    async def send_maintenance_notice(self, maintenance_type: str, 
                                    estimated_duration: str, details: str = ""):
        """Send maintenance notification"""
        message = (f"🔧 <b>Maintenance Notice</b>\n\n"
                  f"⚙️ Type: <code>{maintenance_type}</code>\n"
                  f"⏱️ Duration: <code>{estimated_duration}</code>\n"
                  f"📋 Details: {details}\n\n"
                  f"🤖 Bot may be temporarily unavailable")
        
        await self.send_notification(message, priority='high', category='system', immediate=True)
    
    # Configuration methods
    def update_notification_settings(self, settings: Dict[str, bool]):
        """Update notification settings"""
        self.notification_settings.update(settings)
        logger.info(f"📱 Notification settings updated: {settings}")
    
    def set_rate_limits(self, hourly_limit: int, daily_limit: int):
        """Update rate limits"""
        self.rate_limits['hourly'] = hourly_limit
        self.rate_limits['daily'] = daily_limit
        logger.info(f"📱 Rate limits updated: {hourly_limit}/hour, {daily_limit}/day")
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        return {
            'enabled': self.enabled,
            'message_counts': self.message_counts.copy(),
            'rate_limits': self.rate_limits.copy(),
            'queue_length': len(self.message_queue),
            'settings': self.notification_settings.copy(),
            'telegram_connected': self.telegram_notifier is not None
        }
    
    async def test_notifications(self):
        """Send test notifications"""
        test_messages = [
            ("🧪 Test notification - Low priority", 'low', 'system'),
            ("🧪 Test notification - Medium priority", 'medium', 'system'),
            ("🧪 Test notification - High priority", 'high', 'system'),
        ]
        
        for message, priority, category in test_messages:
            await self.send_notification(message, priority=priority, category=category)
            await asyncio.sleep(1)
        
        logger.info("📱 Test notifications sent")
    
    async def cleanup(self):
        """Cleanup notification manager"""
        try:
            # Process remaining queued messages
            if self.message_queue:
                logger.info(f"📱 Processing {len(self.message_queue)} remaining messages")
                
                # Send summary of remaining messages
                summary = f"📱 <b>Shutdown Summary</b>\n\n{len(self.message_queue)} pending notifications processed"
                await self.send_immediate(NotificationMessage(
                    message=summary,
                    priority='medium',
                    category='system',
                    timestamp=datetime.utcnow()
                ))
            
            # Close connections
            if self.telegram_notifier:
                await self.telegram_notifier.close()
            
            logger.info("📱 Notification manager cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Notification cleanup error: {e}")

# Email notification handler (optional)
class EmailNotifier:
    """Email notification handler"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, 
                 password: str, recipient: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipient = recipient
    
    async def send_email(self, subject: str, body: str, is_html: bool = False):
        """Send email notification"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = self.recipient
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html' if is_html else 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Email send failed: {e}")
            return False

# Webhook notification handler
class WebhookNotifier:
    """Webhook notification handler for custom integrations"""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self):
        """Initialize webhook connection"""
        self.session = aiohttp.ClientSession()
    
    async def send_webhook(self, data: Dict[str, Any]) -> bool:
        """Send webhook notification"""
        try:
            async with self.session.post(
                self.webhook_url, 
                json=data, 
                headers=self.headers
            ) as response:
                return response.status < 400
                
        except Exception as e:
            logger.error(f"❌ Webhook send failed: {e}")
            return False
    
    async def close(self):
        """Close webhook session"""
        if self.session:
            await self.session.close()

# Export main classes
__all__ = [
    'NotificationManager',
    'NotificationMessage',
    'TelegramNotifier',
    'EmailNotifier',
    'WebhookNotifier'
]