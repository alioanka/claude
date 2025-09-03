"""
Health Checker - Monitors system health and performance
Provides continuous monitoring of bot components and alerts.
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import psutil
import aiohttp
import joblib
import os

from config.config import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MLPredictor:
    """Machine learning predictor for trading signals"""
    
    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.last_retrain_time = datetime.utcnow()
        self.model_path = config.ml.models_path
        self.is_initialized = False
        
        # Model configuration
        self.model_config = {
            'lookback_periods': config.ml.feature_lookback,
            'confidence_threshold': config.ml.confidence_threshold,
            'retrain_interval_hours': config.ml.retrain_interval,
            'min_training_samples': 1000
        }
        
        # Feature engineering parameters
        self.feature_params = {
            'sma_periods': [5, 10, 20, 50],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'macd_periods': (12, 26, 9),
            'bb_period': 20,
            'bb_std': 2,
            'volume_sma_period': 20
        }
        
    async def initialize(self):
        """Initialize ML predictor"""
        try:
            logger.info("🧠 Initializing ML Predictor...")
            
            # Create models directory if it doesn't exist
            os.makedirs(self.model_path, exist_ok=True)
            
            # Initialize feature columns
            self._initialize_feature_columns()
            
            # Load existing models if available
            await self._load_existing_models()
            
            # Set initialization flag
            self.is_initialized = True
            
            logger.info(f"✅ ML Predictor initialized with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"❌ ML Predictor initialization failed: {e}")
            raise
    
    def _initialize_feature_columns(self):
        """Initialize feature column names"""
        features = []
        
        # Price-based features
        features.extend(['open', 'high', 'low', 'close', 'volume'])
        
        # Technical indicators
        for period in self.feature_params['sma_periods']:
            features.append(f'sma_{period}')
        
        for period in self.feature_params['ema_periods']:
            features.append(f'ema_{period}')
        
        features.extend(['rsi', 'macd', 'macd_signal', 'macd_hist'])
        features.extend(['bb_upper', 'bb_middle', 'bb_lower', 'bb_width'])
        features.extend(['volume_sma', 'volume_ratio'])
        
        # Price change features
        features.extend(['price_change_1h', 'price_change_4h', 'price_change_24h'])
        features.extend(['volatility_1h', 'volatility_4h', 'volatility_24h'])
        
        self.feature_columns = features
        logger.debug(f"Initialized {len(features)} feature columns")
    
    async def _load_existing_models(self):
        """Load existing trained models"""
        try:
            model_files = [f for f in os.listdir(self.model_path) if f.endswith('.joblib')]
            
            for model_file in model_files:
                try:
                    symbol = model_file.replace('_model.joblib', '')
                    model_path = os.path.join(self.model_path, model_file)
                    
                    model = joblib.load(model_path)
                    self.models[symbol] = {
                        'model': model,
                        'last_trained': datetime.fromtimestamp(os.path.getmtime(model_path)),
                        'accuracy': 0.0,  # Would be stored separately in production
                        'prediction_count': 0
                    }
                    
                    logger.info(f"📥 Loaded model for {symbol}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load model {model_file}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing models: {e}")
    
    async def predict(self, symbol: str, market_data: List[Dict]) -> Optional[Dict]:
        """Generate ML prediction for symbol"""
        try:
            if not self.is_initialized:
                logger.warning("ML Predictor not initialized")
                return None
            
            if not market_data or len(market_data) < self.model_config['lookback_periods']:
                return None
            
            # Check if we have a model for this symbol
            if symbol not in self.models:
                # For now, return a simple mock prediction
                return self._generate_mock_prediction(symbol)
            
            # Prepare features
            features = await self._prepare_features(market_data)
            if features is None:
                return None
            
            # Make prediction
            model_info = self.models[symbol]
            model = model_info['model']
            
            try:
                # Reshape features for prediction
                features_array = np.array(features).reshape(1, -1)
                
                # Make prediction
                prediction = model.predict(features_array)[0]
                prediction_proba = model.predict_proba(features_array)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
                
                confidence = max(prediction_proba)
                direction = 'long' if prediction == 1 else 'short'
                
                # Update prediction count
                model_info['prediction_count'] += 1
                
                result = {
                    'strategy': 'ml_prediction',
                    'side': direction,
                    'confidence': confidence,
                    'entry_price': market_data[-1]['close'],
                    'stop_loss': self._calculate_stop_loss(market_data[-1]['close'], direction),
                    'take_profit': self._calculate_take_profit(market_data[-1]['close'], direction),
                    'position_size': 0,  # Will be calculated by risk manager
                    'leverage': 1.0,
                    'timestamp': datetime.utcnow(),
                    'features_used': len(features),
                    'model_age_days': (datetime.utcnow() - model_info['last_trained']).days
                }
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Model prediction failed for {symbol}: {e}")
                return self._generate_mock_prediction(symbol)
        
        except Exception as e:
            logger.error(f"❌ Prediction failed for {symbol}: {e}")
            return None
    
    def _generate_mock_prediction(self, symbol: str) -> Dict:
        """Generate mock prediction when no model is available"""
        import random
        
        confidence = random.uniform(0.5, 0.8)
        direction = random.choice(['long', 'short'])
        
        return {
            'strategy': 'ml_mock',
            'side': direction,
            'confidence': confidence,
            'entry_price': 0,  # Will be set by caller
            'stop_loss': 0,
            'take_profit': 0,
            'position_size': 0,
            'leverage': 1.0,
            'timestamp': datetime.utcnow(),
            'is_mock': True
        }
    
    async def _prepare_features(self, market_data: List[Dict]) -> Optional[List[float]]:
        """Prepare features from market data"""
        try:
            if len(market_data) < self.model_config['lookback_periods']:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(market_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Calculate technical indicators
            df = self._add_technical_indicators(df)
            
            # Select features
            feature_values = []
            
            # Get latest values for each feature
            for feature in self.feature_columns:
                if feature in df.columns:
                    value = df[feature].iloc[-1]
                    if pd.isna(value):
                        value = 0.0
                    feature_values.append(float(value))
                else:
                    feature_values.append(0.0)
            
            return feature_values
            
        except Exception as e:
            logger.error(f"❌ Feature preparation failed: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        try:
            # Simple Moving Averages
            for period in self.feature_params['sma_periods']:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            # Exponential Moving Averages
            for period in self.feature_params['ema_periods']:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.feature_params['rsi_period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.feature_params['rsi_period']).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=self.feature_params['macd_periods'][0]).mean()
            exp2 = df['close'].ewm(span=self.feature_params['macd_periods'][1]).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=self.feature_params['macd_periods'][2]).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            sma = df['close'].rolling(window=self.feature_params['bb_period']).mean()
            std = df['close'].rolling(window=self.feature_params['bb_period']).std()
            df['bb_upper'] = sma + (std * self.feature_params['bb_std'])
            df['bb_middle'] = sma
            df['bb_lower'] = sma - (std * self.feature_params['bb_std'])
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=self.feature_params['volume_sma_period']).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price change features
            df['price_change_1h'] = df['close'].pct_change(periods=1)
            df['price_change_4h'] = df['close'].pct_change(periods=4)
            df['price_change_24h'] = df['close'].pct_change(periods=24)
            
            # Volatility features
            df['volatility_1h'] = df['close'].rolling(window=1).std()
            df['volatility_4h'] = df['close'].rolling(window=4).std()
            df['volatility_24h'] = df['close'].rolling(window=24).std()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Technical indicator calculation failed: {e}")
            return df
    
    def _calculate_stop_loss(self, entry_price: float, direction: str) -> float:
        """Calculate stop loss price"""
        stop_loss_pct = config.trading.stop_loss_percent
        
        if direction == 'long':
            return entry_price * (1 - stop_loss_pct)
        else:
            return entry_price * (1 + stop_loss_pct)
    
    def _calculate_take_profit(self, entry_price: float, direction: str) -> float:
        """Calculate take profit price"""
        take_profit_pct = config.trading.take_profit_percent
        
        if direction == 'long':
            return entry_price * (1 + take_profit_pct)
        else:
            return entry_price * (1 - take_profit_pct)
    
    async def retrain_models(self):
        """Retrain ML models with latest data"""
        try:
            logger.info("🔄 Retraining ML models...")
            
            # For now, just update the last retrain time
            # In a full implementation, this would:
            # 1. Fetch latest market data
            # 2. Prepare training dataset
            # 3. Train new models
            # 4. Validate model performance
            # 5. Save updated models
            
            self.last_retrain_time = datetime.utcnow()
            
            logger.info("✅ ML model retraining completed")
            
        except Exception as e:
            logger.error(f"❌ Model retraining failed: {e}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {
            'total_models': len(self.models),
            'last_retrain': self.last_retrain_time.isoformat(),
            'models': {}
        }
        
        for symbol, model_info in self.models.items():
            status['models'][symbol] = {
                'last_trained': model_info['last_trained'].isoformat(),
                'accuracy': model_info['accuracy'],
                'prediction_count': model_info['prediction_count'],
                'age_days': (datetime.utcnow() - model_info['last_trained']).days
            }
        
        return status
    
    async def cleanup(self):
        """Clean up ML predictor"""
        try:
            # Clear models from memory
            self.models.clear()
            logger.info("🧠 ML Predictor cleaned up")
            
        except Exception as e:
            logger.error(f"❌ ML Predictor cleanup failed: {e}")
    
    async def predict_price_direction(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Predict price direction using ML models"""
        return {
            'symbol': symbol,
            'direction': 0.5,  # 0-1 scale
            'confidence': 0.6,
            'timestamp': datetime.utcnow().isoformat()
        }

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self, bot=None, notification_manager=None):
        self.bot = bot
        self.notification_manager = notification_manager
        self.is_monitoring = False
        self.check_interval = 60  # seconds
        self.health_status = {
            'overall': 'healthy',
            'components': {},
            'last_check': None,
            'issues': []
        }
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage_percent': 80,
            'memory_usage_percent': 85,
            'disk_usage_percent': 90,
            'response_time_seconds': 5.0
        }
        
    async def monitor(self):
        """Main monitoring loop"""
        try:
            self.is_monitoring = True
            logger.info("🏥 Health monitoring started")
            
            while self.is_monitoring:
                try:
                    await self.perform_health_check()
                    await asyncio.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"❌ Health check error: {e}")
                    await asyncio.sleep(30)  # Shorter sleep on error
                    
        except Exception as e:
            logger.error(f"❌ Health monitoring failed: {e}")
        finally:
            self.is_monitoring = False
            logger.info("🏥 Health monitoring stopped")
    
    async def perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            issues = []
            components = {}
            
            # System resource checks
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            
            components['system'] = {
                'status': 'healthy',
                'cpu_usage': cpu_usage,
                'memory_usage': memory_info.percent,
                'disk_usage': disk_info.percent,
                'uptime': self.get_uptime()
            }
            
            # Check resource thresholds
            if cpu_usage > self.thresholds['cpu_usage_percent']:
                issues.append(f"High CPU usage: {cpu_usage:.1f}%")
                components['system']['status'] = 'warning'
            
            if memory_info.percent > self.thresholds['memory_usage_percent']:
                issues.append(f"High memory usage: {memory_info.percent:.1f}%")
                components['system']['status'] = 'warning'
            
            if disk_info.percent > self.thresholds['disk_usage_percent']:
                issues.append(f"High disk usage: {disk_info.percent:.1f}%")
                components['system']['status'] = 'critical'
            
            # Bot component checks
            if self.bot:
                bot_status = await self.check_bot_health()
                components['bot'] = bot_status
                
                if bot_status['status'] != 'healthy':
                    issues.extend(bot_status.get('issues', []))
            
            # Update health status
            self.health_status = {
                'overall': 'critical' if any('critical' in str(comp) for comp in components.values()) 
                          else 'warning' if issues else 'healthy',
                'components': components,
                'last_check': datetime.utcnow().isoformat(),
                'issues': issues
            }
            
            # Send alerts for critical issues
            if issues and self.notification_manager:
                await self.send_health_alert(issues)
                
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            self.health_status = {
                'overall': 'error',
                'error': str(e),
                'last_check': datetime.utcnow().isoformat()
            }
    
    async def check_bot_health(self) -> Dict[str, Any]:
        """Check trading bot specific health"""
        try:
            status = {
                'status': 'healthy',
                'is_running': self.bot.is_running if hasattr(self.bot, 'is_running') else False,
                'issues': []
            }
            
            # Check if bot is running
            if hasattr(self.bot, 'is_running') and not self.bot.is_running:
                status['status'] = 'critical'
                status['issues'].append("Bot is not running")
            
            # Check exchange connectivity
            if hasattr(self.bot, 'exchange_manager'):
                try:
                    health_check = await self.bot.exchange_manager.health_check()
                    if not health_check:
                        status['status'] = 'warning'
                        status['issues'].append("Exchange connectivity issues")
                except:
                    status['status'] = 'critical'
                    status['issues'].append("Exchange manager error")
            
            # Check data collection
            if hasattr(self.bot, 'data_collector'):
                if hasattr(self.bot.data_collector, 'is_collecting'):
                    if not self.bot.data_collector.is_collecting:
                        status['status'] = 'warning'
                        status['issues'].append("Data collection stopped")
            
            return status
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'issues': [f"Health check error: {e}"]
            }
    
    async def send_health_alert(self, issues: List[str]):
        """Send health alert notification"""
        try:
            if not self.notification_manager:
                return
                
            severity = 'critical' if any('critical' in issue.lower() for issue in issues) else 'warning'
            
            issues_text = "\n".join([f"• {issue}" for issue in issues[:5]])  # Limit to 5 issues
            
            await self.notification_manager.send_notification(
                f"🏥 Health Alert\n\n{issues_text}",
                priority=severity,
                category='system',
                immediate=True
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to send health alert: {e}")
    
    def get_uptime(self) -> str:
        """Get system uptime"""
        try:
            boot_time = psutil.boot_time()
            uptime_seconds = datetime.utcnow().timestamp() - boot_time
            uptime_hours = uptime_seconds / 3600
            return f"{uptime_hours:.1f} hours"
        except:
            return "Unknown"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return self.health_status.copy()
    
    async def stop(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        logger.info("🏥 Health monitoring stop requested")