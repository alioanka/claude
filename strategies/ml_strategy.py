"""
Machine Learning trading strategy using LSTM, XGBoost, and ensemble methods.
"""

import numpy as np
import pandas as pd
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import joblib
import warnings
warnings.filterwarnings('ignore')

from .base_strategy import BaseStrategy
from utils.indicators import TechnicalIndicators

# ML Libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available, LSTM models disabled")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available, gradient boosting models disabled")

try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available, ML models disabled")

logger = logging.getLogger(__name__)

class MLStrategy(BaseStrategy):
    """
    Machine Learning strategy using multiple algorithms:
    1. LSTM for time series prediction
    2. XGBoost for feature-based classification
    3. Ensemble methods for robust predictions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("ml_strategy", config)
        self.indicators = TechnicalIndicators()
        
        # Model configuration
        self.lstm_config = config.get('lstm_config', {
            'sequence_length': 60,
            'hidden_units': [128, 64],
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        })
        
        self.xgb_config = config.get('xgb_config', {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8
        })
        
        # Models
        self.lstm_model = None
        self.xgb_model = None
        self.ensemble_model = None
        self.scaler = None
        self.label_encoder = None
        
        # Feature engineering
        self.feature_columns = []
        self.sequence_data = {}
        
        # Model paths
        self.model_dir = config.get('model_dir', 'storage/models')
        
        # Load existing models
        asyncio.create_task(self._load_models())
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate ML-based trading signal"""
        try:
            if len(data) < self.lstm_config['sequence_length']:
                return None
            
            # Feature engineering
            features_df = await self._engineer_features(data)
            if features_df.empty:
                return None
            
            # Get predictions from all models
            predictions = {}
            
            if TENSORFLOW_AVAILABLE and self.lstm_model:
                lstm_pred = await self._predict_lstm(features_df)
                predictions['lstm'] = lstm_pred
            
            if XGBOOST_AVAILABLE and self.xgb_model:
                xgb_pred = await self._predict_xgboost(features_df)
                predictions['xgboost'] = xgb_pred
            
            if SKLEARN_AVAILABLE and self.ensemble_model:
                ensemble_pred = await self._predict_ensemble(features_df)
                predictions['ensemble'] = ensemble_pred
            
            if not predictions:
                return None
            
            # Combine predictions
            final_prediction = await self._combine_predictions(predictions)
            
            # Convert to trading signal
            signal = self._prediction_to_signal(final_prediction, symbol)
            
            return signal
            
        except Exception as e:
            logger.error(f"ML signal generation failed for {symbol}: {e}")
            return None
    
    async def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML models"""
        try:
            df = data.copy()
            
            # Price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['price_change'] = df['close'] - df['open']
            df['price_range'] = df['high'] - df['low']
            
            # Technical indicators
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            bb_window = 20
            df['bb_middle'] = df['close'].rolling(bb_window).mean()
            bb_std = df['close'].rolling(bb_window).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_price_trend'] = df['volume'] * df['returns']
            
            # Volatility
            df['volatility'] = df['returns'].rolling(20).std()
            df['atr'] = self._calculate_atr(df)
            
            # Momentum features
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # Support/Resistance levels
            df['resistance'] = df['high'].rolling(20).max()
            df['support'] = df['low'].rolling(20).min()
            df['support_resistance_ratio'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
            
            # Market structure
            df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
            df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
            df['trend_strength'] = df['higher_high'] + df['higher_low']
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
            
            # Store feature columns for later use
            self.feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            # Drop rows with NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return pd.DataFrame()
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr
        except:
            return pd.Series(0, index=df.index)
    
    async def _predict_lstm(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate LSTM prediction"""
        try:
            if not TENSORFLOW_AVAILABLE or not self.lstm_model:
                return {'direction': 0.5, 'confidence': 0.0}
            
            # Prepare sequence data
            sequence = self._prepare_lstm_sequence(data)
            if sequence is None:
                return {'direction': 0.5, 'confidence': 0.0}
            
            # Make prediction
            prediction = self.lstm_model.predict(sequence, verbose=0)
            
            # Convert to direction and confidence
            direction_prob = float(prediction[0][0])
            confidence = abs(direction_prob - 0.5) * 2  # Convert to 0-1 scale
            
            return {
                'direction': direction_prob,
                'confidence': confidence,
                'raw_prediction': prediction.tolist()
            }
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return {'direction': 0.5, 'confidence': 0.0}
    
    def _prepare_lstm_sequence(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare sequence data for LSTM"""
        try:
            sequence_length = self.lstm_config['sequence_length']
            if len(data) < sequence_length:
                return None
            
            # Use only price and volume features for LSTM
            features = ['close', 'volume', 'rsi', 'macd', 'bb_position']
            available_features = [f for f in features if f in data.columns]
            
            if not available_features:
                return None
            
            # Get last sequence_length rows
            recent_data = data[available_features].tail(sequence_length).values
            
            # Normalize data
            if self.scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                self.scaler.fit(recent_data)
            
            normalized_data = self.scaler.transform(recent_data)
            
            # Reshape for LSTM (batch_size, sequence_length, features)
            return normalized_data.reshape(1, sequence_length, len(available_features))
            
        except Exception as e:
            logger.error(f"LSTM sequence preparation failed: {e}")
            return None
    
    async def _predict_xgboost(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate XGBoost prediction"""
        try:
            if not XGBOOST_AVAILABLE or not self.xgb_model:
                return {'direction': 0.5, 'confidence': 0.0}
            
            # Prepare features
            features = self._prepare_xgb_features(data)
            if features is None:
                return {'direction': 0.5, 'confidence': 0.0}
            
            # Make prediction
            prediction_proba = self.xgb_model.predict_proba(features)[0]
            
            # Assuming binary classification (0=sell, 1=buy)
            buy_prob = prediction_proba[1] if len(prediction_proba) > 1 else 0.5
            confidence = abs(buy_prob - 0.5) * 2
            
            return {
                'direction': buy_prob,
                'confidence': confidence,
                'raw_prediction': prediction_proba.tolist()
            }
            
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return {'direction': 0.5, 'confidence': 0.0}
    
    def _prepare_xgb_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for XGBoost"""
        try:
            if not self.feature_columns:
                return None
            
            # Get latest row with all features
            latest_data = data[self.feature_columns].tail(1)
            
            if latest_data.empty or latest_data.isnull().any().any():
                return None
            
            return latest_data.values
            
        except Exception as e:
            logger.error(f"XGBoost feature preparation failed: {e}")
            return None
    
    async def _predict_ensemble(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate ensemble prediction"""
        try:
            if not SKLEARN_AVAILABLE or not self.ensemble_model:
                return {'direction': 0.5, 'confidence': 0.0}
            
            features = self._prepare_xgb_features(data)  # Same features as XGBoost
            if features is None:
                return {'direction': 0.5, 'confidence': 0.0}
            
            prediction_proba = self.ensemble_model.predict_proba(features)[0]
            buy_prob = prediction_proba[1] if len(prediction_proba) > 1 else 0.5
            confidence = abs(buy_prob - 0.5) * 2
            
            return {
                'direction': buy_prob,
                'confidence': confidence,
                'raw_prediction': prediction_proba.tolist()
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return {'direction': 0.5, 'confidence': 0.0}
    
    async def _combine_predictions(self, predictions: Dict[str, Dict]) -> Dict[str, float]:
        """Combine predictions from multiple models"""
        try:
            if not predictions:
                return {'direction': 0.5, 'confidence': 0.0}
            
            # Weighted average based on model confidence
            total_weight = 0
            weighted_direction = 0
            
            model_weights = {
                'lstm': 0.4,
                'xgboost': 0.35,
                'ensemble': 0.25
            }
            
            for model_name, pred in predictions.items():
                weight = model_weights.get(model_name, 0.33) * pred['confidence']
                weighted_direction += pred['direction'] * weight
                total_weight += weight
            
            if total_weight == 0:
                return {'direction': 0.5, 'confidence': 0.0}
            
            final_direction = weighted_direction / total_weight
            final_confidence = min(total_weight / len(predictions), 1.0)
            
            return {
                'direction': final_direction,
                'confidence': final_confidence,
                'model_predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Prediction combination failed: {e}")
            return {'direction': 0.5, 'confidence': 0.0}
    
    def _prediction_to_signal(self, prediction: Dict[str, float], symbol: str) -> Dict[str, Any]:
        """Convert ML prediction to trading signal"""
        try:
            direction = prediction['direction']
            confidence = prediction['confidence']
            
            # Minimum confidence threshold
            min_confidence = self.config.get('min_confidence', 0.6)
            
            if confidence < min_confidence:
                return {
                    'signal_type': 'hold',
                    'symbol': symbol,
                    'confidence': confidence,
                    'metadata': prediction
                }
            
            # Convert direction to signal
            if direction > 0.55:  # Buy threshold
                signal_type = 'buy'
            elif direction < 0.45:  # Sell threshold
                signal_type = 'sell'
            else:
                signal_type = 'hold'
            
            return {
                'signal_type': signal_type,
                'symbol': symbol,
                'confidence': confidence,
                'metadata': {
                    'ml_prediction': prediction,
                    'direction_probability': direction,
                    'strategy': 'ml_strategy'
                },
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Signal conversion failed: {e}")
            return {
                'signal_type': 'hold',
                'symbol': symbol,
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
    
    async def train_models(self, training_data: Dict[str, pd.DataFrame], retrain: bool = False):
        """Train all ML models"""
        try:
            logger.info("Starting ML model training...")
            
            # Prepare training dataset
            X_train, y_train = await self._prepare_training_data(training_data)
            
            if X_train is None or len(X_train) == 0:
                logger.error("No training data available")
                return False
            
            # Train LSTM
            if TENSORFLOW_AVAILABLE:
                await self._train_lstm(X_train, y_train)
            
            # Train XGBoost
            if XGBOOST_AVAILABLE:
                await self._train_xgboost(X_train, y_train)
            
            # Train Ensemble
            if SKLEARN_AVAILABLE:
                await self._train_ensemble(X_train, y_train)
            
            # Save models
            await self._save_models()
            
            logger.info("ML model training completed")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    async def _prepare_training_data(self, data: Dict[str, pd.DataFrame]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data from historical data"""
        try:
            all_data = []
            
            for symbol, df in data.items():
                # Engineer features
                features_df = await self._engineer_features(df)
                if not features_df.empty:
                    all_data.append(features_df)
            
            if not all_data:
                return None, None
            
            # Combine all symbol data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Create labels (future price direction)
            combined_df['future_return'] = combined_df['close'].shift(-1) / combined_df['close'] - 1
            combined_df['label'] = (combined_df['future_return'] > 0.001).astype(int)  # 0.1% threshold
            
            # Remove rows with NaN
            combined_df = combined_df.dropna()
            
            if len(combined_df) < 100:
                logger.warning("Insufficient training data")
                return None, None
            
            # Prepare features and labels
            X = combined_df[self.feature_columns].values
            y = combined_df['label'].values
            
            # Normalize features
            if self.scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
            
            X_scaled = self.scaler.fit_transform(X)
            
            logger.info(f"Training data prepared: {len(X_scaled)} samples, {X_scaled.shape[1]} features")
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {e}")
            return None, None
    
    async def _train_lstm(self, X: np.ndarray, y: np.ndarray):
        """Train LSTM model"""
        try:
            if not TENSORFLOW_AVAILABLE:
                return
            
            logger.info("Training LSTM model...")
            
            # Prepare sequence data for LSTM
            sequence_length = self.lstm_config['sequence_length']
            X_sequences, y_sequences = self._create_sequences(X, y, sequence_length)
            
            if len(X_sequences) == 0:
                logger.warning("No sequence data for LSTM training")
                return
            
            # Build LSTM model
            model = Sequential()
            
            # First LSTM layer
            model.add(LSTM(
                self.lstm_config['hidden_units'][0],
                return_sequences=True,
                input_shape=(sequence_length, X.shape[1])
            ))
            model.add(Dropout(self.lstm_config['dropout_rate']))
            model.add(BatchNormalization())
            
            # Second LSTM layer
            model.add(LSTM(self.lstm_config['hidden_units'][1]))
            model.add(Dropout(self.lstm_config['dropout_rate']))
            model.add(BatchNormalization())
            
            # Output layer
            model.add(Dense(1, activation='sigmoid'))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.lstm_config['learning_rate']),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            X_train, X_val, y_train, y_val = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate model
            val_accuracy = max(history.history['val_accuracy'])
            logger.info(f"LSTM model trained. Validation accuracy: {val_accuracy:.3f}")
            
            self.lstm_model = model
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        try:
            X_sequences = []
            y_sequences = []
            
            for i in range(sequence_length, len(X)):
                X_sequences.append(X[i-sequence_length:i])
                y_sequences.append(y[i])
            
            return np.array(X_sequences), np.array(y_sequences)
            
        except Exception as e:
            logger.error(f"Sequence creation failed: {e}")
            return np.array([]), np.array([])
    
    async def _train_xgboost(self, X: np.ndarray, y: np.ndarray):
        """Train XGBoost model"""
        try:
            if not XGBOOST_AVAILABLE:
                return
            
            logger.info("Training XGBoost model...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train XGBoost
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=self.xgb_config['n_estimators'],
                max_depth=self.xgb_config['max_depth'],
                learning_rate=self.xgb_config['learning_rate'],
                subsample=self.xgb_config['subsample'],
                random_state=42,
                eval_metric='logloss'
            )
            
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Evaluate model
            test_accuracy = self.xgb_model.score(X_test, y_test)
            logger.info(f"XGBoost model trained. Test accuracy: {test_accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
    
    async def _train_ensemble(self, X: np.ndarray, y: np.ndarray):
        """Train ensemble model"""
        try:
            if not SKLEARN_AVAILABLE:
                return
            
            logger.info("Training ensemble model...")
            
            # Create base models
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            models = [('rf', rf_model)]
            
            # Add XGBoost to ensemble if available
            if XGBOOST_AVAILABLE:
                xgb_ensemble = xgb.XGBClassifier(n_estimators=50, random_state=42)
                models.append(('xgb', xgb_ensemble))
            
            # Create ensemble
            self.ensemble_model = VotingClassifier(estimators=models, voting='soft')
            
            # Train ensemble
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.ensemble_model.fit(X_train, y_train)
            
            # Evaluate
            test_accuracy = self.ensemble_model.score(X_test, y_test)
            logger.info(f"Ensemble model trained. Test accuracy: {test_accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
    
    async def _load_models(self):
        """Load pre-trained models"""
        try:
            import os
            
            # Load LSTM model
            if TENSORFLOW_AVAILABLE:
                lstm_path = f"{self.model_dir}/lstm_model.h5"
                if os.path.exists(lstm_path):
                    self.lstm_model = load_model(lstm_path)
                    logger.info("LSTM model loaded")
            
            # Load XGBoost model
            if XGBOOST_AVAILABLE:
                xgb_path = f"{self.model_dir}/xgb_model.pkl"
                if os.path.exists(xgb_path):
                    self.xgb_model = joblib.load(xgb_path)
                    logger.info("XGBoost model loaded")
            
            # Load ensemble model
            if SKLEARN_AVAILABLE:
                ensemble_path = f"{self.model_dir}/ensemble_model.pkl"
                if os.path.exists(ensemble_path):
                    self.ensemble_model = joblib.load(ensemble_path)
                    logger.info("Ensemble model loaded")
            
            # Load scaler
            scaler_path = f"{self.model_dir}/scaler.pkl"
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Feature scaler loaded")
                
        except Exception as e:
            logger.warning(f"Model loading failed: {e}")
    
    async def _save_models(self):
        """Save trained models"""
        try:
            import os
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Save LSTM model
            if TENSORFLOW_AVAILABLE and self.lstm_model:
                self.lstm_model.save(f"{self.model_dir}/lstm_model.h5")
                logger.info("LSTM model saved")
            
            # Save XGBoost model
            if XGBOOST_AVAILABLE and self.xgb_model:
                joblib.dump(self.xgb_model, f"{self.model_dir}/xgb_model.pkl")
                logger.info("XGBoost model saved")
            
            # Save ensemble model
            if SKLEARN_AVAILABLE and self.ensemble_model:
                joblib.dump(self.ensemble_model, f"{self.model_dir}/ensemble_model.pkl")
                logger.info("Ensemble model saved")
            
            # Save scaler
            if self.scaler:
                joblib.dump(self.scaler, f"{self.model_dir}/scaler.pkl")
                logger.info("Feature scaler saved")
                
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
    
    async def evaluate_models(self, test_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance"""
        try:
            X_test, y_test = await self._prepare_training_data(test_data)
            
            if X_test is None:
                return {}
            
            results = {}
            
            # Evaluate each model
            models = {
                'lstm': self.lstm_model,
                'xgboost': self.xgb_model,
                'ensemble': self.ensemble_model
            }
            
            for model_name, model in models.items():
                if model is None:
                    continue
                
                try:
                    if model_name == 'lstm':
                        # Prepare sequences for LSTM
                        X_seq, y_seq = self._create_sequences(X_test, y_test, self.lstm_config['sequence_length'])
                        if len(X_seq) > 0:
                            predictions = model.predict(X_seq, verbose=0)
                            y_pred = (predictions > 0.5).astype(int).flatten()
                            y_true = y_seq
                        else:
                            continue
                    else:
                        predictions = model.predict_proba(X_test)[:, 1]
                        y_pred = (predictions > 0.5).astype(int)
                        y_true = y_test
                    
                    # Calculate metrics
                    if SKLEARN_AVAILABLE:
                        accuracy = accuracy_score(y_true, y_pred)
                        precision = precision_score(y_true, y_pred, zero_division=0)
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        
                        results[model_name] = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1
                        }
                        
                        logger.info(f"{model_name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
                    
                except Exception as e:
                    logger.error(f"Evaluation failed for {model_name}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained models"""
        try:
            importance_dict = {}
            
            # XGBoost feature importance
            if self.xgb_model and hasattr(self.xgb_model, 'feature_importances_'):
                xgb_importance = self.xgb_model.feature_importances_
                for i, col in enumerate(self.feature_columns):
                    if i < len(xgb_importance):
                        importance_dict[f"xgb_{col}"] = float(xgb_importance[i])
            
            # Random Forest feature importance (from ensemble)
            if (self.ensemble_model and hasattr(self.ensemble_model, 'estimators_')):
                for name, estimator in self.ensemble_model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importance = estimator.feature_importances_
                        for i, col in enumerate(self.feature_columns):
                            if i < len(importance):
                                importance_dict[f"{name}_{col}"] = float(importance[i])
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return {}