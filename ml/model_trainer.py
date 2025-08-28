"""
Machine Learning model trainer for trading strategies.
Supports various ML algorithms and training pipelines.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import pickle
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Deep learning imports (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from utils.helper import FileManager, ConfigManager
from ml.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Main model trainer class for various ML algorithms."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.models = {}
        self.scalers = {}
        self.feature_engineer = FeatureEngineer()
        self.file_manager = FileManager()
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'regressor': RandomForestRegressor,
                'classifier': RandomForestClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'regressor': GradientBoostingRegressor,
                'classifier': None,  # Would need GradientBoostingClassifier
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'linear': {
                'regressor': LinearRegression,
                'classifier': LogisticRegression,
                'params': {}
            },
            'ridge': {
                'regressor': Ridge,
                'classifier': LogisticRegression,
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'svm': {
                'regressor': SVR,
                'classifier': SVC,
                'params': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear']
                }
            }
        }
        
        if XGBOOST_AVAILABLE:
            self.model_configs['xgboost'] = {
                'regressor': xgb.XGBRegressor,
                'classifier': xgb.XGBClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
    
    def prepare_data(self, 
                     data: Dict[str, pd.DataFrame],
                     target_type: str = 'regression',
                     prediction_horizon: int = 1,
                     features: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            target_type: 'regression' or 'classification'
            prediction_horizon: Number of periods to predict ahead
            features: List of feature column names
            
        Returns:
            Tuple of (X, y) arrays
        """
        logger.info("Preparing data for training")
        
        # Combine all symbol data
        all_data = []
        for symbol, df in data.items():
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            all_data.append(df_copy)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp' if 'timestamp' in combined_df.columns else combined_df.index)
        
        # Engineer features if not provided
        if features is None:
            combined_df = self.feature_engineer.engineer_features(combined_df)
            features = self.feature_engineer.get_feature_columns(combined_df)
        
        # Create target variable
        if target_type == 'regression':
            # Predict future price change
            combined_df['target'] = combined_df.groupby('symbol')['close'].shift(-prediction_horizon)
            combined_df['target'] = (combined_df['target'] - combined_df['close']) / combined_df['close']
        else:
            # Predict future price direction
            combined_df['future_close'] = combined_df.groupby('symbol')['close'].shift(-prediction_horizon)
            combined_df['target'] = (combined_df['future_close'] > combined_df['close']).astype(int)
        
        # Remove rows with NaN values
        combined_df = combined_df.dropna()
        
        if combined_df.empty:
            raise ValueError("No valid data remaining after preparation")
        
        # Extract features and target
        X = combined_df[features].values
        y = combined_df['target'].values
        
        logger.info(f"Prepared data: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def train_model(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    model_type: str = 'random_forest',
                    task_type: str = 'regression',
                    validation_split: float = 0.2,
                    use_grid_search: bool = True,
                    cv_folds: int = 3) -> Dict[str, Any]:
        """
        Train a machine learning model.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to train
            task_type: 'regression' or 'classification'
            validation_split: Fraction of data for validation
            use_grid_search: Whether to use grid search for hyperparameters
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing trained model and metrics
        """
        logger.info(f"Training {model_type} model for {task_type}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Get model class
        if model_type not in self.model_configs:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        config = self.model_configs[model_type]
        if task_type == 'regression':
            model_class = config['regressor']
        else:
            model_class = config['classifier']
            
        if model_class is None:
            raise ValueError(f"Model type {model_type} not supported for {task_type}")
        
        # Train model
        if use_grid_search and config['params']:
            # Use grid search for hyperparameter tuning
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            grid_search = GridSearchCV(
                model_class(),
                config['params'],
                cv=tscv,
                scoring='neg_mean_squared_error' if task_type == 'regression' else 'accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # Use default parameters
            model = model_class()
            model.fit(X_train_scaled, y_train)
            best_params = {}
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        
        # Calculate metrics
        if task_type == 'regression':
            train_metrics = {
                'mse': mean_squared_error(y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'mae': mean_absolute_error(y_train, y_train_pred)
            }
            val_metrics = {
                'mse': mean_squared_error(y_val, y_val_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'mae': mean_absolute_error(y_val, y_val_pred)
            }
        else:
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_train_pred)
            }
            val_metrics = {
                'accuracy': accuracy_score(y_val, y_val_pred)
            }
        
        # Store model and scaler
        model_id = f"{model_type}_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.models[model_id] = model
        self.scalers[model_id] = scaler
        
        results = {
            'model_id': model_id,
            'model': model,
            'scaler': scaler,
            'model_type': model_type,
            'task_type': task_type,
            'best_params': best_params,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_importance': self._get_feature_importance(model),
            'training_data_shape': X_train.shape,
            'validation_data_shape': X_val.shape
        }
        
        logger.info(f"Model training completed. Validation metrics: {val_metrics}")
        return results
    
    def train_lstm_model(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        sequence_length: int = 60,
                        validation_split: float = 0.2,
                        epochs: int = 50,
                        batch_size: int = 32) -> Dict[str, Any]:
        """
        Train LSTM model for time series prediction.
        
        Args:
            X: Feature matrix
            y: Target vector
            sequence_length: Length of input sequences
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Dictionary containing trained model and metrics
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        logger.info("Training LSTM model")
        
        # Prepare sequences
        X_seq, y_seq = self._create_sequences(X, y, sequence_length)
        
        # Split data
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Scale features
        scaler = MinMaxScaler()
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features))
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        
        X_val_scaled = scaler.transform(X_val.reshape(-1, n_features))
        X_val_scaled = X_val_scaled.reshape(X_val.shape[0], n_timesteps, n_features)
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(n_timesteps, n_features)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val_scaled, y_val),
            callbacks=[early_stop],
            verbose=1
        )
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        
        # Calculate metrics
        train_metrics = {
            'mse': mean_squared_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred)
        }
        val_metrics = {
            'mse': mean_squared_error(y_val, y_val_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'mae': mean_absolute_error(y_val, y_val_pred)
        }
        
        # Store model and scaler
        model_id = f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.models[model_id] = model
        self.scalers[model_id] = scaler
        
        results = {
            'model_id': model_id,
            'model': model,
            'scaler': scaler,
            'model_type': 'lstm',
            'task_type': 'regression',
            'sequence_length': sequence_length,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'history': history.history,
            'training_data_shape': X_train_scaled.shape,
            'validation_data_shape': X_val_scaled.shape
        }
        
        logger.info(f"LSTM training completed. Validation RMSE: {val_metrics['rmse']:.4f}")
        return results
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _get_feature_importance(self, model) -> Optional[Dict[str, float]]:
        """Get feature importance from model if available."""
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(enumerate(model.feature_importances_))
            elif hasattr(model, 'coef_'):
                return dict(enumerate(model.coef_.flatten()))
            else:
                return None
        except:
            return None
    
    def predict(self, model_id: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        scaler = self.scalers[model_id]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        predictions = model.predict(X_scaled)
        
        return predictions
    
    def save_model(self, model_id: str, filepath: str):
        """Save trained model to file."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        scaler = self.scalers[model_id]
        
        # Create directory if it doesn't exist
        self.file_manager.ensure_directory_exists(str(Path(filepath).parent))
        
        # Save based on model type
        if isinstance(model, tf.keras.Model) if TENSORFLOW_AVAILABLE else False:
            # Save Keras model
            model.save(f"{filepath}_model.h5")
            joblib.dump(scaler, f"{filepath}_scaler.pkl")
        else:
            # Save sklearn model
            model_data = {
                'model': model,
                'scaler': scaler,
                'model_id': model_id
            }
            joblib.dump(model_data, f"{filepath}.pkl")
        
        logger.info(f"Model {model_id} saved to {filepath}")
    
    def load_model(self, filepath: str, model_id: str = None) -> str:
        """Load trained model from file."""
        try:
            if filepath.endswith('.h5') or Path(f"{filepath}_model.h5").exists():
                # Load Keras model
                if not TENSORFLOW_AVAILABLE:
                    raise ImportError("TensorFlow not available")
                
                model = tf.keras.models.load_model(f"{filepath}_model.h5")
                scaler = joblib.load(f"{filepath}_scaler.pkl")
                model_id = model_id or f"loaded_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                # Load sklearn model
                model_data = joblib.load(f"{filepath}.pkl")
                model = model_data['model']
                scaler = model_data['scaler']
                model_id = model_id or model_data.get('model_id', f"loaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Store in memory
            self.models[model_id] = model
            self.scalers[model_id] = scaler
            
            logger.info(f"Model loaded as {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {e}")
            raise
    
    def evaluate_model(self, model_id: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        # Make predictions
        y_pred = self.predict(model_id, X_test)
        
        # Calculate metrics
        if len(np.unique(y_test)) == 2:  # Classification
            metrics = {
                'accuracy': accuracy_score(y_test, (y_pred > 0.5).astype(int)),
                'precision': accuracy_score(y_test, (y_pred > 0.5).astype(int)),  # Simplified
                'recall': accuracy_score(y_test, (y_pred > 0.5).astype(int))      # Simplified
            }
        else:  # Regression
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
            }
        
        return metrics
    
    def cross_validate_model(self, 
                           X: np.ndarray, 
                           y: np.ndarray,
                           model_type: str = 'random_forest',
                           task_type: str = 'regression',
                           cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation on model."""
        logger.info(f"Cross-validating {model_type} model")
        
        # Get model class
        config = self.model_configs[model_type]
        if task_type == 'regression':
            model_class = config['regressor']
        else:
            model_class = config['classifier']
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        scores = []
        feature_importances = []
        
        scaler = StandardScaler()
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Processing fold {fold + 1}/{cv_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale data
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model = model_class()
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_val_scaled)
            
            # Calculate score
            if task_type == 'regression':
                score = mean_squared_error(y_val, y_pred)
            else:
                score = accuracy_score(y_val, y_pred)
            
            scores.append(score)
            
            # Store feature importance if available
            importance = self._get_feature_importance(model)
            if importance:
                feature_importances.append(importance)
        
        # Aggregate results
        cv_results = {
            'cv_scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'feature_importances': feature_importances
        }
        
        logger.info(f"Cross-validation completed. Mean score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        return cv_results
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a trained model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        info = {
            'model_id': model_id,
            'model_type': type(model).__name__,
            'n_features': None,
            'is_keras_model': isinstance(model, tf.keras.Model) if TENSORFLOW_AVAILABLE else False
        }
        
        # Get number of features
        if hasattr(model, 'n_features_in_'):
            info['n_features'] = model.n_features_in_
        elif hasattr(model, 'input_shape') and TENSORFLOW_AVAILABLE:
            info['n_features'] = model.input_shape[-1]
        
        return info
    
    def list_models(self) -> List[str]:
        """List all trained models."""
        return list(self.models.keys())
    
    def remove_model(self, model_id: str):
        """Remove model from memory."""
        if model_id in self.models:
            del self.models[model_id]
        if model_id in self.scalers:
            del self.scalers[model_id]
        logger.info(f"Removed model {model_id}")

class AutoMLTrainer:
    """Automated machine learning trainer that tries multiple models."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.trainer = ModelTrainer(config)
        self.results = []
    
    def auto_train(self, 
                   X: np.ndarray,
                   y: np.ndarray,
                   task_type: str = 'regression',
                   models_to_try: List[str] = None,
                   use_grid_search: bool = True) -> Dict[str, Any]:
        """
        Automatically train multiple models and select the best one.
        
        Args:
            X: Feature matrix
            y: Target vector
            task_type: 'regression' or 'classification'
            models_to_try: List of model types to try
            use_grid_search: Whether to use grid search
            
        Returns:
            Dictionary with best model results
        """
        if models_to_try is None:
            models_to_try = ['random_forest', 'gradient_boosting', 'linear', 'ridge']
            if XGBOOST_AVAILABLE:
                models_to_try.append('xgboost')
        
        logger.info(f"Starting AutoML training with {len(models_to_try)} models")
        
        self.results = []
        
        for model_type in models_to_try:
            try:
                logger.info(f"Training {model_type}")
                result = self.trainer.train_model(
                    X, y,
                    model_type=model_type,
                    task_type=task_type,
                    use_grid_search=use_grid_search
                )
                self.results.append(result)
                
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                continue
        
        if not self.results:
            raise RuntimeError("No models were successfully trained")
        
        # Select best model based on validation performance
        if task_type == 'regression':
            best_result = min(self.results, key=lambda x: x['val_metrics']['rmse'])
            logger.info(f"Best model: {best_result['model_type']} (RMSE: {best_result['val_metrics']['rmse']:.4f})")
        else:
            best_result = max(self.results, key=lambda x: x['val_metrics']['accuracy'])
            logger.info(f"Best model: {best_result['model_type']} (Accuracy: {best_result['val_metrics']['accuracy']:.4f})")
        
        return {
            'best_model': best_result,
            'all_results': self.results,
            'leaderboard': self._create_leaderboard(task_type)
        }
    
    def _create_leaderboard(self, task_type: str) -> pd.DataFrame:
        """Create leaderboard of model performance."""
        leaderboard_data = []
        
        for result in self.results:
            row = {
                'model_id': result['model_id'],
                'model_type': result['model_type'],
                'best_params': str(result['best_params'])
            }
            
            # Add metrics
            for metric, value in result['val_metrics'].items():
                row[f'val_{metric}'] = value
            
            leaderboard_data.append(row)
        
        df = pd.DataFrame(leaderboard_data)
        
        # Sort by primary metric
        if task_type == 'regression':
            df = df.sort_values('val_rmse')
        else:
            df = df.sort_values('val_accuracy', ascending=False)
        
        return df

# Convenience functions
def train_trading_model(data: Dict[str, pd.DataFrame],
                       model_type: str = 'random_forest',
                       task_type: str = 'regression',
                       prediction_horizon: int = 1,
                       config: Dict = None) -> Dict[str, Any]:
    """
    Convenience function to train a trading model.
    
    Args:
        data: Dictionary of symbol -> DataFrame
        model_type: Type of model to train
        task_type: 'regression' or 'classification'
        prediction_horizon: Number of periods to predict ahead
        config: Configuration dictionary
        
    Returns:
        Training results dictionary
    """
    trainer = ModelTrainer(config)
    
    # Prepare data
    X, y = trainer.prepare_data(data, task_type, prediction_horizon)
    
    # Train model
    return trainer.train_model(X, y, model_type, task_type)

def auto_train_trading_models(data: Dict[str, pd.DataFrame],
                             task_type: str = 'regression',
                             prediction_horizon: int = 1,
                             config: Dict = None) -> Dict[str, Any]:
    """
    Convenience function for automated model training.
    
    Args:
        data: Dictionary of symbol -> DataFrame
        task_type: 'regression' or 'classification'
        prediction_horizon: Number of periods to predict ahead
        config: Configuration dictionary
        
    Returns:
        AutoML results dictionary
    """
    auto_trainer = AutoMLTrainer(config)
    trainer = auto_trainer.trainer
    
    # Prepare data
    X, y = trainer.prepare_data(data, task_type, prediction_horizon)
    
    # Auto train
    return auto_trainer.auto_train(X, y, task_type)