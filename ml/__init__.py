"""
Machine Learning package for trading bot.
Provides ML models, feature engineering, and training capabilities.
"""

from .feature_engineer import FeatureEngineer
from .model_trainer import (
    ModelTrainer,
    AutoMLTrainer,
    train_trading_model,
    auto_train_trading_models
)
from .model_evaluator import (
    ModelEvaluator,
    evaluate_trading_model,
    plot_model_diagnostics
)

__all__ = [
    'FeatureEngineer',
    'ModelTrainer',
    'AutoMLTrainer', 
    'train_trading_model',
    'auto_train_trading_models',
    'ModelEvaluator',
    'evaluate_trading_model',
    'plot_model_diagnostics'
]