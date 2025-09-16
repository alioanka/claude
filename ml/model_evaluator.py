"""
Model evaluation module for assessing trading model performance.
Provides comprehensive evaluation metrics and visualization tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

from utils.helper import PerformanceCalculator, FileManager

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.performance_calc = PerformanceCalculator()
        self.file_manager = FileManager()
        
    def evaluate_regression_model(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Optional timestamps for time-series analysis
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating regression model")
        
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        residuals = y_true - y_pred
        
        # Directional accuracy (for trading)
        y_true_direction = np.sign(y_true)
        y_pred_direction = np.sign(y_pred)
        directional_accuracy = np.mean(y_true_direction == y_pred_direction)
        
        # Statistical tests
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        
        # Percentage within tolerance bands
        tolerance_1pct = np.mean(np.abs(residuals) <= 0.01)
        tolerance_5pct = np.mean(np.abs(residuals) <= 0.05)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'tolerance_1pct': tolerance_1pct,
            'tolerance_5pct': tolerance_5pct,
            'n_samples': len(y_true)
        }
        
        # Time-series specific metrics if timestamps provided
        if timestamps is not None:
            ts_metrics = self._calculate_time_series_metrics(y_true, y_pred, timestamps)
            metrics.update(ts_metrics)
        
        return metrics
    
    def evaluate_classification_model(self,
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating classification model")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'n_samples': len(y_true)
        }
        
        # ROC curve and AUC (for binary classification)
        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            metrics.update({
                'roc_auc': roc_auc,
                'fpr': fpr,
                'tpr': tpr
            })
        
        return metrics
    
    def _calculate_time_series_metrics(self,
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     timestamps: np.ndarray) -> Dict[str, float]:
        """Calculate time-series specific metrics."""
        # Convert to pandas for easier manipulation
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'y_true': y_true,
            'y_pred': y_pred
        })
        df.set_index('timestamp', inplace=True)
        
        # Calculate rolling metrics
        window_size = min(30, len(df) // 4)  # Adaptive window size
        df['residuals'] = df['y_true'] - df['y_pred']
        df['rolling_mse'] = df['residuals'].rolling(window=window_size).apply(lambda x: np.mean(x**2))
        df['rolling_mae'] = df['residuals'].rolling(window=window_size).apply(lambda x: np.mean(np.abs(x)))
        
        # Stability metrics
        mse_stability = 1 - (df['rolling_mse'].std() / df['rolling_mse'].mean()) if df['rolling_mse'].mean() > 0 else 0
        mae_stability = 1 - (df['rolling_mae'].std() / df['rolling_mae'].mean()) if df['rolling_mae'].mean() > 0 else 0
        
        # Trend following ability
        true_trend = np.sign(df['y_true'].diff())
        pred_trend = np.sign(df['y_pred'].diff())
        trend_accuracy = np.mean(true_trend == pred_trend)
        
        return {
            'mse_stability': mse_stability,
            'mae_stability': mae_stability,
            'trend_accuracy': trend_accuracy,
            'avg_rolling_mse': df['rolling_mse'].mean(),
            'avg_rolling_mae': df['rolling_mae'].mean()
        }
    
    def backtest_model_predictions(self,
                                 predictions: np.ndarray,
                                 actual_prices: np.ndarray,
                                 initial_capital: float = 100000,
                                 transaction_cost: float = 0.001) -> Dict[str, Any]:
        """
        Backtest trading performance based on model predictions.
        
        Args:
            predictions: Model predictions (price changes or signals)
            actual_prices: Actual price series
            initial_capital: Initial trading capital
            transaction_cost: Transaction cost as fraction
            
        Returns:
            Dictionary of trading performance metrics
        """
        logger.info("Backtesting model predictions")
        
        # Convert predictions to trading signals
        signals = np.sign(predictions)  # 1 for buy, -1 for sell, 0 for hold
        
        # Calculate returns
        price_returns = np.diff(actual_prices) / actual_prices[:-1]
        
        # Align signals with returns (signals predict next period)
        if len(signals) > len(price_returns):
            signals = signals[:-1]
        elif len(signals) < len(price_returns):
            price_returns = price_returns[:len(signals)]
        
        # Calculate strategy returns
        strategy_returns = signals * price_returns - np.abs(np.diff(np.concatenate([[0], signals]))) * transaction_cost
        
        # Calculate cumulative performance
        cumulative_returns = np.cumprod(1 + strategy_returns)
        final_value = initial_capital * cumulative_returns[-1]
        
        # Performance metrics
        total_return = (final_value - initial_capital) / initial_capital
        sharpe_ratio = self.performance_calc.calculate_sharpe_ratio(pd.Series(strategy_returns))
        max_drawdown = self.performance_calc.calculate_max_drawdown(pd.Series(strategy_returns))
        volatility = self.performance_calc.calculate_volatility(pd.Series(strategy_returns))
        win_rate = self.performance_calc.calculate_win_rate(pd.Series(strategy_returns))
        
        # Trading statistics
        n_trades = np.sum(np.abs(np.diff(np.concatenate([[0], signals]))) > 0)
        avg_trade_return = np.mean(strategy_returns[strategy_returns != 0]) if np.any(strategy_returns != 0) else 0
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'n_trades': n_trades,
            'avg_trade_return': avg_trade_return,
            'strategy_returns': strategy_returns,
            'cumulative_returns': cumulative_returns
        }
    
    def generate_evaluation_report(self,
                                 model_results: Dict[str, Any],
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 model_name: str = "Model") -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model_results: Model evaluation results
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Formatted evaluation report
        """
        report_lines = [
            f"\n{'='*60}",
            f"MODEL EVALUATION REPORT: {model_name}",
            f"{'='*60}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Number of samples: {len(y_true)}",
            ""
        ]
        
        # Regression metrics
        if 'mse' in model_results:
            report_lines.extend([
                "REGRESSION METRICS:",
                "-" * 20,
                f"Mean Squared Error (MSE): {model_results['mse']:.6f}",
                f"Root Mean Squared Error (RMSE): {model_results['rmse']:.6f}",
                f"Mean Absolute Error (MAE): {model_results['mae']:.6f}",
                f"R² Score: {model_results['r2']:.4f}",
                f"Mean Absolute Percentage Error: {model_results.get('mape', 0):.2f}%",
                f"Directional Accuracy: {model_results.get('directional_accuracy', 0):.4f}",
                ""
            ])
        
        # Classification metrics
        if 'accuracy' in model_results:
            report_lines.extend([
                "CLASSIFICATION METRICS:",
                "-" * 25,
                f"Accuracy: {model_results['accuracy']:.4f}",
                f"Precision: {model_results['precision']:.4f}",
                f"Recall: {model_results['recall']:.4f}",
                f"F1 Score: {model_results['f1_score']:.4f}",
            ])
            
            if 'roc_auc' in model_results:
                report_lines.append(f"ROC AUC: {model_results['roc_auc']:.4f}")
            
            report_lines.append("")
        
        # Time series metrics
        if 'trend_accuracy' in model_results:
            report_lines.extend([
                "TIME SERIES METRICS:",
                "-" * 22,
                f"Trend Accuracy: {model_results['trend_accuracy']:.4f}",
                f"MSE Stability: {model_results['mse_stability']:.4f}",
                f"MAE Stability: {model_results['mae_stability']:.4f}",
                ""
            ])
        
        # Residual analysis (for regression)
        if 'residual_mean' in model_results:
            report_lines.extend([
                "RESIDUAL ANALYSIS:",
                "-" * 18,
                f"Residual Mean: {model_results['residual_mean']:.6f}",
                f"Residual Std: {model_results['residual_std']:.6f}",
                f"Within 1% tolerance: {model_results.get('tolerance_1pct', 0):.2%}",
                f"Within 5% tolerance: {model_results.get('tolerance_5pct', 0):.2%}",
                ""
            ])
        
        return "\n".join(report_lines)
    
    def plot_regression_diagnostics(self,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   timestamps: Optional[np.ndarray] = None,
                                   save_path: str = None) -> plt.Figure:
        """
        Create diagnostic plots for regression models.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Optional timestamps
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Regression Model Diagnostics', fontsize=16)
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted')
        
        # Add R² to the plot
        r2 = r2_score(y_true, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Residuals vs Predicted
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted')
        
        # 3. Histogram of Residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        
        # 4. Time series plot (if timestamps available) or Q-Q plot
        if timestamps is not None:
            axes[1, 1].plot(timestamps, y_true, label='Actual', alpha=0.7)
            axes[1, 1].plot(timestamps, y_pred, label='Predicted', alpha=0.7)
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Values')
            axes[1, 1].set_title('Time Series Comparison')
            axes[1, 1].legend()
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            # Q-Q plot for normality check
            from scipy import stats as scipy_stats
            scipy_stats.probplot(residuals, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot (Normality Check)')
        
        plt.tight_layout()
        
        if save_path:
            self.file_manager.ensure_directory_exists(str(Path(save_path).parent))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Regression diagnostics saved to {save_path}")
        
        return fig
    
    def plot_classification_diagnostics(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      y_pred_proba: Optional[np.ndarray] = None,
                                      class_names: Optional[List[str]] = None,
                                      save_path: str = None) -> plt.Figure:
        """
        Create diagnostic plots for classification models.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            class_names: Names of classes
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_plots = 3 if y_pred_proba is not None and len(np.unique(y_true)) == 2 else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        fig.suptitle('Classification Model Diagnostics', fontsize=16)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        axes[0].set_title('Confusion Matrix')
        
        if class_names:
            axes[0].set_xticklabels(class_names)
            axes[0].set_yticklabels(class_names)
        
        # 2. Classification Report Heatmap
        if len(np.unique(y_true)) <= 10:  # Only for reasonable number of classes
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).iloc[:-1, :].T  # Remove 'accuracy' row
            sns.heatmap(report_df.iloc[:-2], annot=True, cmap='RdYlBu', ax=axes[1])
            axes[1].set_title('Classification Report')
        else:
            # For many classes, show class-wise accuracy
            class_acc = []
            for cls in np.unique(y_true):
                mask = y_true == cls
                acc = accuracy_score(y_true[mask], y_pred[mask])
                class_acc.append(acc)
            
            axes[1].bar(range(len(class_acc)), class_acc)
            axes[1].set_xlabel('Class')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Per-Class Accuracy')
        
        # 3. ROC Curve (for binary classification with probabilities)
        if n_plots == 3:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            axes[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[2].set_xlim([0.0, 1.0])
            axes[2].set_ylim([0.0, 1.05])
            axes[2].set_xlabel('False Positive Rate')
            axes[2].set_ylabel('True Positive Rate')
            axes[2].set_title('ROC Curve')
            axes[2].legend(loc="lower right")
        
        plt.tight_layout()
        
        if save_path:
            self.file_manager.ensure_directory_exists(str(Path(save_path).parent))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Classification diagnostics saved to {save_path}")
        
        return fig
    
    def plot_trading_performance(self,
                               backtest_results: Dict[str, Any],
                               benchmark_returns: Optional[np.ndarray] = None,
                               save_path: str = None) -> plt.Figure:
        """
        Plot trading performance from backtest results.
        
        Args:
            backtest_results: Results from backtest_model_predictions
            benchmark_returns: Optional benchmark returns for comparison
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Trading Performance Analysis', fontsize=16)
        
        strategy_returns = backtest_results['strategy_returns']
        cumulative_returns = backtest_results['cumulative_returns']
        
        # 1. Cumulative Returns
        axes[0, 0].plot(cumulative_returns, label='Strategy', linewidth=2)
        if benchmark_returns is not None:
            benchmark_cumulative = np.cumprod(1 + benchmark_returns)
            axes[0, 0].plot(benchmark_cumulative, label='Benchmark', linewidth=2, alpha=0.7)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Rolling Sharpe Ratio
        window = min(30, len(strategy_returns) // 4)
        if window > 5:
            rolling_sharpe = pd.Series(strategy_returns).rolling(window).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
            )
            axes[0, 1].plot(rolling_sharpe)
            axes[0, 1].set_title(f'Rolling Sharpe Ratio ({window}-period)')
            axes[0, 1].set_ylabel('Sharpe Ratio')
            axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Return Distribution
        axes[1, 0].hist(strategy_returns, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].axvline(x=np.mean(strategy_returns), color='g', linestyle='--', alpha=0.7, label='Mean')
        axes[1, 0].set_title('Return Distribution')
        axes[1, 0].set_xlabel('Returns')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 4. Drawdown
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        axes[1, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[1, 1].plot(drawdown, color='red', linewidth=1)
        axes[1, 1].set_title('Drawdown')
        axes[1, 1].set_ylabel('Drawdown')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self.file_manager.ensure_directory_exists(str(Path(save_path).parent))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trading performance plot saved to {save_path}")
        
        return fig
    
    def feature_importance_analysis(self,
                                  model,
                                  feature_names: List[str],
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  top_k: int = 20) -> Dict[str, Any]:
        """
        Analyze feature importance for model interpretability.
        
        Args:
            model: Trained model
            feature_names: Names of features
            X_test: Test features
            y_test: Test targets
            top_k: Number of top features to analyze
            
        Returns:
            Dictionary containing importance analysis
        """
        logger.info("Analyzing feature importance")
        
        importance_dict = {}
        
        # Get built-in feature importance if available
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
            importance_dict['builtin'] = dict(zip(feature_names, importance_scores))
        
        # Permutation importance
        try:
            from sklearn.inspection import permutation_importance
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
            importance_dict['permutation'] = dict(zip(feature_names, perm_importance.importances_mean))
        except Exception as e:
            logger.warning(f"Could not calculate permutation importance: {e}")
        
        # Create summary
        if importance_dict:
            # Use the first available importance method
            primary_importance = list(importance_dict.values())[0]
            sorted_features = sorted(primary_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            return {
                'importance_scores': importance_dict,
                'top_features': sorted_features[:top_k],
                'feature_ranking': [name for name, _ in sorted_features]
            }
        else:
            return {'importance_scores': {}, 'top_features': [], 'feature_ranking': []}
    
    def model_comparison(self,
                        results_list: List[Dict[str, Any]],
                        model_names: Optional[List[str]] = None,
                        metric: str = 'rmse') -> pd.DataFrame:
        """
        Compare multiple model results.
        
        Args:
            results_list: List of model evaluation results
            model_names: Names of models (optional)
            metric: Primary metric for comparison
            
        Returns:
            Comparison DataFrame
        """
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(results_list))]
        
        comparison_data = []
        
        for i, results in enumerate(results_list):
            row = {'model_name': model_names[i]}
            
            # Extract common metrics
            for key in ['mse', 'rmse', 'mae', 'r2', 'accuracy', 'f1_score', 'directional_accuracy']:
                if key in results:
                    row[key] = results[key]
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric
        if metric in df.columns:
            ascending = metric not in ['r2', 'accuracy', 'f1_score', 'directional_accuracy']
            df = df.sort_values(metric, ascending=ascending)
        
        return df
    
    def generate_model_card(self,
                           model_info: Dict[str, Any],
                           evaluation_results: Dict[str, Any],
                           training_data_info: Dict[str, Any]) -> str:
        """
        Generate a model card with comprehensive information.
        
        Args:
            model_info: Information about the model
            evaluation_results: Evaluation results
            training_data_info: Information about training data
            
        Returns:
            Formatted model card
        """
        card_lines = [
            "# MODEL CARD",
            "=" * 50,
            "",
            "## Model Information",
            f"**Model Type:** {model_info.get('model_type', 'Unknown')}",
            f"**Task Type:** {model_info.get('task_type', 'Unknown')}",
            f"**Training Date:** {datetime.now().strftime('%Y-%m-%d')}",
            f"**Model ID:** {model_info.get('model_id', 'N/A')}",
            "",
            "## Training Data",
            f"**Number of Features:** {training_data_info.get('n_features', 'N/A')}",
            f"**Training Samples:** {training_data_info.get('n_train_samples', 'N/A')}",
            f"**Validation Samples:** {training_data_info.get('n_val_samples', 'N/A')}",
            f"**Data Period:** {training_data_info.get('date_range', 'N/A')}",
            "",
            "## Performance Metrics",
        ]
        
        # Add performance metrics
        if 'rmse' in evaluation_results:
            card_lines.extend([
                "**Regression Metrics:**",
                f"- RMSE: {evaluation_results['rmse']:.4f}",
                f"- MAE: {evaluation_results['mae']:.4f}",
                f"- R²: {evaluation_results['r2']:.4f}",
                f"- Directional Accuracy: {evaluation_results.get('directional_accuracy', 0):.4f}",
            ])
        
        if 'accuracy' in evaluation_results:
            card_lines.extend([
                "**Classification Metrics:**",
                f"- Accuracy: {evaluation_results['accuracy']:.4f}",
                f"- F1 Score: {evaluation_results['f1_score']:.4f}",
                f"- Precision: {evaluation_results['precision']:.4f}",
                f"- Recall: {evaluation_results['recall']:.4f}",
            ])
        
        card_lines.extend([
            "",
            "## Model Limitations",
            "- Performance may vary with market conditions",
            "- Requires regular retraining with new data",
            "- Should be used in conjunction with risk management",
            "",
            "## Recommended Use",
            "- Suitable for automated trading systems",
            "- Requires continuous monitoring",
            "- Best used with proper position sizing",
            "",
            f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ])
        
        return "\n".join(card_lines)

# Convenience functions
def evaluate_trading_model(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          timestamps: Optional[np.ndarray] = None,
                          task_type: str = 'regression') -> Dict[str, Any]:
    """
    Convenience function to evaluate a trading model.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        timestamps: Optional timestamps
        task_type: 'regression' or 'classification'
        
    Returns:
        Evaluation results
    """
    evaluator = ModelEvaluator()
    
    if task_type == 'regression':
        return evaluator.evaluate_regression_model(y_true, y_pred, timestamps)
    else:
        return evaluator.evaluate_classification_model(y_true, y_pred)

def plot_model_diagnostics(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          task_type: str = 'regression',
                          timestamps: Optional[np.ndarray] = None,
                          save_path: str = None) -> plt.Figure:
    """
    Convenience function to plot model diagnostics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task_type: 'regression' or 'classification'
        timestamps: Optional timestamps
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    evaluator = ModelEvaluator()
    
    if task_type == 'regression':
        return evaluator.plot_regression_diagnostics(y_true, y_pred, timestamps, save_path)
    else:
        return evaluator.plot_classification_diagnostics(y_true, y_pred, save_path=save_path)