"""
Trading strategies package.
Provides various trading strategies and strategy base classes.
"""

from .mean_reversion import MeanReversionStrategy

__all__ = [
    'MeanReversionStrategy'
]