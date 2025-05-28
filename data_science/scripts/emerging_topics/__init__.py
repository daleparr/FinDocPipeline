"""
Advanced Emerging Topics & Trend Analysis Module

This module provides real-time trend detection with statistical significance testing,
dynamic baseline calculation, and enhanced visualizations for regulatory priority ranking.
"""

from .trend_detection_engine import EmergingTopicsEngine
from .statistical_significance import StatisticalSignificanceTester
from .advanced_visualizations import AdvancedVisualizationEngine

__version__ = "1.0.0"
__author__ = "BoE Data Science Team"

__all__ = [
    'EmergingTopicsEngine',
    'StatisticalSignificanceTester', 
    'AdvancedVisualizationEngine'
]