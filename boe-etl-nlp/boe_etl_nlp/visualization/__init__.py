#!/usr/bin/env python3
"""
Visualization Module
===================

This module contains visualization components for the BoE ETL NLP extension,
including dashboards and charts for NLP analysis results.

Components:
- dashboard: Interactive dashboards for NLP analysis results
"""

from .dashboard import NLPDashboard, create_dashboard

__all__ = [
    "NLPDashboard",
    "create_dashboard",
]