#!/usr/bin/env python3
"""
NLP Dashboard Module
===================

This module provides visualization dashboards for NLP analysis results,
including topic modeling, sentiment analysis, and classification results.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import logging

# Optional imports for visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Visualization features will be limited.")

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logging.warning("Streamlit not available. Web dashboard features will be disabled.")


class NLPDashboard:
    """
    NLP analysis dashboard for visualizing topic modeling, sentiment, and classification results.
    
    This class provides methods to create interactive visualizations and dashboards
    for NLP analysis results from the BoE ETL pipeline.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the NLP dashboard.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def create_topic_distribution_chart(self, df: pd.DataFrame) -> Optional[Any]:
        """
        Create a topic distribution chart.
        
        Args:
            df: DataFrame with topic assignments
            
        Returns:
            Plotly figure if available, None otherwise
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available for visualization")
            return None
        
        if 'primary_topic' not in df.columns:
            self.logger.warning("No topic data found in DataFrame")
            return None
        
        topic_counts = df['primary_topic'].value_counts()
        
        fig = px.pie(
            values=topic_counts.values,
            names=topic_counts.index,
            title="Topic Distribution"
        )
        
        return fig
    
    def create_sentiment_analysis_chart(self, df: pd.DataFrame) -> Optional[Any]:
        """
        Create a sentiment analysis chart.
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            Plotly figure if available, None otherwise
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available for visualization")
            return None
        
        # This is a placeholder - would need actual sentiment data
        if 'data_type' in df.columns:
            sentiment_counts = df['data_type'].value_counts()
            
            fig = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title="Data Type Distribution (Actual vs Projection)"
            )
            
            return fig
        
        return None
    
    def create_financial_terms_wordcloud_data(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Create word frequency data for financial terms.
        
        Args:
            df: DataFrame with financial terms
            
        Returns:
            Dictionary with term frequencies
        """
        if 'all_financial_terms' not in df.columns:
            return {}
        
        # Collect all financial terms
        all_terms = []
        for terms_str in df['all_financial_terms'].fillna(''):
            if terms_str and terms_str != 'NONE':
                terms = terms_str.split('|')
                all_terms.extend(terms)
        
        # Count frequencies
        from collections import Counter
        term_freq = Counter(all_terms)
        
        return dict(term_freq.most_common(50))
    
    def create_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create summary statistics for the dashboard.
        
        Args:
            df: DataFrame with NLP analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        stats = {
            'total_records': len(df),
            'unique_speakers': df['speaker_norm'].nunique() if 'speaker_norm' in df.columns else 0,
            'avg_word_count': df['word_count'].mean() if 'word_count' in df.columns else 0,
            'financial_content_pct': (df['has_financial_terms'].sum() / len(df) * 100) if 'has_financial_terms' in df.columns else 0,
        }
        
        # Topic distribution
        if 'primary_topic' in df.columns:
            stats['topic_distribution'] = df['primary_topic'].value_counts().to_dict()
        
        # Data type distribution
        if 'data_type' in df.columns:
            stats['data_type_distribution'] = df['data_type'].value_counts().to_dict()
        
        return stats
    
    def render_streamlit_dashboard(self, df: pd.DataFrame) -> None:
        """
        Render a Streamlit dashboard for NLP analysis results.
        
        Args:
            df: DataFrame with NLP analysis results
        """
        if not STREAMLIT_AVAILABLE:
            self.logger.error("Streamlit not available for dashboard rendering")
            return
        
        st.title("📊 NLP Analysis Dashboard")
        st.write(f"Analyzing {len(df)} records")
        
        # Summary statistics
        stats = self.create_summary_stats(df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", stats['total_records'])
        with col2:
            st.metric("Unique Speakers", stats['unique_speakers'])
        with col3:
            st.metric("Avg Word Count", f"{stats['avg_word_count']:.1f}")
        with col4:
            st.metric("Financial Content %", f"{stats['financial_content_pct']:.1f}%")
        
        # Topic distribution chart
        if PLOTLY_AVAILABLE:
            topic_fig = self.create_topic_distribution_chart(df)
            if topic_fig:
                st.plotly_chart(topic_fig, use_container_width=True)
            
            sentiment_fig = self.create_sentiment_analysis_chart(df)
            if sentiment_fig:
                st.plotly_chart(sentiment_fig, use_container_width=True)
        
        # Financial terms analysis
        st.subheader("💰 Financial Terms Analysis")
        term_freq = self.create_financial_terms_wordcloud_data(df)
        if term_freq:
            # Display as a simple table since we don't have wordcloud
            term_df = pd.DataFrame(list(term_freq.items()), columns=['Term', 'Frequency'])
            st.dataframe(term_df.head(20))
        
        # Raw data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10))


def create_dashboard(df: pd.DataFrame, config: Optional[Dict] = None) -> NLPDashboard:
    """
    Create an NLP dashboard instance.
    
    Args:
        df: DataFrame with NLP analysis results
        config: Optional configuration dictionary
        
    Returns:
        NLPDashboard instance
    """
    dashboard = NLPDashboard(config)
    return dashboard