"""
Advanced Visualization Engine for Emerging Topics Analysis
Provides enhanced interactive visualizations with statistical significance indicators
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

class AdvancedVisualizationEngine:
    """
    Advanced visualization engine for emerging topics analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        # Color schemes
        self.urgency_colors = self.config.get('urgency_colors', {
            'Critical': '#d32f2f',
            'High': '#f57c00',
            'Medium': '#fbc02d',
            'Low': '#388e3c'
        })
        
        self.significance_colors = self.config.get('significance_colors', {
            'significant': '#2e7d32',
            'not_significant': '#757575'
        })
        
        # Chart dimensions
        self.chart_dimensions = self.config.get('chart_dimensions', {
            'trend_heatmap': {'width': 800, 'height': 600},
            'scatter_plot': {'width': 700, 'height': 500},
            'timeline': {'width': 900, 'height': 400}
        })
    
    def create_trend_heatmap(self, emerging_topics: Dict[str, Any]) -> go.Figure:
        """
        Create interactive trend heatmap with statistical significance indicators
        """
        try:
            if not emerging_topics:
                return self._create_empty_chart("No emerging topics data available")
            
            # Prepare data for heatmap
            topics = []
            growth_rates = []
            sentiment_changes = []
            urgency_levels = []
            p_values = []
            significance_flags = []
            
            for topic_key, topic_data in emerging_topics.items():
                topics.append(topic_data.get('topic_name', topic_key))
                growth_rates.append(topic_data.get('growth_rate', 0))
                sentiment_changes.append(topic_data.get('sentiment_change', 0))
                urgency_levels.append(topic_data.get('regulatory_urgency', 'Low'))
                p_values.append(topic_data.get('p_value', 1.0))
                significance_flags.append('✓' if topic_data.get('is_significant', False) else '✗')
            
            # Create heatmap data matrix
            heatmap_data = np.array([growth_rates, sentiment_changes]).T
            
            # Create custom colorscale based on urgency
            urgency_numeric = [self._urgency_to_numeric(urgency) for urgency in urgency_levels]
            
            fig = go.Figure()
            
            # Add heatmap
            fig.add_trace(go.Heatmap(
                z=heatmap_data,
                x=['Growth Rate (%)', 'Sentiment Change'],
                y=topics,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Intensity"),
                hovertemplate=(
                    "<b>%{y}</b><br>" +
                    "Metric: %{x}<br>" +
                    "Value: %{z:.2f}<br>" +
                    "<extra></extra>"
                )
            ))
            
            # Add significance annotations
            for i, (topic, sig_flag, p_val) in enumerate(zip(topics, significance_flags, p_values)):
                fig.add_annotation(
                    x=1.1,
                    y=i,
                    text=f"{sig_flag} (p={p_val:.3f})",
                    showarrow=False,
                    font=dict(size=10, color='black'),
                    xref="x",
                    yref="y"
                )
            
            # Update layout
            fig.update_layout(
                title="Emerging Topics Trend Heatmap with Statistical Significance",
                width=self.chart_dimensions['trend_heatmap']['width'],
                height=self.chart_dimensions['trend_heatmap']['height'],
                xaxis_title="Metrics",
                yaxis_title="Topics",
                font=dict(size=12)
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_chart(f"Error creating heatmap: {str(e)}")
    
    def create_urgency_scatter_plot(self, emerging_topics: Dict[str, Any]) -> go.Figure:
        """
        Create scatter plot showing growth rate vs sentiment change with urgency coding
        """
        try:
            if not emerging_topics:
                return self._create_empty_chart("No emerging topics data available")
            
            # Prepare data
            topics = []
            growth_rates = []
            sentiment_changes = []
            urgency_levels = []
            recent_mentions = []
            confidence_intervals = []
            p_values = []
            
            for topic_key, topic_data in emerging_topics.items():
                topics.append(topic_data.get('topic_name', topic_key))
                growth_rates.append(topic_data.get('growth_rate', 0))
                sentiment_changes.append(topic_data.get('sentiment_change', 0))
                urgency_levels.append(topic_data.get('regulatory_urgency', 'Low'))
                recent_mentions.append(topic_data.get('recent_mentions', 0))
                confidence_intervals.append(topic_data.get('confidence_interval', [0, 0]))
                p_values.append(topic_data.get('p_value', 1.0))
            
            # Create scatter plot
            fig = go.Figure()
            
            # Group by urgency level
            for urgency in ['Critical', 'High', 'Medium', 'Low']:
                mask = [u == urgency for u in urgency_levels]
                if any(mask):
                    x_vals = [growth_rates[i] for i, m in enumerate(mask) if m]
                    y_vals = [sentiment_changes[i] for i, m in enumerate(mask) if m]
                    topic_names = [topics[i] for i, m in enumerate(mask) if m]
                    mentions = [recent_mentions[i] for i, m in enumerate(mask) if m]
                    p_vals = [p_values[i] for i, m in enumerate(mask) if m]
                    
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='markers',
                        name=f'{urgency} Urgency',
                        marker=dict(
                            color=self.urgency_colors.get(urgency, '#888888'),
                            size=[max(8, min(20, m/2)) for m in mentions],
                            opacity=0.7,
                            line=dict(width=2, color='white')
                        ),
                        text=topic_names,
                        hovertemplate=(
                            "<b>%{text}</b><br>" +
                            "Growth Rate: %{x:.1f}%<br>" +
                            "Sentiment Change: %{y:.3f}<br>" +
                            f"Urgency: {urgency}<br>" +
                            "Recent Mentions: %{marker.size}<br>" +
                            "<extra></extra>"
                        )
                    ))
            
            # Add quadrant lines
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Update layout
            fig.update_layout(
                title="Regulatory Urgency Analysis: Growth Rate vs Sentiment Change",
                xaxis_title="Growth Rate (%)",
                yaxis_title="Sentiment Change",
                width=self.chart_dimensions['scatter_plot']['width'],
                height=self.chart_dimensions['scatter_plot']['height'],
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                font=dict(size=12)
            )
            
            # Add quadrant annotations
            fig.add_annotation(x=50, y=0.1, text="High Growth<br>Positive Sentiment", 
                             showarrow=False, font=dict(size=10, color="green"))
            fig.add_annotation(x=50, y=-0.1, text="High Growth<br>Negative Sentiment", 
                             showarrow=False, font=dict(size=10, color="orange"))
            fig.add_annotation(x=-25, y=0.1, text="Declining<br>Positive Sentiment", 
                             showarrow=False, font=dict(size=10, color="blue"))
            fig.add_annotation(x=-25, y=-0.1, text="Declining<br>Negative Sentiment", 
                             showarrow=False, font=dict(size=10, color="red"))
            
            return fig
            
        except Exception as e:
            return self._create_empty_chart(f"Error creating scatter plot: {str(e)}")
    
    def create_confidence_interval_chart(self, emerging_topics: Dict[str, Any]) -> go.Figure:
        """
        Create confidence interval chart for growth rates
        """
        try:
            if not emerging_topics:
                return self._create_empty_chart("No emerging topics data available")
            
            # Prepare data
            topics = []
            growth_rates = []
            confidence_intervals = []
            significance_flags = []
            
            for topic_key, topic_data in emerging_topics.items():
                topics.append(topic_data.get('topic_name', topic_key))
                growth_rates.append(topic_data.get('growth_rate', 0))
                ci = topic_data.get('confidence_interval', [0, 0])
                confidence_intervals.append(ci)
                significance_flags.append(topic_data.get('is_significant', False))
            
            # Sort by growth rate
            sorted_indices = sorted(range(len(growth_rates)), key=lambda i: growth_rates[i], reverse=True)
            
            fig = go.Figure()
            
            # Add confidence intervals
            for i, idx in enumerate(sorted_indices):
                topic = topics[idx]
                growth = growth_rates[idx]
                ci = confidence_intervals[idx]
                is_sig = significance_flags[idx]
                
                color = self.significance_colors['significant'] if is_sig else self.significance_colors['not_significant']
                
                # Add error bar
                fig.add_trace(go.Scatter(
                    x=[ci[0], growth, ci[1]],
                    y=[i, i, i],
                    mode='lines+markers',
                    name=topic,
                    line=dict(color=color, width=3),
                    marker=dict(
                        color=[color, 'red' if is_sig else 'gray', color],
                        size=[6, 10, 6],
                        symbol=['line-ew', 'circle', 'line-ew']
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{topic}</b><br>" +
                        "Growth Rate: %{x:.1f}%<br>" +
                        f"Significant: {'Yes' if is_sig else 'No'}<br>" +
                        "<extra></extra>"
                    )
                ))
            
            # Update layout
            fig.update_layout(
                title="Growth Rate Confidence Intervals (95% CI)",
                xaxis_title="Growth Rate (%)",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(topics))),
                    ticktext=[topics[idx] for idx in sorted_indices],
                    title="Topics"
                ),
                width=self.chart_dimensions['timeline']['width'],
                height=max(400, len(topics) * 40),
                font=dict(size=12)
            )
            
            # Add significance legend
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(color=self.significance_colors['significant'], size=10),
                name='Statistically Significant',
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(color=self.significance_colors['not_significant'], size=10),
                name='Not Significant',
                showlegend=True
            ))
            
            return fig
            
        except Exception as e:
            return self._create_empty_chart(f"Error creating confidence interval chart: {str(e)}")
    
    def create_statistical_summary_table(self, emerging_topics: Dict[str, Any]) -> pd.DataFrame:
        """
        Create comprehensive statistical summary table
        """
        try:
            if not emerging_topics:
                return pd.DataFrame({'Message': ['No emerging topics data available']})
            
            summary_data = []
            
            for topic_key, topic_data in emerging_topics.items():
                summary_data.append({
                    'Topic': topic_data.get('topic_name', topic_key),
                    'Growth Rate (%)': f"{topic_data.get('growth_rate', 0):.1f}",
                    'Confidence Interval': f"[{topic_data.get('confidence_interval', [0, 0])[0]:.1f}, {topic_data.get('confidence_interval', [0, 0])[1]:.1f}]",
                    'P-Value': f"{topic_data.get('p_value', 1.0):.4f}",
                    'Significant': '✓' if topic_data.get('is_significant', False) else '✗',
                    'Sentiment Change': f"{topic_data.get('sentiment_change', 0):.3f}",
                    'Regulatory Urgency': topic_data.get('regulatory_urgency', 'Low'),
                    'Recent Mentions': topic_data.get('recent_mentions', 0),
                    'Historical Mentions': topic_data.get('historical_mentions', 0),
                    'Trend Classification': topic_data.get('trend_classification', 'Unknown')
                })
            
            df = pd.DataFrame(summary_data)
            
            # Sort by growth rate descending
            df = df.sort_values('Growth Rate (%)', key=lambda x: x.str.replace('%', '').astype(float), ascending=False)
            
            return df
            
        except Exception as e:
            return pd.DataFrame({'Error': [f"Error creating summary table: {str(e)}"]})
    
    def render_methodology_transparency(self, methodology: Dict[str, Any]) -> None:
        """
        Render methodology transparency information
        """
        try:
            st.markdown("### 📊 Statistical Methodology")
            
            # Statistical framework
            if 'statistical_tests' in methodology:
                st.markdown("**Statistical Tests Applied:**")
                for test in methodology['statistical_tests']:
                    st.markdown(f"• {test}")
            
            # Significance testing
            st.markdown("**Significance Testing:**")
            st.markdown(f"• Significance threshold: α = {methodology.get('significance_threshold', 0.05)}")
            st.markdown(f"• Confidence level: {methodology.get('confidence_level', 0.95) * 100}%")
            
            # Growth thresholds
            if 'growth_thresholds' in methodology:
                st.markdown("**Growth Rate Classification:**")
                thresholds = methodology['growth_thresholds']
                st.markdown(f"• Emerging trend: ≥ {thresholds.get('emerging', 50)}%")
                st.markdown(f"• Rapid growth: ≥ {thresholds.get('rapid', 100)}%")
                st.markdown(f"• Explosive growth: ≥ {thresholds.get('explosive', 250)}%")
            
            # Time periods
            if 'time_periods' in methodology:
                st.markdown("**Analysis Periods:**")
                periods = methodology['time_periods']
                st.markdown(f"• Recent period: {periods.get('recent_period', '2 quarters')}")
                st.markdown(f"• Baseline period: {periods.get('baseline_period', '4 quarters')}")
            
        except Exception as e:
            st.error(f"Error rendering methodology: {str(e)}")
    
    def _urgency_to_numeric(self, urgency: str) -> int:
        """
        Convert urgency level to numeric value
        """
        urgency_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        return urgency_map.get(urgency, 1)
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """
        Create empty chart with message
        """
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=16),
            xref="paper",
            yref="paper"
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=600,
            height=400
        )
        return fig
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Return default configuration
        """
        return {
            'urgency_colors': {
                'Critical': '#d32f2f',
                'High': '#f57c00',
                'Medium': '#fbc02d',
                'Low': '#388e3c'
            },
            'significance_colors': {
                'significant': '#2e7d32',
                'not_significant': '#757575'
            },
            'chart_dimensions': {
                'trend_heatmap': {'width': 800, 'height': 600},
                'scatter_plot': {'width': 700, 'height': 500},
                'timeline': {'width': 900, 'height': 400}
            }
        }