"""
Quote Analysis Component for Dashboard Integration.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...emerging_topics import QuoteAnalyzer, AdvancedVisualizations


class QuoteAnalysisComponent:
    """
    Reusable component for quote analysis display in dashboards.
    """
    
    def __init__(self):
        """Initialize the quote analysis component."""
        self.quote_analyzer = QuoteAnalyzer()
        self.visualizer = AdvancedVisualizations()
    
    def render_quote_analysis_widget(
        self, 
        data: pd.DataFrame, 
        topic: str,
        max_quotes: int = 14,
        show_context: bool = True,
        show_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Render a complete quote analysis widget.
        
        Args:
            data: DataFrame with transcript data
            topic: Topic to analyze
            max_quotes: Maximum number of quotes to extract
            show_context: Whether to show quote context
            show_visualizations: Whether to show charts
            
        Returns:
            Dictionary with analysis results
        """
        st.subheader(f"💬 Quote Analysis: {topic}")
        
        # Extract quotes
        with st.spinner(f"Extracting quotes for {topic}..."):
            quotes = self.quote_analyzer.extract_topic_quotes(
                data, topic, max_quotes=max_quotes
            )
        
        if not quotes:
            st.warning(f"No quotes found for {topic}")
            return {'quotes': [], 'analysis': None}
        
        # Analyze contradictory sentiment
        contradiction_analysis = self.quote_analyzer.analyze_contradictory_sentiment(quotes)
        
        # Display summary metrics
        self._render_summary_metrics(quotes, contradiction_analysis)
        
        # Show visualizations if requested
        if show_visualizations:
            self._render_quote_visualizations(quotes)
        
        # Display individual quotes
        self._render_individual_quotes(quotes, show_context)
        
        return {
            'quotes': quotes,
            'analysis': contradiction_analysis,
            'topic': topic
        }
    
    def render_compact_quote_summary(
        self, 
        quotes: List[Dict[str, Any]], 
        topic: str
    ) -> None:
        """
        Render a compact summary of quote analysis.
        
        Args:
            quotes: List of analyzed quotes
            topic: Topic name
        """
        if not quotes:
            st.info(f"No quotes found for {topic}")
            return
        
        # Calculate summary statistics
        avg_transparency = sum(q.get('transparency_score', 100) for q in quotes) / len(quotes)
        avg_sentiment = sum(q.get('sentiment_compound', 0) for q in quotes) / len(quotes)
        
        hedging_count = sum(len(q.get('hedging_indicators', [])) for q in quotes)
        downplaying_count = sum(len(q.get('downplaying_patterns', [])) for q in quotes)
        
        # Display in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Quotes", len(quotes))
        
        with col2:
            color = "normal" if avg_transparency >= 70 else "inverse"
            st.metric("Avg Transparency", f"{avg_transparency:.1f}%", delta_color=color)
        
        with col3:
            sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
            st.metric("Avg Sentiment", sentiment_label)
        
        with col4:
            total_flags = hedging_count + downplaying_count
            flag_color = "inverse" if total_flags > len(quotes) else "normal"
            st.metric("Language Flags", total_flags, delta_color=flag_color)
    
    def _render_summary_metrics(
        self, 
        quotes: List[Dict[str, Any]], 
        contradiction_analysis: Dict[str, Any]
    ) -> None:
        """Render summary metrics for quote analysis."""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Quotes", len(quotes))
        
        with col2:
            avg_transparency = contradiction_analysis.get('average_transparency_score', 100)
            transparency_color = "normal" if avg_transparency >= 70 else "inverse"
            st.metric("Avg Transparency", f"{avg_transparency:.1f}%", delta_color=transparency_color)
        
        with col3:
            contradiction_detected = contradiction_analysis.get('contradiction_detected', False)
            status = "⚠️ Detected" if contradiction_detected else "✅ Clear"
            st.metric("Contradictions", status)
        
        with col4:
            total_hedging = contradiction_analysis.get('total_hedging_indicators', 0)
            hedging_color = "inverse" if total_hedging > len(quotes) * 0.5 else "normal"
            st.metric("Hedging Indicators", total_hedging, delta_color=hedging_color)
        
        # Show contradiction analysis
        if contradiction_detected:
            st.error("🚨 Contradictory Sentiment Analysis")
            st.write(contradiction_analysis.get('analysis', ''))
        else:
            st.success("✅ No Significant Contradictory Sentiment Detected")
    
    def _render_quote_visualizations(self, quotes: List[Dict[str, Any]]) -> None:
        """Render visualizations for quote analysis."""
        
        # Sentiment distribution
        sentiment_chart = self.visualizer.create_sentiment_distribution(quotes)
        st.plotly_chart(sentiment_chart, use_container_width=True)
        
        # Create transparency vs sentiment scatter plot
        transparency_scores = [q.get('transparency_score', 100) for q in quotes]
        sentiment_scores = [q.get('sentiment_compound', 0) for q in quotes]
        speakers = [q.get('speaker', 'Unknown') for q in quotes]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sentiment_scores,
            y=transparency_scores,
            mode='markers',
            marker=dict(
                size=10,
                color=transparency_scores,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Transparency Score")
            ),
            text=speakers,
            hovertemplate="<b>%{text}</b><br>" +
                         "Sentiment: %{x:.2f}<br>" +
                         "Transparency: %{y:.1f}%<br>" +
                         "<extra></extra>"
        ))
        
        fig.update_layout(
            title="Quote Analysis: Sentiment vs Transparency",
            xaxis_title="Sentiment Score",
            yaxis_title="Transparency Score (%)",
            height=400
        )
        
        # Add quadrant lines
        fig.add_hline(y=70, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_individual_quotes(
        self, 
        quotes: List[Dict[str, Any]], 
        show_context: bool = True
    ) -> None:
        """Render individual quote analysis."""
        
        st.subheader("📝 Individual Quote Analysis")
        
        # Sort quotes by transparency score (lowest first for attention)
        sorted_quotes = sorted(quotes, key=lambda x: x.get('transparency_score', 100))
        
        for i, quote in enumerate(sorted_quotes):
            transparency = quote.get('transparency_score', 100)
            
            # Color code based on transparency
            if transparency < 50:
                status = "🔴 Low Transparency"
                color = "red"
            elif transparency < 70:
                status = "🟡 Medium Transparency"
                color = "orange"
            else:
                status = "🟢 High Transparency"
                color = "green"
            
            with st.expander(f"Quote {i+1}: {quote.get('speaker', 'Unknown')} - {status}"):
                
                # Quote content
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Quote:**")
                    st.markdown(f'*"{quote.get("text", "")}"*')
                    
                    if show_context:
                        if quote.get('context_before'):
                            st.write("**Context Before:**")
                            st.caption(quote['context_before'])
                        
                        if quote.get('context_after'):
                            st.write("**Context After:**")
                            st.caption(quote['context_after'])
                
                with col2:
                    st.write("**Analysis Scores:**")
                    
                    # Transparency score with color
                    st.markdown(f"**Transparency:** <span style='color:{color}'>{transparency:.1f}%</span>", 
                               unsafe_allow_html=True)
                    
                    sentiment = quote.get('sentiment_compound', 0)
                    sentiment_color = "green" if sentiment > 0.1 else "red" if sentiment < -0.1 else "gray"
                    st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{sentiment:.2f}</span>", 
                               unsafe_allow_html=True)
                    
                    # Language pattern indicators
                    hedging = quote.get('hedging_indicators', [])
                    downplaying = quote.get('downplaying_patterns', [])
                    contradictions = quote.get('contradiction_flags', [])
                    
                    if hedging:
                        st.write("**Hedging:**")
                        st.caption(", ".join(hedging[:3]))
                    
                    if downplaying:
                        st.write("**Downplaying:**")
                        st.caption(", ".join(downplaying[:3]))
                    
                    if contradictions:
                        st.write("**Contradictions:**")
                        st.caption(", ".join(contradictions[:3]))
                    
                    # Financial terms
                    financial_terms = quote.get('financial_terms', [])
                    if financial_terms:
                        st.write("**Financial Terms:**")
                        st.caption(", ".join(financial_terms[:5]))
    
    def create_quote_comparison_chart(
        self, 
        quotes_by_topic: Dict[str, List[Dict[str, Any]]]
    ) -> go.Figure:
        """
        Create comparison chart across multiple topics.
        
        Args:
            quotes_by_topic: Dictionary mapping topic names to quote lists
            
        Returns:
            Plotly figure comparing topics
        """
        
        topics = list(quotes_by_topic.keys())
        quote_counts = [len(quotes) for quotes in quotes_by_topic.values()]
        
        avg_transparency = []
        avg_sentiment = []
        
        for quotes in quotes_by_topic.values():
            if quotes:
                transparency = sum(q.get('transparency_score', 100) for q in quotes) / len(quotes)
                sentiment = sum(q.get('sentiment_compound', 0) for q in quotes) / len(quotes)
            else:
                transparency = 100
                sentiment = 0
            
            avg_transparency.append(transparency)
            avg_sentiment.append(sentiment)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Quote Counts', 'Average Transparency', 
                          'Average Sentiment', 'Transparency Distribution'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        # Quote counts
        fig.add_trace(
            go.Bar(x=topics, y=quote_counts, name="Quote Count", marker_color='lightblue'),
            row=1, col=1
        )
        
        # Average transparency
        colors = ['red' if t < 70 else 'orange' if t < 85 else 'green' for t in avg_transparency]
        fig.add_trace(
            go.Bar(x=topics, y=avg_transparency, name="Transparency", marker_color=colors),
            row=1, col=2
        )
        
        # Average sentiment
        sentiment_colors = ['red' if s < -0.1 else 'green' if s > 0.1 else 'gray' for s in avg_sentiment]
        fig.add_trace(
            go.Bar(x=topics, y=avg_sentiment, name="Sentiment", marker_color=sentiment_colors),
            row=2, col=1
        )
        
        # Transparency distribution (box plot)
        for topic, quotes in quotes_by_topic.items():
            if quotes:
                transparency_scores = [q.get('transparency_score', 100) for q in quotes]
                fig.add_trace(
                    go.Box(y=transparency_scores, name=topic, showlegend=False),
                    row=2, col=2
                )
        
        fig.update_layout(
            title="Quote Analysis Comparison Across Topics",
            height=800,
            showlegend=False
        )
        
        return fig