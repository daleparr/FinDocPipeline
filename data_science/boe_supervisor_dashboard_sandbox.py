"""
BoE Supervisor Dashboard - Sandbox Version with Advanced Emerging Topics
Safe testing environment for new features without affecting the live dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import base64
import yaml
import sys
import os

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Import the new emerging topics modules
try:
    from emerging_topics.trend_detection_engine import EmergingTopicsEngine
    from emerging_topics.statistical_significance import StatisticalSignificanceTester
    from emerging_topics.advanced_visualizations import AdvancedVisualizationEngine
    from emerging_topics.quote_analyzer import QuoteAnalyzer
    from emerging_topics.financial_verification import FinancialVerificationEngine
    EMERGING_TOPICS_AVAILABLE = True
except ImportError as e:
    st.error(f"Advanced Emerging Topics module not available: {e}")
    EMERGING_TOPICS_AVAILABLE = False

class BoESupervisorDashboardSandbox:
    """
    Enhanced BoE Supervisor Dashboard with Advanced Emerging Topics Analysis
    """
    
    def __init__(self):
        self.setup_page_config()
        self.load_config()
        
        # Initialize emerging topics components if available
        if EMERGING_TOPICS_AVAILABLE:
            self.emerging_topics_engine = EmergingTopicsEngine()
            self.statistical_tester = StatisticalSignificanceTester()
            self.visualization_engine = AdvancedVisualizationEngine()
            self.quote_analyzer = QuoteAnalyzer()
            self.financial_verifier = FinancialVerificationEngine()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="BoE Supervisor Dashboard - Sandbox",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for enhanced styling
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1f4e79 0%, #2d5aa0 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .main-header h1 {
            color: white;
            margin: 0;
            text-align: center;
        }
        .sandbox-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .feature-card {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .urgency-critical { border-left: 4px solid #d32f2f; }
        .urgency-high { border-left: 4px solid #f57c00; }
        .urgency-medium { border-left: 4px solid #fbc02d; }
        .urgency-low { border-left: 4px solid #388e3c; }
        </style>
        """, unsafe_allow_html=True)
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'emerging_topics_config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = self._get_default_config()
        except Exception as e:
            st.warning(f"Could not load config: {e}. Using defaults.")
            self.config = self._get_default_config()
    
    def run(self):
        """Main application entry point"""
        self.render_header()
        self.render_sandbox_warning()
        self.render_sidebar()
        self.render_main_content()
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🏦 Bank of England Supervisor Dashboard - Sandbox</h1>
            <p style="color: white; text-align: center; margin: 0;">
                Advanced Emerging Topics & Trend Analysis Testing Environment
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sandbox_warning(self):
        """Render sandbox environment warning"""
        st.markdown("""
        <div class="sandbox-warning">
            <h4>🧪 Sandbox Environment</h4>
            <p><strong>This is a safe testing environment for new features.</strong></p>
            <ul>
                <li>✅ Test new Advanced Emerging Topics analysis</li>
                <li>✅ Experiment with statistical significance testing</li>
                <li>✅ Preview enhanced visualizations</li>
                <li>⚠️ No impact on live dashboard or production data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls"""
        st.sidebar.title("📊 Analysis Controls")
        
        # Feature selection
        st.sidebar.markdown("### 🎯 Feature Selection")
        
        # Standard analysis options
        self.enable_standard_analysis = st.sidebar.checkbox("Standard Risk Analysis", value=True)
        self.enable_sentiment_analysis = st.sidebar.checkbox("Sentiment Analysis", value=True)
        self.enable_topic_analysis = st.sidebar.checkbox("Topic Analysis", value=True)
        
        # NEW: Advanced Emerging Topics
        st.sidebar.markdown("### 🚀 New Features")
        self.enable_emerging_topics = st.sidebar.checkbox(
            "Advanced Emerging Topics Analysis", 
            value=EMERGING_TOPICS_AVAILABLE,
            disabled=not EMERGING_TOPICS_AVAILABLE,
            help="Real-time trend detection with statistical significance testing"
        )
        
        if self.enable_emerging_topics and EMERGING_TOPICS_AVAILABLE:
            st.sidebar.markdown("**Emerging Topics Settings:**")
            self.significance_threshold = st.sidebar.slider(
                "Significance Threshold", 
                min_value=0.01, 
                max_value=0.10, 
                value=0.05, 
                step=0.01,
                help="P-value threshold for statistical significance"
            )
            
            self.confidence_level = st.sidebar.slider(
                "Confidence Level", 
                min_value=0.90, 
                max_value=0.99, 
                value=0.95, 
                step=0.01,
                help="Confidence level for interval estimation"
            )
        
        # Analysis parameters
        st.sidebar.markdown("### ⚙️ Analysis Parameters")
        self.risk_threshold = st.sidebar.slider("Risk Threshold", 0.1, 0.9, 0.7)
        self.sentiment_sensitivity = st.sidebar.slider("Sentiment Sensitivity", 0.1, 1.0, 0.5)
        
        # File upload
        st.sidebar.markdown("### 📁 Document Upload")
        self.uploaded_files = st.sidebar.file_uploader(
            "Upload Documents",
            type=['pdf', 'txt', 'docx', 'xlsx'],
            accept_multiple_files=True,
            help="Upload financial documents for analysis"
        )
        
        # Analysis button
        self.run_analysis = st.sidebar.button(
            "🔍 Run Analysis",
            type="primary",
            use_container_width=True
        )
    
    def render_main_content(self):
        """Render the main content area"""
        if not self.uploaded_files:
            self.render_welcome_screen()
        elif self.run_analysis:
            self.render_analysis_results()
        else:
            self.render_file_preview()
    
    def render_welcome_screen(self):
        """Render welcome screen with feature overview"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Standard Features
            - **Risk Assessment**: Comprehensive risk scoring
            - **Sentiment Analysis**: Document sentiment evaluation
            - **Topic Analysis**: Key topic identification
            - **Regulatory Flags**: Compliance monitoring
            """)
        
        with col2:
            if EMERGING_TOPICS_AVAILABLE:
                st.markdown("""
                ### 🚀 New: Advanced Emerging Topics
                - **Real-time Trend Detection**: Statistical trend analysis
                - **Significance Testing**: P-values and confidence intervals
                - **Regulatory Urgency Scoring**: Priority-based ranking
                - **Enhanced Visualizations**: Interactive charts with statistical indicators
                """)
            else:
                st.error("Advanced Emerging Topics module not available")
        
        st.markdown("---")
        st.info("👆 Upload documents in the sidebar to begin analysis")
    
    def render_file_preview(self):
        """Render file preview and document analysis summary"""
        st.markdown("### 📋 Document Analysis Summary")
        
        if self.uploaded_files:
            # Create document summary
            doc_summary = self._create_document_summary()
            st.dataframe(doc_summary, use_container_width=True)
            
            # Document quality assessment
            quality_score = self._assess_document_quality()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents Uploaded", len(self.uploaded_files))
            with col2:
                st.metric("Quality Score", f"{quality_score:.1%}")
            with col3:
                status = "✅ Ready" if quality_score > 0.7 else "⚠️ Review" if quality_score > 0.4 else "❌ Poor"
                st.metric("Status", status)
            
            if quality_score <= 0.4:
                st.warning("⚠️ Document quality is low. Consider uploading additional documents for better analysis.")
    
    def render_analysis_results(self):
        """Render comprehensive analysis results"""
        st.markdown("### 📊 Analysis Results")
        
        # Create sample data for demonstration
        sample_data = self._create_sample_data()
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Standard analysis
        if self.enable_standard_analysis:
            status_text.text("Running risk assessment...")
            progress_bar.progress(20)
            self.render_risk_analysis(sample_data)
        
        if self.enable_sentiment_analysis:
            status_text.text("Analyzing sentiment...")
            progress_bar.progress(40)
            self.render_sentiment_analysis(sample_data)
        
        if self.enable_topic_analysis:
            status_text.text("Identifying topics...")
            progress_bar.progress(60)
            self.render_topic_analysis(sample_data)
        
        # NEW: Advanced Emerging Topics Analysis
        if self.enable_emerging_topics and EMERGING_TOPICS_AVAILABLE:
            status_text.text("Detecting emerging topics...")
            progress_bar.progress(80)
            self.render_emerging_topics_analysis(sample_data)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Clear progress after a moment
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
    
    def render_emerging_topics_analysis(self, sample_data):
        """Render the new Advanced Emerging Topics analysis"""
        st.markdown("### 🚀 Advanced Emerging Topics & Trend Analysis")
        
        try:
            # Run emerging topics detection
            emerging_results = self.emerging_topics_engine.detect_emerging_topics(sample_data)
            
            if 'emerging_topics' in emerging_results and emerging_results['emerging_topics']:
                # Summary metrics
                summary = emerging_results.get('analysis_summary', {})
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Topics Analyzed", summary.get('total_topics_analyzed', 0))
                with col2:
                    st.metric("Significant Trends", summary.get('significant_trends_count', 0))
                with col3:
                    st.metric("High Urgency", summary.get('high_urgency_count', 0))
                with col4:
                    st.metric("Avg Growth Rate", f"{summary.get('avg_growth_rate', 0)}%")
                
                # Detailed topic analysis
                st.markdown("#### 📈 Emerging Topics Details")
                
                for topic_key, topic_data in emerging_results['emerging_topics'].items():
                    self._render_topic_card(topic_data)
                
                # Visualizations
                st.markdown("#### 📊 Advanced Visualizations")
                
                # Trend heatmap
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Trend Heatmap**")
                    heatmap_fig = self.visualization_engine.create_trend_heatmap(
                        emerging_results['emerging_topics']
                    )
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Urgency Scatter Plot**")
                    scatter_fig = self.visualization_engine.create_urgency_scatter_plot(
                        emerging_results['emerging_topics']
                    )
                    st.plotly_chart(scatter_fig, use_container_width=True)
                
                # Confidence intervals
                st.markdown("**Growth Rate Confidence Intervals**")
                ci_fig = self.visualization_engine.create_confidence_interval_chart(
                    emerging_results['emerging_topics']
                )
                st.plotly_chart(ci_fig, use_container_width=True)
                
                # Statistical summary table
                st.markdown("#### 📋 Statistical Summary")
                summary_table = self.visualization_engine.create_statistical_summary_table(
                    emerging_results['emerging_topics']
                )
                st.dataframe(summary_table, use_container_width=True)
                
                # Methodology transparency
                if st.expander("📊 Statistical Methodology & Transparency"):
                    methodology = emerging_results.get('methodology', {})
                    self.visualization_engine.render_methodology_transparency(methodology)
                
                # NEW: Quote Analysis Section
                st.markdown("#### 💬 Detailed Quote Analysis")
                
                # Topic selection for quote analysis
                available_topics = list(emerging_results['emerging_topics'].keys())
                if available_topics:
                    selected_topic = st.selectbox(
                        "Select topic for detailed quote analysis:",
                        options=available_topics,
                        format_func=lambda x: emerging_results['emerging_topics'][x].get('topic_name', x)
                    )
                    
                    if selected_topic:
                        self.render_quote_analysis(sample_data, selected_topic)
                
            else:
                st.info("No significant emerging topics detected in the current dataset.")
                # Show quote analysis for Climate Risk as demonstration
                st.markdown("#### 💬 Climate Risk Quote Analysis (Demo)")
                self.render_quote_analysis(sample_data, "Climate Risk")
                
        except Exception as e:
            st.error(f"Error in emerging topics analysis: {str(e)}")
            st.info("Using fallback analysis for demonstration...")
            
            # Show fallback analysis
            fallback_results = self.emerging_topics_engine._get_fallback_analysis()
            
            # Render fallback topic cards
            for topic_key, topic_data in fallback_results['emerging_topics'].items():
                self._render_topic_card(topic_data)
    
    def _render_topic_card(self, topic_data):
        """Render individual topic analysis card"""
        urgency = topic_data.get('regulatory_urgency', 'Low')
        urgency_class = f"urgency-{urgency.lower()}"
        
        st.markdown(f"""
        <div class="feature-card {urgency_class}">
            <h4>{topic_data['topic_name']}</h4>
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span><strong>Urgency:</strong> {urgency}</span>
                <span><strong>Growth Rate:</strong> {topic_data['growth_rate']}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span><strong>Statistical Significance:</strong> 
                    {'✅ Significant' if topic_data.get('is_significant') else '❌ Not Significant'}</span>
                <span><strong>P-value:</strong> {topic_data.get('p_value', 1.0):.4f}</span>
            </div>
            <div style="margin-bottom: 10px;">
                <strong>Confidence Interval:</strong> 
                [{topic_data.get('confidence_interval', [0, 0])[0]:.1f}%, {topic_data.get('confidence_interval', [0, 0])[1]:.1f}%]
            </div>
            <div style="margin-bottom: 10px;">
                <strong>Key Speakers:</strong> {', '.join(topic_data.get('speakers', [])[:3])}
            </div>
            <div>
                <strong>Trend Classification:</strong> {topic_data.get('trend_classification', 'Unknown')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_risk_analysis(self, sample_data):
        """Render standard risk analysis"""
        st.markdown("#### 🎯 Risk Assessment")
        
        # Sample risk metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Risk Score", "0.72", "↑ 0.05")
        with col2:
            st.metric("Credit Risk", "0.68", "↓ 0.02")
        with col3:
            st.metric("Operational Risk", "0.75", "↑ 0.08")
    
    def render_sentiment_analysis(self, sample_data):
        """Render sentiment analysis"""
        st.markdown("#### 😊 Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Sentiment", "0.45", "↓ 0.10")
        with col2:
            st.metric("Sentiment Volatility", "0.23", "↑ 0.05")
    
    def render_topic_analysis(self, sample_data):
        """Render topic analysis"""
        st.markdown("#### 📝 Topic Analysis")
        
        # Sample topic distribution
        topics = ['Risk Management', 'Regulatory Compliance', 'Financial Performance', 'Digital Transformation']
        values = [0.35, 0.28, 0.22, 0.15]
        
        fig = px.pie(values=values, names=topics, title="Topic Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_sample_data(self):
        """Create sample data for analysis demonstration"""
        # Create realistic sample data that mimics the expected structure
        np.random.seed(42)  # For reproducible results
        
        quarters = ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024', 'Q1 2025']
        topics = [
            'Risk Management', 'Regulatory Compliance', 'Cyber Security', 
            'Climate Risk', 'Digital Transformation', 'Financial Performance',
            'Operational Risk', 'Credit Risk', 'Market Risk'
        ]
        speakers = ['CEO', 'CFO', 'Chief Risk Officer', 'Chief Technology Officer', 'Analyst']
        
        data = []
        for i in range(200):  # Generate 200 sample records
            data.append({
                'text': f"Sample text about {np.random.choice(topics)} discussion...",
                'quarter': np.random.choice(quarters),
                'speaker_norm': np.random.choice(speakers),
                'primary_topic': np.random.choice(topics),
                'sentiment_score': np.random.normal(0.5, 0.2),
                'source_file': f'document_{i//20 + 1}.pdf'
            })
        
        return pd.DataFrame(data)
    
    def _create_document_summary(self):
        """Create document analysis summary"""
        analysis = []
        for i, file in enumerate(self.uploaded_files):
            doc_type = self._classify_document_type(file.name)
            quarter = self._extract_quarter_from_filename(file.name)
            quality = np.random.uniform(0.3, 0.95)  # Random quality for demo
            
            analysis.append({
                'Filename': file.name,
                'Document Type': doc_type,
                'Estimated Quarter': quarter,
                'Size (KB)': round(file.size / 1024, 1),
                'Quality Score': f"{quality:.1%}",
                'Status': '✅ Ready' if quality > 0.7 else '⚠️ Review' if quality > 0.4 else '❌ Poor'
            })
        
        return pd.DataFrame(analysis)
    
    def _classify_document_type(self, filename):
        """Classify document type based on filename"""
        filename_lower = filename.lower()
        if 'transcript' in filename_lower or 'earnings' in filename_lower:
            return 'Earnings Transcript'
        elif 'financial' in filename_lower or 'statement' in filename_lower:
            return 'Financial Statement'
        elif any(ext in filename_lower for ext in ['.xlsx', '.xls', '.csv']):
            return 'Financial Data'
        else:
            return 'Other Document'
    
    def _extract_quarter_from_filename(self, filename):
        """Extract quarter from filename"""
        import re
        patterns = [
            r'Q(\d)[_\s]?(\d{4})',
            r'(\d{4})[_\s]?Q(\d)',
            r'quarter[_\s]?(\d)[_\s]?(\d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    if pattern.startswith('Q'):
                        return f"Q{groups[0]} {groups[1]}"
                    else:
                        return f"Q{groups[1]} {groups[0]}"
        
        return "Unknown"
    
    def _assess_document_quality(self):
        """Assess overall document quality"""
        if not self.uploaded_files:
            return 0.0
        
        # Simulate quality assessment
        file_count = len(self.uploaded_files)
        type_diversity = len(set(self._classify_document_type(f.name) for f in self.uploaded_files))
        size_adequacy = min(1.0, sum(f.size for f in self.uploaded_files) / (10 * 1024 * 1024))
        
        quality = (min(1.0, file_count / 8) * 0.4 +
                  min(1.0, type_diversity / 3) * 0.3 +
                  size_adequacy * 0.3)
        
        return quality
    
    def _get_default_config(self):
        """Return default configuration"""
        return {
            'temporal_analysis': {
                'baseline_period': 4,
                'recent_period': 2,
                'minimum_data_points': 5
            },
            'statistical_testing': {
                'significance_threshold': 0.05,
                'confidence_level': 0.95
            },
            'growth_classification': {
                'emerging': 50.0,
                'rapid': 100.0,
                'explosive': 250.0
            }
        }
    
    def render_quote_analysis(self, sample_data, topic):
        """Render detailed quote analysis for a specific topic"""
        try:
            if not EMERGING_TOPICS_AVAILABLE:
                st.error("Quote analysis requires the emerging topics modules")
                return
            
            # Extract quotes for the topic
            quotes = self.quote_analyzer.extract_topic_quotes(sample_data, topic, max_quotes=14)
            
            if not quotes:
                st.warning(f"No quotes found for topic: {topic}")
                return
            
            # Show summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Quotes", len(quotes))
            with col2:
                avg_sentiment = np.mean([q.get('sentiment_score', 0) for q in quotes])
                st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
            with col3:
                speakers = len(set(q.get('speaker', 'Unknown') for q in quotes))
                st.metric("Unique Speakers", speakers)
            with col4:
                time_span = "Q1 2025"  # Simplified for demo
                st.metric("Time Span", time_span)
            
            # Analyze contradictory sentiment
            contradiction_analysis = self.quote_analyzer.analyze_contradictory_sentiment(quotes)
            
            # Display contradiction analysis
            st.markdown("##### 🔍 Contradictory Sentiment Analysis")
            
            # Overall assessment
            assessment = contradiction_analysis['overall_assessment']
            transparency_score = contradiction_analysis['transparency_score']
            
            if assessment == 'significant_downplaying':
                st.error(f"⚠️ **Significant Downplaying Detected** (Transparency: {transparency_score:.1%})")
                st.markdown("The analysis suggests the topic may be significantly downplayed or deflected.")
            elif assessment == 'moderate_hedging':
                st.warning(f"⚠️ **Moderate Hedging Detected** (Transparency: {transparency_score:.1%})")
                st.markdown("The analysis suggests moderate use of hedging language and deflection.")
            elif assessment == 'cautious_but_honest':
                st.info(f"ℹ️ **Cautious but Honest** (Transparency: {transparency_score:.1%})")
                st.markdown("The analysis suggests cautious language but generally honest communication.")
            else:
                st.success(f"✅ **Honest and Open** (Transparency: {transparency_score:.1%})")
                st.markdown("The analysis suggests open and honest communication about the topic.")
            
            # Detailed contradiction metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                downplaying_count = len(contradiction_analysis['downplaying_indicators'])
                st.metric("Downplaying Indicators", downplaying_count)
            with col2:
                hedging_count = len(contradiction_analysis['hedging_patterns'])
                st.metric("Hedging Patterns", hedging_count)
            with col3:
                deflection_count = len(contradiction_analysis['deflection_attempts'])
                st.metric("Deflection Attempts", deflection_count)
            
            # Display individual quotes
            st.markdown("##### 📝 Individual Quotes with Timestamps")
            
            for i, quote in enumerate(quotes, 1):
                with st.expander(f"Quote {i}: {quote['speaker']} - {quote['timestamp']}"):
                    # Quote content
                    st.markdown(f"**Quote:** {quote['text']}")
                    
                    # Metadata
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Speaker:** {quote['speaker']}")
                        st.markdown(f"**Timestamp:** {quote['timestamp']}")
                        st.markdown(f"**Quarter:** {quote['quarter']}")
                    with col2:
                        st.markdown(f"**Sentiment Score:** {quote['sentiment_score']:.3f}")
                        st.markdown(f"**Source:** {quote['source_file']}")
                        st.markdown(f"**Relevance:** {quote['topic_relevance_score']:.2f}")
                    
                    # Contradiction analysis for this quote
                    quote_contradiction = quote.get('contradiction_analysis', {})
                    if any(quote_contradiction.values()):
                        st.markdown("**Contradiction Indicators:**")
                        if quote_contradiction.get('downplaying_count', 0) > 0:
                            st.markdown(f"• Downplaying: {quote_contradiction['downplaying_count']} instances")
                        if quote_contradiction.get('hedging_count', 0) > 0:
                            st.markdown(f"• Hedging: {quote_contradiction['hedging_count']} instances")
                        if quote_contradiction.get('deflection_count', 0) > 0:
                            st.markdown(f"• Deflection: {quote_contradiction['deflection_count']} instances")
                    
                    # Context
                    if quote.get('context_before') or quote.get('context_after'):
                        st.markdown("**Context:**")
                        if quote.get('context_before'):
                            st.markdown(f"*Before:* {quote['context_before']}")
                        if quote.get('context_after'):
                            st.markdown(f"*After:* {quote['context_after']}")
                    
                    # NEW: Financial Verification
                    if EMERGING_TOPICS_AVAILABLE and hasattr(self, 'financial_verifier'):
                        # Check if this quote contains exposure claims
                        quote_text = quote['text']
                        if any(term in quote_text.lower() for term in ['exposure', 'portfolio', 'limited', 'significant', 'high-carbon']):
                            st.markdown("**🔍 Financial Verification:**")
                            try:
                                verification_result = self.financial_verifier.verify_exposure_claim(quote_text, pd.DataFrame())
                                
                                if verification_result['discrepancy_detected']:
                                    st.error(f"⚠️ **Potential Misstatement Detected**")
                                    st.markdown(f"• **Claimed:** {verification_result['claim_analysis']['exposure_level_claimed'].title()}")
                                    st.markdown(f"• **Actual:** {verification_result['actual_exposure']['total_high_carbon_exposure']:.1%} exposure")
                                    st.markdown(f"• **Severity:** {verification_result['severity'].title()}")
                                else:
                                    st.success("✅ Claim appears consistent with financial data")
                                
                                # Show regulatory flags if any
                                if verification_result.get('regulatory_flags'):
                                    st.markdown("**Regulatory Flags:**")
                                    for flag in verification_result['regulatory_flags']:
                                        st.markdown(f"• {flag['description']} ({flag['priority'].upper()} priority)")
                                        
                            except Exception as e:
                                st.info(f"Financial verification unavailable: {str(e)}")
            
            # Summary insights
            st.markdown("##### 🎯 Key Insights")
            
            # Sentiment trend
            sentiments = [q.get('sentiment_score', 0) for q in quotes]
            if sentiments:
                sentiment_trend = "increasing" if sentiments[-1] > sentiments[0] else "decreasing" if sentiments[-1] < sentiments[0] else "stable"
                st.markdown(f"• **Sentiment Trend:** {sentiment_trend.title()} over time")
            
            # Speaker analysis
            speaker_counts = {}
            for quote in quotes:
                speaker = quote.get('speaker', 'Unknown')
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            
            most_active_speaker = max(speaker_counts.items(), key=lambda x: x[1])
            st.markdown(f"• **Most Active Speaker:** {most_active_speaker[0]} ({most_active_speaker[1]} quotes)")
            
            # Urgency indicators
            all_urgency_indicators = []
            for quote in quotes:
                all_urgency_indicators.extend(quote.get('urgency_indicators', []))
            
            if all_urgency_indicators:
                unique_indicators = list(set(all_urgency_indicators))
                st.markdown(f"• **Common Urgency Terms:** {', '.join(unique_indicators[:5])}")
            
            # Recommendation
            st.markdown("##### 💡 Regulatory Recommendation")
            
            if contradiction_analysis['transparency_score'] < 0.5:
                st.error("**High Priority Review Required:** Significant concerns about transparency and potential downplaying of risks.")
            elif contradiction_analysis['transparency_score'] < 0.7:
                st.warning("**Medium Priority Review:** Some hedging detected, consider follow-up questions in next review.")
            else:
                st.success("**Standard Monitoring:** Communication appears transparent and appropriate.")
                
        except Exception as e:
            st.error(f"Error in quote analysis: {str(e)}")
            st.info("Displaying sample analysis for demonstration...")
            
            # Show sample quotes for demonstration
            st.markdown("**Sample Climate Risk Quotes (14 mentions):**")
            
            sample_quotes = [
                "Climate risk remains a key focus area for our institution.",
                "We are actively monitoring our exposure to transition risks.",
                "While climate change presents challenges, we believe we are well positioned.",
                "Our portfolio has limited exposure to high-carbon sectors.",
                "The regulatory environment around climate risk is evolving rapidly.",
                "Climate scenario analysis shows potential impacts on our credit portfolio.",
                "However, we believe these risks are manageable given our diversification.",
                "We are committed to achieving net zero by 2050.",
                "This is not just a regulatory requirement but a business imperative.",
                "We continue to assess various climate-related factors.",
                "Our stress testing frameworks include climate scenarios.",
                "We have enhanced our ESG reporting capabilities.",
                "Climate considerations are integrated into our risk management.",
                "We are preparing for enhanced disclosure requirements."
            ]
            
            for i, quote in enumerate(sample_quotes, 1):
                st.markdown(f"**{i}.** {quote} *(2025-01-15 10:{30+i*2}:00 - Various Speakers)*")

def main():
    """Main application entry point"""
    dashboard = BoESupervisorDashboardSandbox()
    dashboard.run()

if __name__ == "__main__":
    main()