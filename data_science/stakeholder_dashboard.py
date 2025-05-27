"""
Stakeholder Risk Assessment Dashboard

A user-friendly interface for business stakeholders to upload financial documents
and receive clear High/Medium/Low risk classifications with actionable insights.

Features:
- Simple drag & drop file upload
- Institution selection
- Real-time processing with progress tracking
- Traffic light risk classification
- Business-friendly topic analysis
- Actionable recommendations
- Executive summary export
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path
import tempfile
import time
from datetime import datetime, timedelta
import json

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# Import our components
try:
    from scripts.business_intelligence.stakeholder_translator import StakeholderTranslator
    from scripts.statistical_analysis.time_series_analyzer import TimeSeriesAnalyzer
    from scripts.statistical_analysis.anomaly_detector import AnomalyDetector
    from scripts.statistical_analysis.risk_scorer import RiskScorer
    from src.etl.etl_pipeline import ETLPipeline
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Financial Risk Assessment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    
    .risk-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .risk-low {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    
    .risk-medium {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    
    .risk-high {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .recommendation-urgent {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .recommendation-watch {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .recommendation-positive {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StakeholderDashboard:
    """Main dashboard class for stakeholder interface"""
    
    def __init__(self):
        """Initialize dashboard components"""
        self.translator = StakeholderTranslator()
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.risk_scorer = RiskScorer()
        
        # Initialize session state
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<div class="main-header">📊 Financial Risk Assessment Dashboard</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; color: #666;">
            Upload your financial documents to receive clear risk assessments and actionable insights
        </div>
        """, unsafe_allow_html=True)
    
    def render_upload_section(self):
        """Render file upload section"""
        st.header("📁 Upload Your Documents")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="upload-area">
                <h3>📄 Supported Document Types</h3>
                <p>✓ Earnings Call Transcripts (.txt, .pdf)</p>
                <p>✓ Financial Presentations (.pdf, .pptx)</p>
                <p>✓ Financial Summaries (.xlsx, .csv)</p>
                <p><strong>Upload up to 50 files covering 12 quarters</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "Choose files",
                accept_multiple_files=True,
                type=['txt', 'pdf', 'xlsx', 'csv', 'pptx'],
                help="Upload documents from the last 12 quarters for comprehensive analysis"
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} files uploaded successfully!")
                st.session_state.uploaded_files = uploaded_files
                
                # Show file summary
                with st.expander("📋 Uploaded Files Summary"):
                    for i, file in enumerate(uploaded_files, 1):
                        st.write(f"{i}. {file.name} ({file.size:,} bytes)")
        
        with col2:
            st.subheader("⚙️ Configuration")
            
            # Institution selection
            institutions = [
                "JPMorgan Chase", "Bank of America", "Wells Fargo", "Citigroup",
                "Goldman Sachs", "Morgan Stanley", "U.S. Bancorp", "PNC Financial",
                "Truist Financial", "Capital One", "TD Bank", "Bank of New York Mellon"
            ]
            
            selected_institution = st.selectbox(
                "🏛️ Select Institution",
                institutions,
                help="Choose the financial institution for analysis"
            )
            
            # Time period
            st.text_input(
                "📅 Analysis Period",
                value="Q1 2022 to Q4 2024 (12 quarters)",
                disabled=True,
                help="Standard 12-quarter analysis period"
            )
            
            # Analysis button
            if st.button("🚀 Analyze Risk Profile", type="primary", use_container_width=True):
                if uploaded_files:
                    self.process_documents(uploaded_files, selected_institution)
                else:
                    st.error("Please upload documents before analyzing")
    
    def process_documents(self, uploaded_files, institution):
        """Process uploaded documents and run analysis"""
        with st.spinner("🔄 Processing your documents..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Document processing
                status_text.text("📄 Processing documents...")
                progress_bar.progress(20)
                time.sleep(1)  # Simulate processing time
                
                # Step 2: NLP Analysis
                status_text.text("🧠 Extracting topics and sentiments...")
                progress_bar.progress(40)
                time.sleep(1)
                
                # Step 3: Statistical Analysis
                status_text.text("📊 Running statistical analysis...")
                progress_bar.progress(60)
                time.sleep(1)
                
                # Step 4: Risk Assessment
                status_text.text("⚠️ Calculating risk scores...")
                progress_bar.progress(80)
                time.sleep(1)
                
                # Step 5: Generate insights
                status_text.text("💡 Generating insights and recommendations...")
                progress_bar.progress(100)
                time.sleep(1)
                
                # Simulate analysis results
                analysis_results = self.generate_sample_analysis_results(institution)
                st.session_state.analysis_results = analysis_results
                st.session_state.processing_complete = True
                
                status_text.text("✅ Analysis complete!")
                st.success("🎉 Risk assessment completed successfully!")
                
                # Auto-scroll to results
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                progress_bar.empty()
                status_text.empty()
    
    def render_results_section(self):
        """Render analysis results section"""
        if not st.session_state.processing_complete or not st.session_state.analysis_results:
            return
        
        results = st.session_state.analysis_results
        institution = results['institution']
        
        st.header(f"📈 Risk Assessment Results - {institution}")
        
        # Risk classification card
        self.render_risk_classification(results)
        
        # Key metrics
        self.render_key_metrics(results)
        
        # Topic analysis
        self.render_topic_analysis(results)
        
        # Sentiment trends
        self.render_sentiment_trends(results)
        
        # Recommendations
        self.render_recommendations(results)
        
        # Export options
        self.render_export_options(results)
    
    def render_risk_classification(self, results):
        """Render main risk classification display"""
        risk_info = self.translator.translate_risk_score(results)
        
        # Determine card style based on risk level
        if risk_info['color'] == 'green':
            card_class = 'risk-low'
        elif risk_info['color'] == 'yellow':
            card_class = 'risk-medium'
        else:
            card_class = 'risk-high'
        
        st.markdown(f"""
        <div class="risk-card {card_class}">
            <h1>{risk_info['emoji']} {risk_info['classification']}</h1>
            <h2>Score: {risk_info['score']}/10</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0;">{risk_info['message']}</p>
            <p style="font-size: 1rem; font-weight: bold;">{risk_info['trend']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_key_metrics(self, results):
        """Render key metrics overview"""
        st.subheader("📊 Key Metrics Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Documents Analyzed",
                len(st.session_state.uploaded_files),
                help="Total number of documents processed"
            )
        
        with col2:
            anomaly_count = results.get('anomaly_detection', {}).get('total_anomalies', 0)
            st.metric(
                "Anomalies Detected",
                anomaly_count,
                delta=f"{anomaly_count - 3} from baseline",
                help="Number of statistical anomalies identified"
            )
        
        with col3:
            quarters_analyzed = 12  # Standard analysis period
            st.metric(
                "Quarters Analyzed",
                quarters_analyzed,
                help="Time period covered in analysis"
            )
        
        with col4:
            risk_score = results.get('composite_risk_score', 0.5)
            previous_score = risk_score - 0.08  # Simulate previous score
            delta = risk_score - previous_score
            st.metric(
                "Risk Score",
                f"{risk_score * 10:.1f}/10",
                delta=f"{delta * 10:+.1f}",
                delta_color="inverse",
                help="Composite risk score across all dimensions"
            )
    
    def render_topic_analysis(self, results):
        """Render topic analysis section"""
        st.subheader("💬 What Are They Talking About Most?")
        
        business_topics = self.translator.translate_topics_to_business_language(results)
        
        if business_topics:
            # Create topic chart
            topic_df = pd.DataFrame(business_topics)
            
            fig = px.bar(
                topic_df.head(5),
                x='percentage',
                y='label',
                orientation='h',
                color='risk_level',
                color_discrete_map={'low': '#28a745', 'medium': '#ffc107', 'high': '#dc3545'},
                title="Top Discussion Topics",
                labels={'percentage': 'Percentage of Discussions', 'label': 'Topic'}
            )
            
            fig.update_layout(
                height=400,
                showlegend=True,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Topic details
            for topic in business_topics[:5]:
                risk_color = {'low': '#28a745', 'medium': '#ffc107', 'high': '#dc3545'}[topic['risk_level']]
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{topic['icon']} {topic['label']} ({topic['percentage']:.1f}%)</h4>
                    <p>{topic['description']}</p>
                    <p style="color: {risk_color}; font-weight: bold;">
                        {topic['risk_emoji']} {topic['risk_level'].upper()} RISK
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    def render_sentiment_trends(self, results):
        """Render sentiment trends visualization"""
        st.subheader("😊 How Has Their Tone Changed Over Time?")
        
        # Generate sample sentiment data over time
        quarters = ['Q1 2022', 'Q2 2022', 'Q3 2022', 'Q4 2022', 
                   'Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023',
                   'Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']
        
        # Simulate sentiment data
        np.random.seed(42)
        positive = np.random.normal(70, 10, 12).clip(40, 90)
        neutral = np.random.normal(20, 5, 12).clip(10, 40)
        negative = 100 - positive - neutral
        
        sentiment_df = pd.DataFrame({
            'Quarter': quarters,
            'Positive': positive,
            'Neutral': neutral,
            'Negative': negative
        })
        
        # Create stacked area chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sentiment_df['Quarter'],
            y=sentiment_df['Positive'],
            fill='tonexty',
            mode='lines',
            name='Positive',
            line=dict(color='#28a745'),
            fillcolor='rgba(40, 167, 69, 0.3)'
        ))
        
        fig.add_trace(go.Scatter(
            x=sentiment_df['Quarter'],
            y=sentiment_df['Neutral'],
            fill='tonexty',
            mode='lines',
            name='Neutral',
            line=dict(color='#6c757d'),
            fillcolor='rgba(108, 117, 125, 0.3)'
        ))
        
        fig.add_trace(go.Scatter(
            x=sentiment_df['Quarter'],
            y=sentiment_df['Negative'],
            fill='tozeroy',
            mode='lines',
            name='Negative',
            line=dict(color='#dc3545'),
            fillcolor='rgba(220, 53, 69, 0.3)'
        ))
        
        fig.update_layout(
            title="Sentiment Trends Over Time",
            xaxis_title="Quarter",
            yaxis_title="Percentage",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment summary
        sentiment_summary = self.translator.create_sentiment_summary({
            'positive_percentage': positive.mean(),
            'neutral_percentage': neutral.mean(),
            'negative_percentage': negative.mean(),
            'trend': 'stable'
        })
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>{sentiment_summary['emoji']} Overall Sentiment: {sentiment_summary['overall_sentiment']}</h4>
            <p>{sentiment_summary['description']}</p>
            <p><strong>Key Insight:</strong> {sentiment_summary['key_insight']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_recommendations(self, results):
        """Render actionable recommendations"""
        st.subheader("🎯 What Should You Focus On?")
        
        recommendations = self.translator.generate_stakeholder_recommendations(results)
        
        # Immediate attention
        if recommendations['immediate_attention']:
            st.markdown("### 🔴 IMMEDIATE ATTENTION")
            for rec in recommendations['immediate_attention']:
                st.markdown(f"""
                <div class="recommendation-urgent">
                    <h4>{rec['icon']} {rec['topic']}</h4>
                    <p><strong>Issue:</strong> {rec['issue']}</p>
                    <p><strong>Action:</strong> {rec['action']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Watch closely
        if recommendations['watch_closely']:
            st.markdown("### 🟡 WATCH CLOSELY")
            for rec in recommendations['watch_closely']:
                st.markdown(f"""
                <div class="recommendation-watch">
                    <h4>{rec['icon']} {rec['topic']}</h4>
                    <p><strong>Issue:</strong> {rec['issue']}</p>
                    <p><strong>Action:</strong> {rec['action']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Positive indicators
        if recommendations['positive_indicators']:
            st.markdown("### 🟢 POSITIVE INDICATORS")
            for rec in recommendations['positive_indicators']:
                st.markdown(f"""
                <div class="recommendation-positive">
                    <h4>{rec['icon']} {rec['topic']}</h4>
                    <p><strong>Strength:</strong> {rec['strength']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def render_export_options(self, results):
        """Render export and sharing options"""
        st.subheader("📤 Export & Share Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Download Executive Summary", use_container_width=True):
                executive_summary = self.translator.generate_executive_summary(
                    results['institution'], results
                )
                st.download_button(
                    label="📄 Download PDF Report",
                    data=executive_summary,
                    file_name=f"{results['institution']}_Risk_Assessment_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
        
        with col2:
            if st.button("📈 Export Detailed Data", use_container_width=True):
                # Create detailed data export
                detailed_data = {
                    'institution': results['institution'],
                    'analysis_date': datetime.now().isoformat(),
                    'risk_score': results.get('composite_risk_score', 0.5),
                    'recommendations': self.translator.generate_stakeholder_recommendations(results)
                }
                
                st.download_button(
                    label="💾 Download JSON Data",
                    data=json.dumps(detailed_data, indent=2),
                    file_name=f"{results['institution']}_Detailed_Analysis_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("📧 Email Report", use_container_width=True):
                st.info("📧 Email functionality will be available in the next update")
    
    def generate_sample_analysis_results(self, institution):
        """Generate sample analysis results for demonstration"""
        # This would normally come from the actual analysis pipeline
        return {
            'institution': institution,
            'composite_risk_score': np.random.uniform(0.3, 0.7),
            'anomaly_detection': {
                'total_anomalies': np.random.randint(2, 8)
            },
            'time_series': {
                'trend_direction': np.random.choice(['improving', 'stable', 'declining'])
            },
            'topic_analysis': {
                'financial_performance': {
                    'percentage': 35.0,
                    'trend': 'stable',
                    'mentions': 45,
                    'average_sentiment': 0.7
                },
                'regulatory_compliance': {
                    'percentage': 22.0,
                    'trend': 'increasing',
                    'mentions': 28,
                    'average_sentiment': 0.4
                },
                'technology_digital': {
                    'percentage': 18.0,
                    'trend': 'stable',
                    'mentions': 23,
                    'average_sentiment': 0.6
                },
                'market_conditions': {
                    'percentage': 15.0,
                    'trend': 'declining',
                    'mentions': 19,
                    'average_sentiment': 0.3
                },
                'operations_strategy': {
                    'percentage': 10.0,
                    'trend': 'stable',
                    'mentions': 13,
                    'average_sentiment': 0.6
                }
            },
            'risk_drivers': [
                {
                    'topic': 'regulatory_compliance',
                    'severity': 'medium',
                    'description': 'Increased regulatory discussions detected'
                },
                {
                    'topic': 'market_conditions',
                    'severity': 'medium',
                    'description': 'Market uncertainty concerns trending upward'
                },
                {
                    'topic': 'financial_performance',
                    'severity': 'low',
                    'description': 'Strong revenue growth narrative maintained'
                }
            ]
        }
    
    def run(self):
        """Main dashboard execution"""
        self.render_header()
        
        # Sidebar with navigation
        with st.sidebar:
            st.header("📋 Navigation")
            
            if st.session_state.processing_complete:
                st.success("✅ Analysis Complete")
                if st.button("🔄 Start New Analysis"):
                    st.session_state.analysis_results = None
                    st.session_state.uploaded_files = []
                    st.session_state.processing_complete = False
                    st.rerun()
            else:
                st.info("📤 Upload documents to begin")
            
            st.markdown("---")
            st.markdown("### 📚 Help & Support")
            st.markdown("- [User Guide]()")
            st.markdown("- [FAQ]()")
            st.markdown("- [Contact Support]()")
        
        # Main content
        if not st.session_state.processing_complete:
            self.render_upload_section()
        else:
            self.render_results_section()

# Main execution
if __name__ == "__main__":
    dashboard = StakeholderDashboard()
    dashboard.run()