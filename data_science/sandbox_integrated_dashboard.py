"""
SANDBOX: Integrated BoE Dashboard (Port 8505 + Technical Dashboard)

Safe testing environment that combines:
- Main Bank of England Supervisor Dashboard functionality
- Technical Data Science Dashboard (from port 8510)
- Shared inference results between both views

This sandbox protects existing versions while testing integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import tempfile
from typing import Dict, List, Any, Optional, Tuple
import logging
import plotly.express as px
import plotly.graph_objects as go

# Add project paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "scripts"))

# Import components safely
try:
    from scripts.statistical_validation import (
        StatisticalValidationEngine,
        TechnicalDashboard,
        TechnicalVisualizationEngine,
        StatisticalResults
    )
    TECHNICAL_DASHBOARD_AVAILABLE = True
except ImportError as e:
    st.error(f"Technical dashboard components not available: {e}")
    TECHNICAL_DASHBOARD_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="BoE Sandbox: Integrated Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sandbox styling
st.markdown("""
<style>
    .sandbox-header {
        background: linear-gradient(90deg, #ff6b6b 0%, #ffa500 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .main-dashboard-section {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
        margin-bottom: 1rem;
    }
    
    .technical-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    
    .shared-data-section {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
    
    .safety-notice {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class SandboxIntegratedDashboard:
    """
    Sandbox environment for testing integrated dashboard functionality
    
    Safely combines main supervisor dashboard with technical validation
    without affecting production versions.
    """
    
    def __init__(self):
        """Initialize the sandbox dashboard"""
        self.technical_dashboard = None
        if TECHNICAL_DASHBOARD_AVAILABLE:
            self.technical_dashboard = TechnicalDashboard()
        
        # Initialize session state for shared data
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state for data sharing between components"""
        
        if 'sandbox_risk_scores' not in st.session_state:
            st.session_state.sandbox_risk_scores = None
        if 'sandbox_true_values' not in st.session_state:
            st.session_state.sandbox_true_values = None
        if 'sandbox_features' not in st.session_state:
            st.session_state.sandbox_features = None
        if 'sandbox_analysis_complete' not in st.session_state:
            st.session_state.sandbox_analysis_complete = False
        if 'sandbox_data_source' not in st.session_state:
            st.session_state.sandbox_data_source = "Sandbox Test"
    
    def render_sandbox_dashboard(self):
        """Render the complete sandbox dashboard"""
        
        # Sandbox header
        st.markdown("""
        <div class="sandbox-header">
            🧪 SANDBOX ENVIRONMENT: Integrated BoE Dashboard Testing
            <br><small>Safe testing of Port 8505 + Technical Dashboard integration</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Safety notice
        st.markdown("""
        <div class="safety-notice">
            <strong>🛡️ SAFETY NOTICE:</strong> This is a sandbox environment for testing integration. 
            No existing production dashboards will be affected. All data and results are isolated.
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar configuration
        self._render_sandbox_sidebar()
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab_technical = st.tabs([
            "📊 Risk Analysis (Main)",
            "🏛️ Supervisor Dashboard", 
            "📄 Reports & Export",
            "🔬 Technical Validation"
        ])
        
        with tab1:
            self._render_main_risk_analysis_tab()
        
        with tab2:
            self._render_supervisor_dashboard_tab()
        
        with tab3:
            self._render_reports_tab()
        
        with tab_technical:
            self._render_technical_validation_tab()
    
    def _render_sandbox_sidebar(self):
        """Render sandbox configuration sidebar"""
        
        with st.sidebar:
            st.markdown("## 🧪 Sandbox Configuration")
            
            # Environment status
            st.markdown("### 🔧 Environment Status")
            st.success("✅ Sandbox Mode Active")
            st.info(f"🔒 Production Protected")
            
            # Data source selection
            st.markdown("### 📊 Test Data Source")
            data_source_option = st.selectbox(
                "Select Test Data",
                ["Generate Banking Dataset", "Upload Test Files", "Use Demo Data"],
                index=0
            )
            
            # File upload section
            if data_source_option == "Upload Test Files":
                st.markdown("#### 📁 Upload Data Files")
                
                uploaded_risk_scores = st.file_uploader(
                    "Upload Risk Analysis Results (JSON/CSV/Excel)",
                    type=['json', 'csv', 'xlsx', 'xls'],
                    help="JSON export from Port 8505 or CSV/Excel with risk scores"
                )
                
                uploaded_true_values = st.file_uploader(
                    "Upload Ground Truth (JSON/CSV/Excel) - Optional",
                    type=['json', 'csv', 'xlsx', 'xls'],
                    help="Ground truth data for validation (if available)"
                )
                
                uploaded_features = st.file_uploader(
                    "Upload Features (JSON/CSV/Excel) - Optional",
                    type=['json', 'csv', 'xlsx', 'xls'],
                    help="Feature matrix for cross-validation analysis"
                )
                
                if uploaded_risk_scores is not None:
                    if st.button("📊 Process Uploaded Files"):
                        self._process_uploaded_files(
                            uploaded_risk_scores, uploaded_true_values, uploaded_features
                        )
            
            # Analysis configuration
            st.markdown("### ⚙️ Analysis Settings")
            
            # Risk analysis settings
            with st.expander("📊 Risk Analysis Settings"):
                risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5)
                confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
                sample_size = st.number_input("Sample Size", 100, 2000, 500)
            
            # Technical validation settings
            with st.expander("🔬 Technical Settings"):
                include_bootstrap = st.checkbox("Bootstrap Confidence Intervals", True, key="sidebar_bootstrap")
                include_hypothesis_tests = st.checkbox("Hypothesis Testing", True, key="sidebar_hypothesis")
                include_cross_validation = st.checkbox("Cross-Validation", True, key="sidebar_cv")
                bootstrap_samples = st.number_input("Bootstrap Samples", 1000, 10000, 5000, key="sidebar_bootstrap_samples")
            
            # Generate test data
            if st.button("🎲 Generate Test Data", type="primary"):
                self._generate_sandbox_test_data(
                    data_source_option, sample_size, risk_threshold
                )
            
            # Clear sandbox data
            if st.button("🗑️ Clear Sandbox Data"):
                self._clear_sandbox_data()
                st.rerun()
    
    def _render_main_risk_analysis_tab(self):
        """Render main risk analysis tab (simulating port 8505 functionality)"""
        
        st.markdown("""
        <div class="main-dashboard-section">
        <h3>📊 Main Risk Analysis Dashboard</h3>
        <p>Simulating main Bank of England Supervisor Dashboard functionality (Port 8505)</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.sandbox_risk_scores is None:
            st.warning("⚠️ No analysis data available. Please generate test data from the sidebar.")
            return
        
        # Display main dashboard results
        col1, col2, col3, col4 = st.columns(4)
        
        risk_scores = st.session_state.sandbox_risk_scores
        
        with col1:
            st.metric("Total Entities", len(risk_scores))
        
        with col2:
            high_risk_count = np.sum(risk_scores > 0.7)
            st.metric("High Risk Entities", high_risk_count)
        
        with col3:
            avg_risk = np.mean(risk_scores)
            st.metric("Average Risk Score", f"{avg_risk:.3f}")
        
        with col4:
            risk_trend = "↗️ Increasing" if avg_risk > 0.5 else "↘️ Decreasing"
            st.metric("Risk Trend", risk_trend)
        
        # Risk distribution chart
        st.markdown("### 📈 Risk Score Distribution")
        
        import plotly.express as px
        import plotly.graph_objects as go
        
        fig = px.histogram(
            x=risk_scores,
            nbins=30,
            title="Risk Score Distribution",
            labels={'x': 'Risk Score', 'y': 'Frequency'}
        )
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk categories
        st.markdown("### 🎯 Risk Categories")
        
        low_risk = np.sum(risk_scores < 0.3)
        medium_risk = np.sum((risk_scores >= 0.3) & (risk_scores < 0.7))
        high_risk = np.sum(risk_scores >= 0.7)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🟢 Low Risk", low_risk, f"{low_risk/len(risk_scores)*100:.1f}%")
        
        with col2:
            st.metric("🟡 Medium Risk", medium_risk, f"{medium_risk/len(risk_scores)*100:.1f}%")
        
        with col3:
            st.metric("🔴 High Risk", high_risk, f"{high_risk/len(risk_scores)*100:.1f}%")
        
        # Mark analysis as complete
        st.session_state.sandbox_analysis_complete = True
        
        st.success("✅ Risk analysis complete! Results are now available for technical validation.")
    
    def _render_supervisor_dashboard_tab(self):
        """Render supervisor dashboard tab"""
        
        st.markdown("""
        <div class="main-dashboard-section">
        <h3>🏛️ Supervisor Dashboard</h3>
        <p>Executive summary and regulatory oversight view</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.sandbox_analysis_complete:
            st.warning("⚠️ Please complete risk analysis first.")
            return
        
        # Executive summary
        st.markdown("### 📋 Executive Summary")
        
        risk_scores = st.session_state.sandbox_risk_scores
        
        # Key findings
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔍 Key Findings")
            high_risk_pct = np.sum(risk_scores > 0.7) / len(risk_scores) * 100
            
            if high_risk_pct > 20:
                st.error(f"🚨 High risk concentration: {high_risk_pct:.1f}% of entities")
            elif high_risk_pct > 10:
                st.warning(f"⚠️ Elevated risk levels: {high_risk_pct:.1f}% of entities")
            else:
                st.success(f"✅ Risk levels manageable: {high_risk_pct:.1f}% high risk")
            
            st.write(f"• Average risk score: {np.mean(risk_scores):.3f}")
            st.write(f"• Risk standard deviation: {np.std(risk_scores):.3f}")
            st.write(f"• Maximum risk score: {np.max(risk_scores):.3f}")
        
        with col2:
            st.markdown("#### 📊 Risk Distribution")
            
            # Pie chart of risk categories
            low_risk = np.sum(risk_scores < 0.3)
            medium_risk = np.sum((risk_scores >= 0.3) & (risk_scores < 0.7))
            high_risk = np.sum(risk_scores >= 0.7)
            
            fig = go.Figure(data=[go.Pie(
                labels=['Low Risk', 'Medium Risk', 'High Risk'],
                values=[low_risk, medium_risk, high_risk],
                hole=.3,
                marker_colors=['#28a745', '#ffc107', '#dc3545']
            )])
            
            fig.update_layout(
                title="Risk Category Distribution",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Regulatory compliance
        st.markdown("### 📜 Regulatory Compliance")
        
        compliance_score = 100 - (high_risk_pct * 2)  # Simple compliance metric
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Compliance Score", f"{compliance_score:.1f}%")
        
        with col2:
            st.metric("Entities Reviewed", len(risk_scores))
        
        with col3:
            st.metric("Review Date", datetime.now().strftime("%Y-%m-%d"))
    
    def _render_reports_tab(self):
        """Render reports and export tab"""
        
        st.markdown("""
        <div class="main-dashboard-section">
        <h3>📄 Reports & Export</h3>
        <p>Generate reports combining main analysis and technical validation</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.sandbox_analysis_complete:
            st.warning("⚠️ Please complete risk analysis first.")
            return
        
        # Report generation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Main Dashboard Report")
            
            if st.button("📄 Generate Executive Summary"):
                summary_data = self._generate_executive_summary()
                st.json(summary_data)
                
                # Download button
                summary_json = json.dumps(summary_data, indent=2)
                st.download_button(
                    label="📥 Download Executive Summary",
                    data=summary_json,
                    file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            st.markdown("#### 🔬 Technical Validation Report")
            
            if TECHNICAL_DASHBOARD_AVAILABLE and st.button("📊 Generate Technical Report"):
                technical_data = self._generate_technical_summary()
                st.json(technical_data)
                
                # Download button
                technical_json = json.dumps(technical_data, indent=2, default=str)
                st.download_button(
                    label="📥 Download Technical Report",
                    data=technical_json,
                    file_name=f"technical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Combined report
        st.markdown("#### 📋 Combined Report")
        
        if st.button("📊 Generate Combined Report", type="primary"):
            combined_data = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "report_type": "Combined Supervisor + Technical Report",
                    "data_source": st.session_state.sandbox_data_source,
                    "environment": "Sandbox Testing"
                },
                "executive_summary": self._generate_executive_summary(),
                "technical_validation": self._generate_technical_summary() if TECHNICAL_DASHBOARD_AVAILABLE else {"error": "Technical validation not available"}
            }
            
            st.json(combined_data)
            
            # Download combined report
            combined_json = json.dumps(combined_data, indent=2, default=str)
            st.download_button(
                label="📥 Download Combined Report",
                data=combined_json,
                file_name=f"combined_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def _render_technical_validation_tab(self):
        """Render technical validation tab (integrated from port 8510)"""
        
        st.markdown("""
        <div class="technical-section">
        <h3>🔬 Technical Data Science Validation</h3>
        <p>Advanced statistical analysis of risk assessment results</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not TECHNICAL_DASHBOARD_AVAILABLE:
            st.error("❌ Technical dashboard components are not available.")
            return
        
        if not st.session_state.sandbox_analysis_complete:
            st.warning("⚠️ Please complete risk analysis first to enable technical validation.")
            return
        
        # Shared data section
        st.markdown("""
        <div class="shared-data-section">
        <h4>📊 Shared Analysis Data</h4>
        <p>Using results from the main risk analysis for technical validation</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display shared data summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Scores", len(st.session_state.sandbox_risk_scores))
        
        with col2:
            st.metric("Ground Truth", 
                     len(st.session_state.sandbox_true_values) if st.session_state.sandbox_true_values is not None else "Available")
        
        with col3:
            st.metric("Features", 
                     f"{st.session_state.sandbox_features.shape[1]} cols" if st.session_state.sandbox_features is not None else "Available")
        
        # Technical validation configuration
        with st.expander("🔧 Technical Validation Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                confidence_level = st.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1)
                significance_threshold = st.selectbox("Significance Threshold", [0.01, 0.05, 0.10], index=1)
            
            with col2:
                include_bootstrap = st.checkbox("Bootstrap Confidence Intervals", True, key="tech_bootstrap")
                include_cross_validation = st.checkbox("Cross-Validation Analysis", True, key="tech_cv")
        
        # Run technical validation
        if st.button("🚀 Run Technical Validation", type="primary"):
            
            # Configure technical dashboard
            self.technical_dashboard.validation_engine.confidence_level = confidence_level
            
            # Run the integrated technical validation
            with st.spinner("Running comprehensive technical validation..."):
                try:
                    self.technical_dashboard.render_dashboard(
                        risk_scores=st.session_state.sandbox_risk_scores,
                        true_values=st.session_state.sandbox_true_values,
                        features=st.session_state.sandbox_features,
                        data_source=f"Sandbox: {st.session_state.sandbox_data_source}"
                    )
                    
                    st.success("✅ Technical validation completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Technical validation failed: {str(e)}")
                    logger.error(f"Technical validation error: {e}", exc_info=True)
    
    def _generate_sandbox_test_data(self, data_source_option: str, sample_size: int, risk_threshold: float):
        """Generate test data for sandbox"""
        
        np.random.seed(42)  # Reproducible results
        
        if data_source_option == "Generate Banking Dataset":
            # Generate realistic banking data
            features = pd.DataFrame({
                'loan_to_value_ratio': np.random.beta(2, 5, sample_size) * 100,
                'debt_to_income_ratio': np.random.gamma(2, 10, sample_size),
                'credit_score': np.random.normal(650, 100, sample_size),
                'employment_years': np.random.exponential(5, sample_size),
                'property_value': np.random.lognormal(12, 0.5, sample_size),
                'loan_amount': np.random.lognormal(11, 0.7, sample_size)
            })
            
            # Create realistic risk model
            features_norm = (features - features.mean()) / features.std()
            coefficients = np.array([0.3, 0.4, -0.5, -0.2, -0.1, 0.3])
            
            # True risk scores
            linear_combination = features_norm.values @ coefficients
            true_values = 1 / (1 + np.exp(-linear_combination))
            
            # Predicted risk scores with realistic error
            prediction_error = np.random.normal(0, 0.15, sample_size)
            risk_scores = 1 / (1 + np.exp(-(linear_combination + prediction_error)))
            
            data_source = "Generated Banking Dataset"
            
        else:
            # Simple test data
            risk_scores = np.random.beta(2, 5, sample_size)
            true_values = risk_scores + np.random.normal(0, 0.1, sample_size)
            features = pd.DataFrame(np.random.randn(sample_size, 4), 
                                  columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
            data_source = data_source_option
        
        # Store in session state
        st.session_state.sandbox_risk_scores = risk_scores
        st.session_state.sandbox_true_values = true_values
        st.session_state.sandbox_features = features
        st.session_state.sandbox_data_source = data_source
        st.session_state.sandbox_analysis_complete = False
        
        st.success(f"✅ Generated {sample_size} samples for testing!")
        st.rerun()
    
    def _process_uploaded_files(self,
                               uploaded_risk_scores,
                               uploaded_true_values=None,
                               uploaded_features=None):
        """Process uploaded files and load data into session state"""
        
        try:
            # Process risk scores file
            if uploaded_risk_scores.name.endswith('.json'):
                # Handle JSON export from Port 8505
                import json
                json_data = json.load(uploaded_risk_scores)
                
                # Extract risk scores from JSON structure
                risk_scores = self._extract_risk_scores_from_json(json_data)
                
                if risk_scores is None:
                    st.error("❌ Could not find risk scores in JSON file. Please check the file structure.")
                    return
                    
            elif uploaded_risk_scores.name.endswith('.csv'):
                risk_df = pd.read_csv(uploaded_risk_scores)
                # Extract risk scores (assume first column or column named 'risk_score')
                if 'risk_score' in risk_df.columns:
                    risk_scores = risk_df['risk_score'].values
                elif 'risk_scores' in risk_df.columns:
                    risk_scores = risk_df['risk_scores'].values
                else:
                    risk_scores = risk_df.iloc[:, 0].values  # First column
            else:
                risk_df = pd.read_excel(uploaded_risk_scores)
                # Extract risk scores (assume first column or column named 'risk_score')
                if 'risk_score' in risk_df.columns:
                    risk_scores = risk_df['risk_score'].values
                elif 'risk_scores' in risk_df.columns:
                    risk_scores = risk_df['risk_scores'].values
                else:
                    risk_scores = risk_df.iloc[:, 0].values  # First column
            
            # Process true values if provided
            true_values = None
            if uploaded_true_values is not None:
                if uploaded_true_values.name.endswith('.json'):
                    import json
                    true_json = json.load(uploaded_true_values)
                    true_values = self._extract_values_from_json(true_json, 'true_values')
                elif uploaded_true_values.name.endswith('.csv'):
                    true_df = pd.read_csv(uploaded_true_values)
                    if 'true_value' in true_df.columns:
                        true_values = true_df['true_value'].values
                    elif 'ground_truth' in true_df.columns:
                        true_values = true_df['ground_truth'].values
                    else:
                        true_values = true_df.iloc[:, 0].values
                else:
                    true_df = pd.read_excel(uploaded_true_values)
                    if 'true_value' in true_df.columns:
                        true_values = true_df['true_value'].values
                    elif 'ground_truth' in true_df.columns:
                        true_values = true_df['ground_truth'].values
                    else:
                        true_values = true_df.iloc[:, 0].values
            
            # Process features if provided
            features = None
            if uploaded_features is not None:
                if uploaded_features.name.endswith('.json'):
                    import json
                    features_json = json.load(uploaded_features)
                    features = self._extract_features_from_json(features_json)
                elif uploaded_features.name.endswith('.csv'):
                    features = pd.read_csv(uploaded_features)
                else:
                    features = pd.read_excel(uploaded_features)
            
            # Validate data
            if len(risk_scores) == 0:
                st.error("❌ No risk scores found in uploaded file.")
                return
            
            if true_values is not None and len(true_values) != len(risk_scores):
                st.warning("⚠️ Ground truth length doesn't match risk scores. Truncating to match.")
                min_len = min(len(risk_scores), len(true_values))
                risk_scores = risk_scores[:min_len]
                true_values = true_values[:min_len]
            
            if features is not None and len(features) != len(risk_scores):
                st.warning("⚠️ Features length doesn't match risk scores. Truncating to match.")
                min_len = min(len(risk_scores), len(features))
                risk_scores = risk_scores[:min_len]
                features = features.iloc[:min_len]
                if true_values is not None:
                    true_values = true_values[:min_len]
            
            # Store in session state
            st.session_state.sandbox_risk_scores = risk_scores
            st.session_state.sandbox_true_values = true_values
            st.session_state.sandbox_features = features
            st.session_state.sandbox_data_source = f"Uploaded Files ({len(risk_scores)} samples)"
            st.session_state.sandbox_analysis_complete = False
            
            # Display success message with data summary
            st.success(f"✅ Successfully loaded {len(risk_scores)} samples!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Scores", len(risk_scores))
            with col2:
                st.metric("Ground Truth", "Yes" if true_values is not None else "No")
            with col3:
                st.metric("Features", f"{features.shape[1]} cols" if features is not None else "No")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing uploaded files: {str(e)}")
            logger.error(f"File processing error: {e}", exc_info=True)
    
    def _extract_risk_scores_from_json(self, json_data: dict) -> Optional[np.ndarray]:
        """Extract risk scores from JSON export from Port 8505"""
        
        try:
            # Common JSON structures from Port 8505 exports
            possible_paths = [
                # Direct risk scores array
                ['risk_scores'],
                ['results', 'risk_scores'],
                ['analysis', 'risk_scores'],
                ['data', 'risk_scores'],
                
                # Risk analysis results
                ['risk_analysis', 'scores'],
                ['analysis_results', 'risk_scores'],
                ['supervisor_analysis', 'risk_scores'],
                
                # Individual risk entries
                ['entities', 'risk_score'],  # Array of objects with risk_score field
                ['results', 'entities', 'risk_score'],
                ['analysis', 'entities', 'risk_score'],
                
                # Alternative naming
                ['predictions'],
                ['scores'],
                ['risk_values'],
                ['risk_assessment', 'scores']
            ]
            
            # Try each possible path
            for path in possible_paths:
                current_data = json_data
                
                try:
                    # Navigate through the JSON structure
                    for key in path:
                        if isinstance(current_data, dict) and key in current_data:
                            current_data = current_data[key]
                        else:
                            break
                    else:
                        # Successfully navigated the path
                        if isinstance(current_data, list):
                            # Check if it's a list of numbers or objects with risk scores
                            if len(current_data) > 0:
                                if isinstance(current_data[0], (int, float)):
                                    # Direct array of risk scores
                                    return np.array(current_data)
                                elif isinstance(current_data[0], dict):
                                    # Array of objects - extract risk scores
                                    risk_score_keys = ['risk_score', 'score', 'risk', 'prediction', 'value']
                                    for key in risk_score_keys:
                                        if key in current_data[0]:
                                            scores = [item[key] for item in current_data if key in item]
                                            return np.array(scores)
                        elif isinstance(current_data, (int, float)):
                            # Single risk score
                            return np.array([current_data])
                            
                except (KeyError, TypeError, IndexError):
                    continue
            
            # If no standard path worked, try to find any numeric arrays
            def find_numeric_arrays(data, max_depth=3, current_depth=0):
                if current_depth > max_depth:
                    return None
                    
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], (int, float)) and len(data) > 5:  # Assume risk scores array has >5 entries
                        return np.array(data)
                elif isinstance(data, dict):
                    for value in data.values():
                        result = find_numeric_arrays(value, max_depth, current_depth + 1)
                        if result is not None:
                            return result
                return None
            
            numeric_array = find_numeric_arrays(json_data)
            if numeric_array is not None:
                return numeric_array
                
            return None
            
        except Exception as e:
            logger.error(f"Error extracting risk scores from JSON: {e}")
            return None
    
    def _extract_values_from_json(self, json_data: dict, value_type: str) -> Optional[np.ndarray]:
        """Extract values (true values, ground truth) from JSON"""
        
        try:
            possible_paths = [
                [value_type],
                ['ground_truth'],
                ['true_values'],
                ['actual_values'],
                ['labels'],
                ['targets'],
                ['results', value_type],
                ['data', value_type]
            ]
            
            for path in possible_paths:
                current_data = json_data
                try:
                    for key in path:
                        if isinstance(current_data, dict) and key in current_data:
                            current_data = current_data[key]
                        else:
                            break
                    else:
                        if isinstance(current_data, list) and len(current_data) > 0:
                            if isinstance(current_data[0], (int, float)):
                                return np.array(current_data)
                except (KeyError, TypeError, IndexError):
                    continue
                    
            return None
            
        except Exception as e:
            logger.error(f"Error extracting {value_type} from JSON: {e}")
            return None
    
    def _extract_features_from_json(self, json_data: dict) -> Optional[pd.DataFrame]:
        """Extract features from JSON"""
        
        try:
            possible_paths = [
                ['features'],
                ['input_features'],
                ['data', 'features'],
                ['analysis', 'features'],
                ['entities'],
                ['records']
            ]
            
            for path in possible_paths:
                current_data = json_data
                try:
                    for key in path:
                        if isinstance(current_data, dict) and key in current_data:
                            current_data = current_data[key]
                        else:
                            break
                    else:
                        if isinstance(current_data, list) and len(current_data) > 0:
                            # Try to convert to DataFrame
                            if isinstance(current_data[0], dict):
                                return pd.DataFrame(current_data)
                            elif isinstance(current_data[0], list):
                                return pd.DataFrame(current_data)
                        elif isinstance(current_data, dict):
                            return pd.DataFrame([current_data])
                except (KeyError, TypeError, IndexError):
                    continue
                    
            return None
            
        except Exception as e:
            logger.error(f"Error extracting features from JSON: {e}")
            return None
    
    def _clear_sandbox_data(self):
        """Clear all sandbox data"""
        
        st.session_state.sandbox_risk_scores = None
        st.session_state.sandbox_true_values = None
        st.session_state.sandbox_features = None
        st.session_state.sandbox_analysis_complete = False
        st.session_state.sandbox_data_source = "Sandbox Test"
        
        st.success("✅ Sandbox data cleared!")
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary from main dashboard"""
        
        risk_scores = st.session_state.sandbox_risk_scores
        
        return {
            "summary_metadata": {
                "generated_at": datetime.now().isoformat(),
                "data_source": st.session_state.sandbox_data_source,
                "total_entities": len(risk_scores)
            },
            "risk_metrics": {
                "average_risk_score": float(np.mean(risk_scores)),
                "risk_standard_deviation": float(np.std(risk_scores)),
                "maximum_risk_score": float(np.max(risk_scores)),
                "minimum_risk_score": float(np.min(risk_scores))
            },
            "risk_categories": {
                "low_risk_count": int(np.sum(risk_scores < 0.3)),
                "medium_risk_count": int(np.sum((risk_scores >= 0.3) & (risk_scores < 0.7))),
                "high_risk_count": int(np.sum(risk_scores >= 0.7))
            },
            "compliance": {
                "high_risk_percentage": float(np.sum(risk_scores > 0.7) / len(risk_scores) * 100),
                "compliance_score": float(100 - (np.sum(risk_scores > 0.7) / len(risk_scores) * 100 * 2))
            }
        }
    
    def _generate_technical_summary(self) -> Dict[str, Any]:
        """Generate technical summary from validation"""
        
        if not TECHNICAL_DASHBOARD_AVAILABLE:
            return {"error": "Technical validation not available"}
        
        try:
            validator = StatisticalValidationEngine()
            results = validator.validate_risk_scores(
                st.session_state.sandbox_risk_scores,
                st.session_state.sandbox_true_values,
                st.session_state.sandbox_features
            )
            
            summary = validator.generate_statistical_summary(results)
            
            return {
                "technical_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "validation_engine": "StatisticalValidationEngine",
                    "confidence_level": validator.confidence_level
                },
                "data_quality": {
                    "quality_score": summary.get('data_quality_score', 0),
                    "missing_data_percentage": results.data_quality.get('missing_percentage', 0),
                    "outliers_percentage": results.data_quality.get('outliers_percentage', 0)
                },
                "statistical_validation": {
                    "statistical_confidence": summary.get('statistical_confidence', 'unknown'),
                    "significant_tests": len([p for p in results.p_values.values() if isinstance(p, float) and p < 0.05]),
                    "total_tests": len([p for p in results.p_values.values() if isinstance(p, float)])
                },
                "model_performance": {
                    "r2_score": results.model_performance.get('r2_score', None),
                    "rmse": results.model_performance.get('rmse', None),
                    "mae": results.model_performance.get('mae', None)
                } if results.model_performance else None,
                "key_findings": summary.get('key_findings', []),
                "recommendations": summary.get('recommendations', [])
            }
            
        except Exception as e:
            return {"error": f"Technical validation failed: {str(e)}"}

def main():
    """Main function to run the sandbox dashboard"""
    
    # Initialize and render sandbox dashboard
    sandbox = SandboxIntegratedDashboard()
    sandbox.render_sandbox_dashboard()
    
    # Footer with sandbox information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
    🧪 <strong>SANDBOX ENVIRONMENT</strong> - Safe testing of integrated dashboard functionality<br>
    Port 8505 (Main) + Port 8510 (Technical) integration testing<br>
    No production systems affected • All data isolated • Safe for experimentation
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()