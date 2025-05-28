"""
Technical Data Science Dashboard Launcher for Bank of England Mosaic Lens

Comprehensive statistical validation dashboard with advanced analytics,
confidence scoring, p-value reporting, and model diagnostics.

Features:
- Statistical validation engine with bootstrap confidence intervals
- Hypothesis testing with multiple testing corrections
- Effect size analysis and practical significance assessment
- Model performance diagnostics and cross-validation
- Data quality assessment and outlier detection
- Interactive technical visualizations
- Exportable technical reports

Usage:
    streamlit run technical_dashboard_launcher.py --server.port 8510
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import tempfile
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging

# Add project paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "scripts"))

# Import statistical validation components
try:
    from scripts.statistical_validation import (
        StatisticalValidationEngine,
        TechnicalDashboard,
        TechnicalVisualizationEngine,
        StatisticalResults
    )
    STATISTICAL_VALIDATION_AVAILABLE = True
except ImportError as e:
    st.error(f"Statistical validation components not available: {e}")
    STATISTICAL_VALIDATION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="BoE Technical Data Science Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for technical styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff 0%, #e6f3ff 100%);
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #2c5aa0;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .technical-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        color: #1f4e79;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<div class="main-header">🔬 Bank of England Technical Data Science Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Statistical Validation & Model Diagnostics</div>', 
                unsafe_allow_html=True)
    
    if not STATISTICAL_VALIDATION_AVAILABLE:
        st.error("❌ Statistical validation components are not available. Please check the installation.")
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## 🔧 Dashboard Configuration")
        
        # Data source selection
        data_source_option = st.selectbox(
            "Select Data Source",
            ["Upload Files", "Generate Sample Data", "Use Demo Dataset"],
            index=2
        )
        
        # Statistical configuration
        st.markdown("### 📊 Statistical Settings")
        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
        significance_threshold = st.slider("Significance Threshold (α)", 0.01, 0.10, 0.05, 0.01)
        
        # Analysis options
        st.markdown("### 🔍 Analysis Options")
        include_bootstrap = st.checkbox("Bootstrap Confidence Intervals", True)
        include_hypothesis_tests = st.checkbox("Hypothesis Testing", True)
        include_effect_sizes = st.checkbox("Effect Size Analysis", True)
        include_cross_validation = st.checkbox("Cross-Validation", True)
        
        # Advanced options
        with st.expander("🔬 Advanced Options"):
            random_seed = st.number_input("Random Seed", 1, 9999, 42)
            bootstrap_samples = st.number_input("Bootstrap Samples", 1000, 50000, 10000)
            cv_folds = st.number_input("CV Folds", 3, 10, 5)
    
    # Data loading section
    risk_scores, true_values, features, data_info = load_data(data_source_option)
    
    if risk_scores is None:
        st.warning("⚠️ Please provide data to begin statistical validation.")
        return
    
    # Display data information
    display_data_info(risk_scores, true_values, features, data_info)
    
    # Initialize technical dashboard
    if st.button("🚀 Run Comprehensive Statistical Validation", type="primary"):
        run_statistical_validation(
            risk_scores=risk_scores,
            true_values=true_values,
            features=features,
            data_source=data_info.get('source', 'Unknown'),
            confidence_level=confidence_level,
            random_seed=random_seed
        )

def load_data(data_source_option: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame], Dict[str, Any]]:
    """Load data based on user selection"""
    
    if data_source_option == "Upload Files":
        return load_uploaded_data()
    elif data_source_option == "Generate Sample Data":
        return generate_sample_data()
    elif data_source_option == "Use Demo Dataset":
        return load_demo_dataset()
    else:
        return None, None, None, {}

def load_uploaded_data() -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame], Dict[str, Any]]:
    """Load data from uploaded files"""
    
    st.markdown("### 📁 Upload Data Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_scores_file = st.file_uploader(
            "Risk Scores (CSV/Excel)",
            type=['csv', 'xlsx'],
            help="Upload file containing risk scores"
        )
    
    with col2:
        true_values_file = st.file_uploader(
            "True Values (CSV/Excel) - Optional",
            type=['csv', 'xlsx'],
            help="Upload file containing ground truth values for validation"
        )
    
    features_file = st.file_uploader(
        "Features (CSV/Excel) - Optional",
        type=['csv', 'xlsx'],
        help="Upload file containing feature matrix for cross-validation"
    )
    
    if risk_scores_file is None:
        return None, None, None, {}
    
    try:
        # Load risk scores
        if risk_scores_file.name.endswith('.csv'):
            risk_df = pd.read_csv(risk_scores_file)
        else:
            risk_df = pd.read_excel(risk_scores_file)
        
        risk_scores = risk_df.iloc[:, 0].values  # First column
        
        # Load true values if provided
        true_values = None
        if true_values_file is not None:
            if true_values_file.name.endswith('.csv'):
                true_df = pd.read_csv(true_values_file)
            else:
                true_df = pd.read_excel(true_values_file)
            true_values = true_df.iloc[:, 0].values
        
        # Load features if provided
        features = None
        if features_file is not None:
            if features_file.name.endswith('.csv'):
                features = pd.read_csv(features_file)
            else:
                features = pd.read_excel(features_file)
        
        data_info = {
            'source': 'Uploaded Files',
            'risk_scores_shape': risk_scores.shape,
            'true_values_shape': true_values.shape if true_values is not None else None,
            'features_shape': features.shape if features is not None else None
        }
        
        return risk_scores, true_values, features, data_info
        
    except Exception as e:
        st.error(f"Error loading uploaded data: {e}")
        return None, None, None, {}

def generate_sample_data() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """Generate synthetic sample data for testing"""
    
    st.markdown("### 🎲 Generate Sample Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_samples = st.number_input("Number of Samples", 100, 10000, 1000)
    
    with col2:
        n_features = st.number_input("Number of Features", 3, 20, 5)
    
    with col3:
        noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1)
    
    if st.button("Generate Data"):
        np.random.seed(42)
        
        # Generate features
        features = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i+1}' for i in range(n_features)]
        )
        
        # Generate true relationship
        true_coefficients = np.random.randn(n_features)
        true_values = features.values @ true_coefficients + np.random.normal(0, 0.1, n_samples)
        
        # Generate predictions with noise
        risk_scores = true_values + np.random.normal(0, noise_level, n_samples)
        
        data_info = {
            'source': 'Generated Sample Data',
            'n_samples': n_samples,
            'n_features': n_features,
            'noise_level': noise_level,
            'true_r2': 1 - (noise_level**2 / (np.var(true_values) + noise_level**2))
        }
        
        st.success(f"✅ Generated {n_samples} samples with {n_features} features")
        
        return risk_scores, true_values, features, data_info
    
    return None, None, None, {}

def load_demo_dataset() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """Load demonstration dataset"""
    
    st.markdown("### 🎯 Demo Dataset: Bank Risk Assessment")
    
    # Generate realistic banking risk assessment data
    np.random.seed(42)
    n_samples = 500
    
    # Create realistic features
    features_data = {
        'loan_to_value_ratio': np.random.beta(2, 5, n_samples) * 100,
        'debt_to_income_ratio': np.random.gamma(2, 10, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples),
        'employment_years': np.random.exponential(5, n_samples),
        'property_value': np.random.lognormal(12, 0.5, n_samples),
        'loan_amount': np.random.lognormal(11, 0.7, n_samples)
    }
    
    features = pd.DataFrame(features_data)
    
    # Normalize features
    features_normalized = (features - features.mean()) / features.std()
    
    # Create true risk model
    true_coefficients = np.array([0.3, 0.4, -0.5, -0.2, -0.1, 0.3])
    true_risk_scores = features_normalized.values @ true_coefficients
    
    # Add realistic noise and transform to probability scale
    noise = np.random.normal(0, 0.2, n_samples)
    true_values = 1 / (1 + np.exp(-(true_risk_scores + noise)))  # Sigmoid transformation
    
    # Generate model predictions with some error
    prediction_noise = np.random.normal(0, 0.1, n_samples)
    risk_scores = 1 / (1 + np.exp(-(true_risk_scores + prediction_noise)))
    
    data_info = {
        'source': 'Demo Banking Risk Dataset',
        'description': 'Synthetic banking risk assessment data with realistic features',
        'n_samples': n_samples,
        'features': list(features.columns),
        'target': 'Default Probability',
        'model_type': 'Logistic Risk Model'
    }
    
    st.markdown("""
    <div class="technical-info">
    <strong>📊 Demo Dataset Information:</strong><br>
    • <strong>Samples:</strong> 500 loan applications<br>
    • <strong>Features:</strong> 6 financial risk indicators<br>
    • <strong>Target:</strong> Default probability (0-1)<br>
    • <strong>Model:</strong> Logistic regression with realistic noise<br>
    • <strong>Use Case:</strong> Bank loan default risk assessment
    </div>
    """, unsafe_allow_html=True)
    
    return risk_scores, true_values, features, data_info

def display_data_info(risk_scores: Optional[np.ndarray], 
                     true_values: Optional[np.ndarray],
                     features: Optional[pd.DataFrame],
                     data_info: Dict[str, Any]) -> None:
    """Display information about loaded data"""
    
    if risk_scores is None:
        return
    
    st.markdown("### 📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Risk Scores", len(risk_scores))
    
    with col2:
        st.metric("True Values", len(true_values) if true_values is not None else "N/A")
    
    with col3:
        st.metric("Features", f"{features.shape[1]} cols" if features is not None else "N/A")
    
    with col4:
        st.metric("Data Source", data_info.get('source', 'Unknown'))
    
    # Data quality preview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Risk Scores Distribution")
        risk_stats = {
            'Mean': f"{np.mean(risk_scores):.4f}",
            'Std': f"{np.std(risk_scores):.4f}",
            'Min': f"{np.min(risk_scores):.4f}",
            'Max': f"{np.max(risk_scores):.4f}",
            'Missing': f"{np.sum(np.isnan(risk_scores))}"
        }
        st.json(risk_stats)
    
    with col2:
        if true_values is not None:
            st.markdown("#### 🎯 True Values Distribution")
            true_stats = {
                'Mean': f"{np.mean(true_values):.4f}",
                'Std': f"{np.std(true_values):.4f}",
                'Min': f"{np.min(true_values):.4f}",
                'Max': f"{np.max(true_values):.4f}",
                'Missing': f"{np.sum(np.isnan(true_values))}"
            }
            st.json(true_stats)
        else:
            st.markdown("#### ⚠️ No True Values")
            st.info("Ground truth values not provided. Some validation metrics will be unavailable.")
    
    # Features preview
    if features is not None:
        with st.expander("🔍 Features Preview"):
            st.dataframe(features.head(), use_container_width=True)
            st.markdown(f"**Shape:** {features.shape[0]} rows × {features.shape[1]} columns")

def run_statistical_validation(risk_scores: np.ndarray,
                              true_values: Optional[np.ndarray],
                              features: Optional[pd.DataFrame],
                              data_source: str,
                              confidence_level: float,
                              random_seed: int) -> None:
    """Run comprehensive statistical validation"""
    
    st.markdown("---")
    st.markdown("## 🔬 Statistical Validation Results")
    
    # Initialize technical dashboard
    dashboard = TechnicalDashboard()
    
    # Configure validation engine
    dashboard.validation_engine.confidence_level = confidence_level
    dashboard.validation_engine.random_state = random_seed
    
    # Run the dashboard
    try:
        dashboard.render_dashboard(
            risk_scores=risk_scores,
            true_values=true_values,
            features=features,
            data_source=data_source
        )
        
        # Add footer with technical information
        st.markdown("---")
        st.markdown("""
        <div class="technical-info">
        <strong>🔬 Technical Dashboard Information:</strong><br>
        • <strong>Statistical Engine:</strong> Bootstrap confidence intervals, hypothesis testing, effect sizes<br>
        • <strong>Model Diagnostics:</strong> R², RMSE, residual analysis, Q-Q plots<br>
        • <strong>Data Quality:</strong> Missing data, outliers, distribution analysis<br>
        • <strong>Visualizations:</strong> Interactive Plotly charts with technical details<br>
        • <strong>Export:</strong> JSON reports, statistical summaries, full results
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error running statistical validation: {e}")
        logger.error(f"Statistical validation error: {e}", exc_info=True)

if __name__ == "__main__":
    main()