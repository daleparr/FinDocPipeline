"""
Integration Component for Technical Dashboard into Main BoE Package (Port 8505)

This component provides seamless integration of the Technical Data Science Dashboard
into the existing Bank of England Supervisor Dashboard running on localhost:8505.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project paths for integration
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "scripts"))

# Import technical dashboard components
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

class TechnicalDashboardIntegration:
    """
    Integration wrapper for Technical Dashboard into main BoE package
    
    Provides seamless integration with existing supervisor dashboard while
    maintaining separation of concerns and modular architecture.
    """
    
    def __init__(self):
        """Initialize the integration component"""
        self.dashboard = None
        if TECHNICAL_DASHBOARD_AVAILABLE:
            self.dashboard = TechnicalDashboard()
    
    def render_technical_validation_tab(self, 
                                      risk_scores: Optional[np.ndarray] = None,
                                      true_values: Optional[np.ndarray] = None,
                                      features: Optional[pd.DataFrame] = None,
                                      data_source: str = "Main Dashboard") -> None:
        """
        Render technical validation as a tab in the main dashboard
        
        Args:
            risk_scores: Risk scores from main analysis
            true_values: Ground truth values if available
            features: Feature matrix from analysis
            data_source: Source identifier for reporting
        """
        
        if not TECHNICAL_DASHBOARD_AVAILABLE:
            st.error("❌ Technical dashboard components are not available.")
            st.info("Please ensure all statistical validation dependencies are installed.")
            return
        
        st.markdown("## 🔬 Technical Data Science Validation")
        st.markdown("*Advanced Statistical Analysis & Model Diagnostics*")
        
        # Check if data is available from main dashboard
        if risk_scores is None:
            st.warning("⚠️ No risk analysis data available from main dashboard.")
            st.info("Please run risk analysis first, then return to this tab for technical validation.")
            
            # Provide option to use demo data for testing
            if st.button("🎯 Use Demo Data for Testing"):
                risk_scores, true_values, features = self._generate_demo_data()
                data_source = "Demo Banking Dataset"
                st.success("✅ Demo data loaded for technical validation testing.")
        
        if risk_scores is not None:
            # Display data summary
            self._display_integration_summary(risk_scores, true_values, features, data_source)
            
            # Configuration options (simplified for integration)
            with st.expander("🔧 Technical Analysis Configuration"):
                col1, col2 = st.columns(2)
                
                with col1:
                    confidence_level = st.selectbox(
                        "Confidence Level",
                        [0.90, 0.95, 0.99],
                        index=1,
                        help="Statistical confidence level for intervals"
                    )
                
                with col2:
                    include_cross_validation = st.checkbox(
                        "Include Cross-Validation",
                        value=True if features is not None else False,
                        disabled=features is None,
                        help="Perform cross-validation analysis (requires features)"
                    )
            
            # Run technical validation
            if st.button("🚀 Run Technical Validation", type="primary"):
                self._run_integrated_validation(
                    risk_scores, true_values, features, data_source, confidence_level
                )
    
    def render_technical_sidebar_metrics(self, 
                                       risk_scores: Optional[np.ndarray] = None,
                                       true_values: Optional[np.ndarray] = None) -> None:
        """
        Render key technical metrics in the main dashboard sidebar
        
        Args:
            risk_scores: Risk scores from main analysis
            true_values: Ground truth values if available
        """
        
        if not TECHNICAL_DASHBOARD_AVAILABLE or risk_scores is None:
            return
        
        st.sidebar.markdown("### 🔬 Technical Metrics")
        
        try:
            # Quick validation for sidebar metrics
            validator = StatisticalValidationEngine()
            results = validator.validate_risk_scores(risk_scores, true_values)
            
            # Display key metrics
            quality_score = self._calculate_quality_score(results.data_quality)
            st.sidebar.metric(
                "Data Quality",
                f"{quality_score:.0f}%",
                delta="✅" if quality_score >= 80 else "⚠️"
            )
            
            if results.model_performance and 'r2_score' in results.model_performance:
                r2_score = results.model_performance['r2_score']
                st.sidebar.metric(
                    "Model R²",
                    f"{r2_score:.3f}",
                    delta="✅" if r2_score >= 0.8 else "⚠️"
                )
            
            # Statistical confidence
            confidence_score = self._calculate_confidence_score(results.p_values)
            confidence_label = "High" if confidence_score >= 70 else "Medium" if confidence_score >= 40 else "Low"
            st.sidebar.metric(
                "Statistical Confidence",
                confidence_label,
                delta="✅" if confidence_score >= 70 else "⚠️"
            )
            
        except Exception as e:
            st.sidebar.error(f"Technical metrics error: {str(e)[:50]}...")
    
    def get_technical_summary_for_export(self, 
                                       risk_scores: np.ndarray,
                                       true_values: Optional[np.ndarray] = None,
                                       features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate technical summary for export/reporting
        
        Args:
            risk_scores: Risk scores from analysis
            true_values: Ground truth values if available
            features: Feature matrix if available
            
        Returns:
            Dictionary containing technical summary metrics
        """
        
        if not TECHNICAL_DASHBOARD_AVAILABLE:
            return {"error": "Technical dashboard not available"}
        
        try:
            validator = StatisticalValidationEngine()
            results = validator.validate_risk_scores(risk_scores, true_values, features)
            summary = validator.generate_statistical_summary(results)
            
            return {
                "technical_validation": {
                    "data_quality_score": summary.get('data_quality_score', 0),
                    "statistical_confidence": summary.get('statistical_confidence', 'unknown'),
                    "key_findings": summary.get('key_findings', []),
                    "recommendations": summary.get('recommendations', []),
                    "model_performance": {
                        "r2_score": results.model_performance.get('r2_score', None),
                        "rmse": results.model_performance.get('rmse', None),
                        "mae": results.model_performance.get('mae', None)
                    } if results.model_performance else None,
                    "confidence_intervals": results.confidence_intervals,
                    "effect_sizes": results.effect_sizes
                }
            }
            
        except Exception as e:
            return {"error": f"Technical validation failed: {str(e)}"}
    
    def _display_integration_summary(self, 
                                   risk_scores: np.ndarray,
                                   true_values: Optional[np.ndarray],
                                   features: Optional[pd.DataFrame],
                                   data_source: str) -> None:
        """Display summary of data for integration"""
        
        st.markdown("### 📊 Data Summary for Technical Validation")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Risk Scores", len(risk_scores))
        
        with col2:
            st.metric("True Values", len(true_values) if true_values is not None else "N/A")
        
        with col3:
            st.metric("Features", f"{features.shape[1]} cols" if features is not None else "N/A")
        
        with col4:
            st.metric("Data Source", data_source)
        
        # Quick data quality preview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Risk Scores")
            st.write(f"Mean: {np.mean(risk_scores):.4f}")
            st.write(f"Std: {np.std(risk_scores):.4f}")
            st.write(f"Range: [{np.min(risk_scores):.4f}, {np.max(risk_scores):.4f}]")
        
        with col2:
            if true_values is not None:
                st.markdown("#### 🎯 Ground Truth")
                st.write(f"Mean: {np.mean(true_values):.4f}")
                st.write(f"Std: {np.std(true_values):.4f}")
                st.write(f"Correlation: {np.corrcoef(risk_scores, true_values)[0,1]:.4f}")
            else:
                st.markdown("#### ⚠️ No Ground Truth")
                st.info("Some validation metrics will be limited without ground truth values.")
    
    def _run_integrated_validation(self, 
                                 risk_scores: np.ndarray,
                                 true_values: Optional[np.ndarray],
                                 features: Optional[pd.DataFrame],
                                 data_source: str,
                                 confidence_level: float) -> None:
        """Run technical validation in integrated mode"""
        
        # Configure validation engine
        if self.dashboard:
            self.dashboard.validation_engine.confidence_level = confidence_level
        
        # Run the full technical dashboard
        with st.spinner("Running comprehensive technical validation..."):
            try:
                self.dashboard.render_dashboard(
                    risk_scores=risk_scores,
                    true_values=true_values,
                    features=features,
                    data_source=data_source
                )
                
                st.success("✅ Technical validation completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Technical validation failed: {str(e)}")
                st.info("Please check the data format and try again.")
    
    def _generate_demo_data(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Generate demo data for testing integration"""
        
        np.random.seed(42)
        n_samples = 300
        
        # Create banking-style features
        features = pd.DataFrame({
            'loan_to_value': np.random.beta(2, 3, n_samples) * 100,
            'debt_to_income': np.random.gamma(2, 10, n_samples),
            'credit_score': np.random.normal(650, 100, n_samples),
            'employment_years': np.random.exponential(5, n_samples)
        })
        
        # Create realistic risk model
        features_norm = (features - features.mean()) / features.std()
        coefficients = np.array([0.3, 0.4, -0.5, -0.2])
        
        # True and predicted values
        linear_combination = features_norm.values @ coefficients
        true_values = 1 / (1 + np.exp(-linear_combination))
        
        prediction_error = np.random.normal(0, 0.15, n_samples)
        risk_scores = 1 / (1 + np.exp(-(linear_combination + prediction_error)))
        
        return risk_scores, true_values, features
    
    def _calculate_quality_score(self, data_quality: Dict[str, Any]) -> float:
        """Calculate overall data quality score"""
        if not data_quality:
            return 0
        
        score = 100
        missing_pct = data_quality.get('missing_percentage', 0)
        outlier_pct = data_quality.get('outliers_percentage', 0)
        
        if missing_pct > 10:
            score -= 30
        elif missing_pct > 5:
            score -= 15
        
        if outlier_pct > 10:
            score -= 20
        elif outlier_pct > 5:
            score -= 10
        
        return max(0, score)
    
    def _calculate_confidence_score(self, p_values: Dict[str, float]) -> float:
        """Calculate statistical confidence score"""
        if not p_values:
            return 0
        
        significant_tests = sum(1 for p in p_values.values() 
                              if isinstance(p, float) and p < 0.05)
        total_tests = sum(1 for p in p_values.values() if isinstance(p, float))
        
        return (significant_tests / total_tests * 100) if total_tests > 0 else 0

# Integration helper functions for main dashboard

def add_technical_validation_tab(risk_scores: Optional[np.ndarray] = None,
                               true_values: Optional[np.ndarray] = None,
                               features: Optional[pd.DataFrame] = None,
                               data_source: str = "Main Dashboard") -> None:
    """
    Add technical validation tab to existing dashboard
    
    Usage in main dashboard:
        with tab_technical:
            add_technical_validation_tab(risk_scores, true_values, features)
    """
    integration = TechnicalDashboardIntegration()
    integration.render_technical_validation_tab(risk_scores, true_values, features, data_source)

def add_technical_sidebar_metrics(risk_scores: Optional[np.ndarray] = None,
                                true_values: Optional[np.ndarray] = None) -> None:
    """
    Add technical metrics to sidebar
    
    Usage in main dashboard sidebar:
        add_technical_sidebar_metrics(risk_scores, true_values)
    """
    integration = TechnicalDashboardIntegration()
    integration.render_technical_sidebar_metrics(risk_scores, true_values)

def get_technical_export_data(risk_scores: np.ndarray,
                            true_values: Optional[np.ndarray] = None,
                            features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Get technical validation data for export
    
    Usage for reports:
        technical_data = get_technical_export_data(risk_scores, true_values, features)
    """
    integration = TechnicalDashboardIntegration()
    return integration.get_technical_summary_for_export(risk_scores, true_values, features)

# Example integration code for main dashboard
def example_integration_usage():
    """
    Example of how to integrate technical dashboard into main BoE dashboard
    """
    
    # In main dashboard file, add this to your tabs:
    """
    tab1, tab2, tab3, tab_technical = st.tabs([
        "Risk Analysis", 
        "Supervisor Dashboard", 
        "Reports",
        "🔬 Technical Validation"
    ])
    
    with tab1:
        # Existing risk analysis code
        risk_scores = run_risk_analysis()
        
    with tab2:
        # Existing supervisor dashboard code
        pass
        
    with tab3:
        # Existing reports code
        pass
        
    with tab_technical:
        # NEW: Technical validation integration
        add_technical_validation_tab(
            risk_scores=risk_scores,
            true_values=historical_outcomes,  # if available
            features=input_features,          # if available
            data_source="BoE Risk Analysis"
        )
    
    # In sidebar, add technical metrics:
    with st.sidebar:
        # Existing sidebar content
        add_technical_sidebar_metrics(risk_scores, historical_outcomes)
    """

if __name__ == "__main__":
    # Test the integration component
    st.set_page_config(
        page_title="Technical Dashboard Integration Test",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Technical Dashboard Integration Test")
    st.markdown("Testing integration component for main BoE package (port 8505)")
    
    # Test the integration
    integration = TechnicalDashboardIntegration()
    integration.render_technical_validation_tab()