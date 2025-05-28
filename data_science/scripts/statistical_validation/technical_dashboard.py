"""
Technical Data Science Dashboard for Bank of England Mosaic Lens

Comprehensive statistical validation dashboard for supervisors with advanced
analytics, confidence scoring, p-value reporting, and model diagnostics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import json

from .statistical_validation_engine import StatisticalValidationEngine, StatisticalResults
from .technical_visualizations import TechnicalVisualizationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalDashboard:
    """
    Technical Data Science Dashboard for statistical validation and model diagnostics
    
    Features:
    - Comprehensive statistical validation
    - Interactive technical visualizations
    - Model performance diagnostics
    - Data quality assessment
    - Confidence scoring and p-value reporting
    - Export capabilities for technical reports
    """
    
    def __init__(self):
        """Initialize the technical dashboard"""
        self.validation_engine = StatisticalValidationEngine()
        self.viz_engine = TechnicalVisualizationEngine()
        
        # Initialize session state for caching
        if 'validation_results' not in st.session_state:
            st.session_state.validation_results = None
        if 'technical_data' not in st.session_state:
            st.session_state.technical_data = None
    
    def render_dashboard(self, 
                        risk_scores: np.ndarray,
                        true_values: Optional[np.ndarray] = None,
                        features: Optional[pd.DataFrame] = None,
                        data_source: str = "Unknown") -> None:
        """
        Render the complete technical dashboard
        
        Args:
            risk_scores: Array of risk scores to validate
            true_values: Optional ground truth values
            features: Optional feature matrix
            data_source: Name of the data source
        """
        
        # Dashboard header
        st.markdown("## 🔬 Technical Data Science Dashboard")
        st.markdown("*Advanced Statistical Validation & Model Diagnostics*")
        
        # Create columns for key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Perform statistical validation
        with st.spinner("Performing comprehensive statistical validation..."):
            validation_results = self.validation_engine.validate_risk_scores(
                risk_scores=risk_scores,
                true_values=true_values,
                features=features
            )
            st.session_state.validation_results = validation_results
        
        # Display key metrics in header
        self._display_key_metrics(validation_results, col1, col2, col3, col4)
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Statistical Summary",
            "📈 Confidence Intervals", 
            "🎯 Hypothesis Testing",
            "🔍 Model Diagnostics",
            "📋 Data Quality",
            "📄 Technical Report"
        ])
        
        with tab1:
            self._render_statistical_summary_tab(validation_results)
        
        with tab2:
            self._render_confidence_intervals_tab(validation_results)
        
        with tab3:
            self._render_hypothesis_testing_tab(validation_results)
        
        with tab4:
            self._render_model_diagnostics_tab(validation_results, risk_scores, true_values)
        
        with tab5:
            self._render_data_quality_tab(validation_results)
        
        with tab6:
            self._render_technical_report_tab(validation_results, data_source)
    
    def _display_key_metrics(self, 
                           results: StatisticalResults, 
                           col1, col2, col3, col4) -> None:
        """Display key metrics in the header"""
        
        # Data Quality Score
        quality_score = self._calculate_quality_score(results.data_quality)
        with col1:
            st.metric(
                label="Data Quality Score",
                value=f"{quality_score:.0f}%",
                delta=f"{'✅' if quality_score >= 80 else '⚠️' if quality_score >= 60 else '❌'}"
            )
        
        # Statistical Confidence
        confidence_score = self._calculate_confidence_score(results.p_values)
        with col2:
            st.metric(
                label="Statistical Confidence",
                value=f"{confidence_score:.0f}%",
                delta=f"{'High' if confidence_score >= 70 else 'Medium' if confidence_score >= 40 else 'Low'}"
            )
        
        # Model Performance (if available)
        r2_score = results.model_performance.get('r2_score', 0)
        with col3:
            st.metric(
                label="Model R² Score",
                value=f"{r2_score:.3f}",
                delta=f"{'Excellent' if r2_score >= 0.9 else 'Good' if r2_score >= 0.8 else 'Fair' if r2_score >= 0.6 else 'Poor'}"
            )
        
        # Effect Size Summary
        max_effect = max([abs(v) for v in results.effect_sizes.values()]) if results.effect_sizes else 0
        with col4:
            st.metric(
                label="Max Effect Size",
                value=f"{max_effect:.3f}",
                delta=f"{'Large' if max_effect >= 0.8 else 'Medium' if max_effect >= 0.5 else 'Small' if max_effect >= 0.2 else 'Negligible'}"
            )
    
    def _render_statistical_summary_tab(self, results: StatisticalResults) -> None:
        """Render the statistical summary tab"""
        
        st.markdown("### 📊 Statistical Validation Summary")
        
        # Create statistical summary cards
        summary_fig = self.viz_engine.create_statistical_summary_cards(results)
        st.plotly_chart(summary_fig, use_container_width=True)
        
        # Detailed statistical summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Descriptive Statistics")
            if results.confidence_intervals:
                stats_data = []
                for metric, (lower, upper) in results.confidence_intervals.items():
                    mean_val = (lower + upper) / 2
                    margin_error = (upper - lower) / 2
                    stats_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Mean': f"{mean_val:.4f}",
                        'CI Lower': f"{lower:.4f}",
                        'CI Upper': f"{upper:.4f}",
                        'Margin of Error': f"{margin_error:.4f}"
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Effect Sizes")
            if results.effect_sizes:
                effect_data = []
                for effect_name, effect_value in results.effect_sizes.items():
                    significance = results.practical_significance.get(effect_name, 'unknown')
                    effect_data.append({
                        'Effect Type': effect_name.replace('_', ' ').title(),
                        'Effect Size': f"{effect_value:.4f}",
                        'Magnitude': significance.title(),
                        'Interpretation': self._interpret_effect_size(abs(effect_value))
                    })
                
                effect_df = pd.DataFrame(effect_data)
                st.dataframe(effect_df, use_container_width=True)
        
        # Statistical interpretation
        st.markdown("#### 🔍 Statistical Interpretation")
        interpretation = self._generate_statistical_interpretation(results)
        st.info(interpretation)
    
    def _render_confidence_intervals_tab(self, results: StatisticalResults) -> None:
        """Render the confidence intervals tab"""
        
        st.markdown("### 📈 Confidence Interval Analysis")
        
        if not results.confidence_intervals:
            st.warning("No confidence interval data available.")
            return
        
        # Create confidence interval plot
        ci_fig = self.viz_engine.create_confidence_interval_plot(results)
        st.plotly_chart(ci_fig, use_container_width=True)
        
        # Confidence interval interpretation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Bootstrap vs Parametric CIs")
            ci_comparison = self._compare_confidence_intervals(results.confidence_intervals)
            st.write(ci_comparison)
        
        with col2:
            st.markdown("#### 🎯 Precision Analysis")
            precision_analysis = self._analyze_precision(results.confidence_intervals)
            st.write(precision_analysis)
        
        # Detailed CI table
        st.markdown("#### 📋 Detailed Confidence Intervals")
        ci_data = []
        for metric, (lower, upper) in results.confidence_intervals.items():
            width = upper - lower
            center = (lower + upper) / 2
            ci_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Lower Bound': f"{lower:.6f}",
                'Upper Bound': f"{upper:.6f}",
                'Center': f"{center:.6f}",
                'Width': f"{width:.6f}",
                'Relative Width (%)': f"{(width/abs(center)*100) if center != 0 else 0:.2f}"
            })
        
        ci_df = pd.DataFrame(ci_data)
        st.dataframe(ci_df, use_container_width=True)
    
    def _render_hypothesis_testing_tab(self, results: StatisticalResults) -> None:
        """Render the hypothesis testing tab"""
        
        st.markdown("### 🎯 Hypothesis Testing & P-Value Analysis")
        
        if not results.p_values:
            st.warning("No hypothesis testing data available.")
            return
        
        # Create p-value heatmap
        pvalue_fig = self.viz_engine.create_pvalue_heatmap(results)
        st.plotly_chart(pvalue_fig, use_container_width=True)
        
        # Multiple testing correction
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔬 Hypothesis Test Results")
            test_data = []
            for test_name, p_value in results.p_values.items():
                if isinstance(p_value, float) and test_name != 'bonferroni_corrected_alpha':
                    significance = self._determine_significance(p_value)
                    test_data.append({
                        'Test': test_name.replace('_', ' ').title(),
                        'P-Value': f"{p_value:.6f}",
                        'Significance': significance,
                        'Result': '✅ Significant' if p_value < 0.05 else '❌ Not Significant'
                    })
            
            if test_data:
                test_df = pd.DataFrame(test_data)
                st.dataframe(test_df, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Multiple Testing Correction")
            if 'bonferroni_corrected_alpha' in results.p_values:
                corrected_alpha = results.p_values['bonferroni_corrected_alpha']
                significant_after_correction = results.p_values.get('significant_tests_after_correction', 0)
                
                st.metric("Original α", "0.05")
                st.metric("Bonferroni Corrected α", f"{corrected_alpha:.6f}")
                st.metric("Significant After Correction", f"{significant_after_correction}")
                
                # Interpretation
                if significant_after_correction > 0:
                    st.success(f"✅ {significant_after_correction} test(s) remain significant after Bonferroni correction")
                else:
                    st.warning("⚠️ No tests remain significant after multiple testing correction")
        
        # Distribution tests
        if results.significance_tests:
            st.markdown("#### 📊 Distribution & Normality Tests")
            dist_data = []
            for test_name, test_result in results.significance_tests.items():
                if isinstance(test_result, dict):
                    dist_data.append({
                        'Test': test_name.replace('_', ' ').title(),
                        'Statistic': f"{test_result.get('statistic', 'N/A'):.6f}",
                        'P-Value': f"{test_result.get('p_value', 'N/A'):.6f}" if 'p_value' in test_result else 'N/A',
                        'Result': test_result.get('interpretation', 'N/A'),
                        'Normal?': '✅ Yes' if test_result.get('is_normal', False) else '❌ No'
                    })
            
            if dist_data:
                dist_df = pd.DataFrame(dist_data)
                st.dataframe(dist_df, use_container_width=True)
    
    def _render_model_diagnostics_tab(self, 
                                    results: StatisticalResults,
                                    risk_scores: np.ndarray,
                                    true_values: Optional[np.ndarray]) -> None:
        """Render the model diagnostics tab"""
        
        st.markdown("### 🔍 Model Performance & Diagnostics")
        
        if true_values is None:
            st.warning("Model diagnostics require ground truth values for comparison.")
            return
        
        # Create model diagnostic plots
        residuals = true_values - risk_scores
        diag_fig = self.viz_engine.create_model_diagnostics_plot(risk_scores, true_values, residuals)
        st.plotly_chart(diag_fig, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📈 Regression Metrics")
            if results.model_performance:
                perf_data = []
                regression_metrics = ['r2_score', 'mse', 'rmse', 'mae', 'mape']
                for metric in regression_metrics:
                    if metric in results.model_performance:
                        value = results.model_performance[metric]
                        perf_data.append({
                            'Metric': metric.upper().replace('_', ' '),
                            'Value': f"{value:.6f}",
                            'Quality': self._assess_metric_quality(metric, value)
                        })
                
                if perf_data:
                    perf_df = pd.DataFrame(perf_data)
                    st.dataframe(perf_df, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔗 Correlation Analysis")
            if results.model_performance:
                corr_data = []
                correlation_metrics = ['pearson_correlation', 'spearman_correlation']
                for metric in correlation_metrics:
                    if metric in results.model_performance:
                        value = results.model_performance[metric]
                        corr_data.append({
                            'Type': metric.replace('_correlation', '').title(),
                            'Correlation': f"{value:.6f}",
                            'Strength': self._assess_correlation_strength(abs(value))
                        })
                
                if corr_data:
                    corr_df = pd.DataFrame(corr_data)
                    st.dataframe(corr_df, use_container_width=True)
        
        with col3:
            st.markdown("#### 📊 Residual Analysis")
            if results.model_performance:
                residual_data = []
                residual_metrics = ['residual_mean', 'residual_std', 'residual_skewness', 'residual_kurtosis']
                for metric in residual_metrics:
                    if metric in results.model_performance:
                        value = results.model_performance[metric]
                        residual_data.append({
                            'Metric': metric.replace('residual_', '').title(),
                            'Value': f"{value:.6f}",
                            'Assessment': self._assess_residual_metric(metric, value)
                        })
                
                if residual_data:
                    residual_df = pd.DataFrame(residual_data)
                    st.dataframe(residual_df, use_container_width=True)
        
        # Cross-validation results
        if any('cv' in key for key in results.model_performance.keys()):
            st.markdown("#### 🔄 Cross-Validation Results")
            cv_data = []
            for key, value in results.model_performance.items():
                if 'cv' in key:
                    cv_data.append({
                        'Model': key.replace('_cv_mean', '').replace('_cv_std', '').replace('_', ' ').title(),
                        'Metric': 'Mean' if 'mean' in key else 'Std',
                        'Value': f"{value:.6f}"
                    })
            
            if cv_data:
                cv_df = pd.DataFrame(cv_data)
                st.dataframe(cv_df, use_container_width=True)
    
    def _render_data_quality_tab(self, results: StatisticalResults) -> None:
        """Render the data quality tab"""
        
        st.markdown("### 📋 Data Quality Assessment")
        
        if not results.data_quality:
            st.warning("No data quality information available.")
            return
        
        # Create data quality visualization
        quality_fig = self.viz_engine.create_data_quality_assessment(results)
        st.plotly_chart(quality_fig, use_container_width=True)
        
        # Detailed quality metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Data Completeness")
            completeness_data = []
            quality_metrics = ['missing_percentage', 'infinite_values', 'unique_values']
            for metric in quality_metrics:
                if metric in results.data_quality:
                    value = results.data_quality[metric]
                    completeness_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': f"{value:.2f}" + ("%" if 'percentage' in metric else ""),
                        'Status': self._assess_data_completeness(metric, value)
                    })
            
            if completeness_data:
                completeness_df = pd.DataFrame(completeness_data)
                st.dataframe(completeness_df, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Distribution Quality")
            distribution_data = []
            dist_metrics = ['skewness', 'kurtosis', 'coefficient_of_variation']
            for metric in dist_metrics:
                if metric in results.data_quality:
                    value = results.data_quality[metric]
                    distribution_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': f"{value:.6f}",
                        'Assessment': self._assess_distribution_quality(metric, value)
                    })
            
            if distribution_data:
                distribution_df = pd.DataFrame(distribution_data)
                st.dataframe(distribution_df, use_container_width=True)
        
        # Feature quality (if available)
        if 'feature_quality' in results.data_quality:
            st.markdown("#### 🔍 Feature Quality Analysis")
            feature_quality = results.data_quality['feature_quality']
            
            feature_data = []
            for feature_name, quality_metrics in feature_quality.items():
                feature_data.append({
                    'Feature': feature_name,
                    'Missing (%)': f"{quality_metrics.get('missing_percentage', 0):.2f}",
                    'Unique Values': quality_metrics.get('unique_values', 'N/A'),
                    'Data Type': quality_metrics.get('data_type', 'N/A'),
                    'Mean': f"{quality_metrics.get('mean', 0):.4f}" if 'mean' in quality_metrics else 'N/A',
                    'Std': f"{quality_metrics.get('std', 0):.4f}" if 'std' in quality_metrics else 'N/A',
                    'Outliers (%)': f"{quality_metrics.get('outliers_percentage', 0):.2f}" if 'outliers_percentage' in quality_metrics else 'N/A'
                })
            
            if feature_data:
                feature_df = pd.DataFrame(feature_data)
                st.dataframe(feature_df, use_container_width=True)
    
    def _render_technical_report_tab(self, results: StatisticalResults, data_source: str) -> None:
        """Render the technical report tab"""
        
        st.markdown("### 📄 Technical Statistical Report")
        
        # Generate comprehensive summary
        summary = self.validation_engine.generate_statistical_summary(results)
        
        # Report header
        st.markdown(f"""
        **Statistical Validation Report**  
        **Data Source:** {data_source}  
        **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
        **Analysis Type:** Comprehensive Statistical Validation
        """)
        
        # Executive summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Overall Assessment")
            st.metric("Data Quality Score", f"{summary.get('data_quality_score', 0):.0f}%")
            st.metric("Statistical Confidence", summary.get('statistical_confidence', 'Unknown').title())
            
            # Key findings
            st.markdown("#### 🔍 Key Findings")
            findings = summary.get('key_findings', [])
            if findings:
                for finding in findings:
                    st.write(f"• {finding}")
            else:
                st.write("No significant findings identified.")
        
        with col2:
            st.markdown("#### 💡 Recommendations")
            recommendations = summary.get('recommendations', [])
            if recommendations:
                for rec in recommendations:
                    st.write(f"• {rec}")
            else:
                st.success("✅ No immediate recommendations - data quality is satisfactory")
            
            # Statistical power assessment
            st.markdown("#### ⚡ Statistical Power")
            power_assessment = self._assess_statistical_power(results)
            st.write(power_assessment)
        
        # Detailed technical summary
        st.markdown("#### 📋 Detailed Technical Summary")
        
        # Create expandable sections for detailed results
        with st.expander("🔬 Statistical Test Results"):
            if results.p_values:
                st.json(results.p_values)
            else:
                st.write("No statistical test results available.")
        
        with st.expander("📈 Confidence Intervals"):
            if results.confidence_intervals:
                ci_dict = {k: {'lower': v[0], 'upper': v[1]} for k, v in results.confidence_intervals.items()}
                st.json(ci_dict)
            else:
                st.write("No confidence interval data available.")
        
        with st.expander("🎯 Effect Sizes"):
            if results.effect_sizes:
                st.json(results.effect_sizes)
            else:
                st.write("No effect size data available.")
        
        with st.expander("📊 Model Performance"):
            if results.model_performance:
                st.json(results.model_performance)
            else:
                st.write("No model performance data available.")
        
        # Export options
        st.markdown("#### 📤 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Summary JSON"):
                summary_json = json.dumps(summary, indent=2)
                st.download_button(
                    label="Download Summary",
                    data=summary_json,
                    file_name=f"statistical_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📈 Export Full Results"):
                full_results = {
                    'confidence_intervals': results.confidence_intervals,
                    'p_values': results.p_values,
                    'effect_sizes': results.effect_sizes,
                    'model_performance': results.model_performance,
                    'data_quality': results.data_quality
                }
                results_json = json.dumps(full_results, indent=2, default=str)
                st.download_button(
                    label="Download Full Results",
                    data=results_json,
                    file_name=f"full_statistical_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("📄 Generate PDF Report"):
                st.info("PDF generation feature coming soon...")
    
    # Helper methods for assessments and interpretations
    
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
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude"""
        if effect_size >= 0.8:
            return "Large practical effect"
        elif effect_size >= 0.5:
            return "Medium practical effect"
        elif effect_size >= 0.2:
            return "Small practical effect"
        else:
            return "Negligible effect"
    
    def _determine_significance(self, p_value: float) -> str:
        """Determine significance level"""
        if p_value < 0.001:
            return "Highly Significant (p < 0.001)"
        elif p_value < 0.01:
            return "Very Significant (p < 0.01)"
        elif p_value < 0.05:
            return "Significant (p < 0.05)"
        elif p_value < 0.1:
            return "Marginally Significant (p < 0.1)"
        else:
            return "Not Significant (p ≥ 0.1)"
    
    def _assess_metric_quality(self, metric: str, value: float) -> str:
        """Assess quality of performance metric"""
        if metric == 'r2_score':
            if value >= 0.9:
                return "Excellent"
            elif value >= 0.8:
                return "Good"
            elif value >= 0.6:
                return "Fair"
            else:
                return "Poor"
        elif metric in ['mse', 'rmse', 'mae']:
            return "Lower is better"
        elif metric == 'mape':
            if value <= 10:
                return "Excellent"
            elif value <= 20:
                return "Good"
            elif value <= 50:
                return "Fair"
            else:
                return "Poor"
        else:
            return "N/A"
    
    def _assess_correlation_strength(self, correlation: float) -> str:
        """Assess correlation strength"""
        if correlation >= 0.9:
            return "Very Strong"
        elif correlation >= 0.7:
            return "Strong"
        elif correlation >= 0.5:
            return "Moderate"
        elif correlation >= 0.3:
            return "Weak"
        else:
            return "Very Weak"
    
    def _assess_residual_metric(self, metric: str, value: float) -> str:
        """Assess residual metric quality"""
        if metric == 'residual_mean':
            return "Good" if abs(value) < 0.01 else "Check for bias"
        elif metric == 'residual_skewness':
            return "Normal" if abs(value) < 0.5 else "Skewed"
        elif metric == 'residual_kurtosis':
            return "Normal" if abs(value) < 1 else "Heavy tails"
        else:
            return "N/A"
    
    def _assess_data_completeness(self, metric: str, value: float) -> str:
        """Assess data completeness"""
        if metric == 'missing_percentage':
            if value == 0:
                return "✅ Perfect"
            elif value < 5:
                return "✅ Good"
            elif value < 10:
                return "⚠️ Fair"
            else:
                return "❌ Poor"
        elif metric == 'infinite_values':
            return "✅ Good" if value == 0 else "❌ Issues"
        else:
            return "N/A"
    
    def _assess_distribution_quality(self, metric: str, value: float) -> str:
        """Assess distribution quality"""
        if metric == 'skewness':
            if abs(value) < 0.5:
                return "✅ Normal"
            elif abs(value) < 1:
                return "⚠️ Moderate"
            else:
                return "❌ Highly skewed"
        elif metric == 'kurtosis':
            if abs(value) < 1:
                return "✅ Normal"
            elif abs(value) < 2:
                return "⚠️ Moderate"
            else:
                return "❌ Heavy tails"
        elif metric == 'coefficient_of_variation':
            if value < 0.1:
                return "✅ Low variability"
            elif value < 0.3:
                return "⚠️ Moderate variability"
            else:
                return "❌ High variability"
        else:
            return "N/A"
    
    def _compare_confidence_intervals(self, confidence_intervals: Dict[str, Tuple[float, float]]) -> str:
        """Compare bootstrap vs parametric confidence intervals"""
        bootstrap_cis = {k: v for k, v in confidence_intervals.items() if 'bootstrap' in k}
        parametric_cis = {k: v for k, v in confidence_intervals.items() if 'parametric' in k}
        
        if not bootstrap_cis or not parametric_cis:
            return "Insufficient data for comparison."
        
        # Compare widths
        bootstrap_widths = [upper - lower for lower, upper in bootstrap_cis.values()]
        parametric_widths = [upper - lower for lower, upper in parametric_cis.values()]
        
        avg_bootstrap_width = np.mean(bootstrap_widths)
        avg_parametric_width = np.mean(parametric_widths)
        
        if avg_bootstrap_width > avg_parametric_width:
            return "Bootstrap CIs are wider, suggesting non-normal distribution."
        else:
            return "Parametric CIs are similar to bootstrap, suggesting normal distribution."
    
    def _analyze_precision(self, confidence_intervals: Dict[str, Tuple[float, float]]) -> str:
        """Analyze precision of estimates"""
        if not confidence_intervals:
            return "No confidence intervals available."
        
        widths = [upper - lower for lower, upper in confidence_intervals.values()]
        avg_width = np.mean(widths)
        
        if avg_width < 0.01:
            return "✅ High precision estimates"
        elif avg_width < 0.1:
            return "⚠️ Moderate precision estimates"
        else:
            return "❌ Low precision estimates - consider more data"
    
    def _generate_statistical_interpretation(self, results: StatisticalResults) -> str:
        """Generate overall statistical interpretation"""
        interpretations = []
        
        # Data quality interpretation
        if results.data_quality:
            missing_pct = results.data_quality.get('missing_percentage', 0)
            if missing_pct > 10:
                interpretations.append(f"⚠️ High missing data rate ({missing_pct:.1f}%) may affect reliability.")
            else:
                interpretations.append(f"✅ Good data completeness ({100-missing_pct:.1f}% complete).")
        
        # Statistical significance interpretation
        if results.p_values:
            significant_tests = sum(1 for p in results.p_values.values() 
                                  if isinstance(p, float) and p < 0.05)
            total_tests = sum(1 for p in results.p_values.values() if isinstance(p, float))
            
            if significant_tests > 0:
                interpretations.append(f"✅ {significant_tests}/{total_tests} statistical tests show significance.")
            else:
                interpretations.append("⚠️ No statistically significant results detected.")
        
        # Effect size interpretation
        if results.effect_sizes:
            large_effects = sum(1 for effect in results.effect_sizes.values() if abs(effect) >= 0.8)
            if large_effects > 0:
                interpretations.append(f"🎯 {large_effects} large practical effect(s) detected.")
        
        # Model performance interpretation
        if results.model_performance:
            r2 = results.model_performance.get('r2_score', 0)
            if r2 >= 0.8:
                interpretations.append(f"📈 Excellent model performance (R² = {r2:.3f}).")
            elif r2 >= 0.6:
                interpretations.append(f"📊 Good model performance (R² = {r2:.3f}).")
            else:
                interpretations.append(f"⚠️ Model performance needs improvement (R² = {r2:.3f}).")
        
        return " ".join(interpretations) if interpretations else "Analysis completed successfully."
    
    def _assess_statistical_power(self, results: StatisticalResults) -> str:
        """Assess statistical power of the analysis"""
        if not results.p_values:
            return "Insufficient data for power analysis."
        
        significant_tests = sum(1 for p in results.p_values.values() 
                              if isinstance(p, float) and p < 0.05)
        total_tests = sum(1 for p in results.p_values.values() if isinstance(p, float))
        
        if total_tests == 0:
            return "No statistical tests performed."
        
        power_ratio = significant_tests / total_tests
        
        if power_ratio >= 0.8:
            return "✅ High statistical power - results are reliable."
        elif power_ratio >= 0.5:
            return "⚠️ Moderate statistical power - consider more data."
        else:
            return "❌ Low statistical power - increase sample size recommended."