"""
Simple Test Script for Statistical Validation Components

Tests the comprehensive statistical validation engine without Unicode characters
for Windows compatibility.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import logging

# Add project paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "scripts"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_statistical_validation_basic():
    """Basic test of statistical validation engine"""
    
    print("Testing Statistical Validation Engine...")
    
    try:
        from scripts.statistical_validation import StatisticalValidationEngine
        
        # Generate test data
        np.random.seed(42)
        n_samples = 500
        
        # Create test data
        true_scores = np.random.beta(2, 5, n_samples)
        noise = np.random.normal(0, 0.1, n_samples)
        predicted_scores = true_scores + noise
        
        # Create feature matrix
        features = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.exponential(1, n_samples),
            'feature_3': np.random.uniform(-1, 1, n_samples)
        })
        
        # Initialize validation engine
        validator = StatisticalValidationEngine(confidence_level=0.95, random_state=42)
        
        # Run validation
        print("  Running statistical validation...")
        results = validator.validate_risk_scores(
            risk_scores=predicted_scores,
            true_values=true_scores,
            features=features
        )
        
        # Check results
        assert results.confidence_intervals is not None
        assert results.p_values is not None
        assert results.model_performance is not None
        assert results.data_quality is not None
        
        print("  PASSED: Confidence intervals calculated:", len(results.confidence_intervals))
        print("  PASSED: P-values calculated:", len(results.p_values))
        print("  PASSED: Model performance metrics:", len(results.model_performance))
        print("  PASSED: Data quality metrics:", len(results.data_quality))
        
        # Test summary generation
        summary = validator.generate_statistical_summary(results)
        print("  PASSED: Statistical summary generated")
        print("  Data Quality Score:", summary['data_quality_score'])
        print("  Statistical Confidence:", summary['statistical_confidence'])
        
        return True
        
    except Exception as e:
        print("  FAILED: Error in statistical validation:", str(e))
        return False

def test_visualizations_basic():
    """Basic test of visualization components"""
    
    print("\nTesting Technical Visualizations...")
    
    try:
        from scripts.statistical_validation import (
            StatisticalValidationEngine, 
            TechnicalVisualizationEngine
        )
        
        # Generate test data
        np.random.seed(42)
        n_samples = 300
        
        risk_scores = np.random.beta(2, 5, n_samples)
        true_values = risk_scores + np.random.normal(0, 0.1, n_samples)
        
        # Run validation
        validator = StatisticalValidationEngine()
        results = validator.validate_risk_scores(risk_scores, true_values)
        
        # Test visualizations
        viz_engine = TechnicalVisualizationEngine()
        
        # Test each visualization
        summary_fig = viz_engine.create_statistical_summary_cards(results)
        assert summary_fig is not None
        print("  PASSED: Statistical summary cards created")
        
        ci_fig = viz_engine.create_confidence_interval_plot(results)
        assert ci_fig is not None
        print("  PASSED: Confidence interval plot created")
        
        pvalue_fig = viz_engine.create_pvalue_heatmap(results)
        assert pvalue_fig is not None
        print("  PASSED: P-value heatmap created")
        
        diag_fig = viz_engine.create_model_diagnostics_plot(risk_scores, true_values)
        assert diag_fig is not None
        print("  PASSED: Model diagnostics plot created")
        
        quality_fig = viz_engine.create_data_quality_assessment(results)
        assert quality_fig is not None
        print("  PASSED: Data quality assessment created")
        
        return True
        
    except Exception as e:
        print("  FAILED: Error in visualizations:", str(e))
        return False

def test_dashboard_basic():
    """Basic test of dashboard components"""
    
    print("\nTesting Technical Dashboard...")
    
    try:
        from scripts.statistical_validation import TechnicalDashboard
        
        # Initialize dashboard
        dashboard = TechnicalDashboard()
        
        # Test initialization
        assert dashboard.validation_engine is not None
        assert dashboard.viz_engine is not None
        print("  PASSED: Dashboard initialized successfully")
        
        # Test helper methods
        test_data_quality = {'missing_percentage': 5, 'outliers_percentage': 8}
        quality_score = dashboard._calculate_quality_score(test_data_quality)
        assert 0 <= quality_score <= 100
        print("  PASSED: Quality score calculation:", quality_score)
        
        test_p_values = {'test1': 0.03, 'test2': 0.08, 'test3': 0.15}
        confidence_score = dashboard._calculate_confidence_score(test_p_values)
        assert 0 <= confidence_score <= 100
        print("  PASSED: Confidence score calculation:", confidence_score)
        
        return True
        
    except Exception as e:
        print("  FAILED: Error in dashboard:", str(e))
        return False

def test_integration_basic():
    """Basic integration test"""
    
    print("\nTesting Component Integration...")
    
    try:
        from scripts.statistical_validation import (
            StatisticalValidationEngine,
            TechnicalVisualizationEngine, 
            TechnicalDashboard
        )
        
        # Generate realistic test data
        np.random.seed(42)
        n_samples = 400
        
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
        
        print("  Generated realistic banking dataset")
        print("  Samples:", n_samples)
        print("  Features:", features.shape[1])
        print("  Risk range: [{:.3f}, {:.3f}]".format(np.min(risk_scores), np.max(risk_scores)))
        
        # Test full pipeline
        print("  Running full validation pipeline...")
        
        # Statistical validation
        validator = StatisticalValidationEngine(confidence_level=0.95)
        results = validator.validate_risk_scores(risk_scores, true_values, features)
        
        # Visualizations
        viz_engine = TechnicalVisualizationEngine()
        summary_fig = viz_engine.create_statistical_summary_cards(results)
        ci_fig = viz_engine.create_confidence_interval_plot(results)
        
        # Summary
        summary = validator.generate_statistical_summary(results)
        
        print("  PASSED: Full pipeline completed")
        print("  Data Quality Score:", summary['data_quality_score'])
        print("  Statistical Confidence:", summary['statistical_confidence'])
        print("  Model R-squared:", results.model_performance.get('r2_score', 0))
        print("  Key Findings:", len(summary['key_findings']))
        
        # Dashboard integration
        dashboard = TechnicalDashboard()
        print("  PASSED: Dashboard integration ready")
        
        return True
        
    except Exception as e:
        print("  FAILED: Error in integration:", str(e))
        return False

def main():
    """Run all tests"""
    
    print("Starting Statistical Validation Component Tests")
    print("=" * 50)
    
    tests = [
        ("Statistical Validation Engine", test_statistical_validation_basic),
        ("Technical Visualizations", test_visualizations_basic),
        ("Technical Dashboard", test_dashboard_basic),
        ("Component Integration", test_integration_basic)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  FAILED: Unexpected error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nSUCCESS: All tests passed!")
        print("Statistical validation components are ready for production.")
        print("\nTo launch technical dashboard:")
        print("  streamlit run technical_dashboard_launcher.py --server.port 8510")
    else:
        print("\nWARNING: Some tests failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)