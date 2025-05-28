"""
Test Script for Statistical Validation Components

Tests the comprehensive statistical validation engine, technical visualizations,
and dashboard components for the Bank of England Mosaic Lens project.
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

def test_statistical_validation_engine():
    """Test the statistical validation engine"""
    
    print("🔬 Testing Statistical Validation Engine...")
    
    try:
        from scripts.statistical_validation import StatisticalValidationEngine
        
        # Generate test data
        np.random.seed(42)
        n_samples = 1000
        
        # Create realistic risk assessment data
        true_scores = np.random.beta(2, 5, n_samples)  # True risk scores
        noise = np.random.normal(0, 0.1, n_samples)
        predicted_scores = true_scores + noise  # Predicted scores with noise
        
        # Create feature matrix
        features = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.exponential(1, n_samples),
            'feature_3': np.random.uniform(-1, 1, n_samples),
            'feature_4': np.random.gamma(2, 1, n_samples)
        })
        
        # Initialize validation engine
        validator = StatisticalValidationEngine(confidence_level=0.95, random_state=42)
        
        # Run comprehensive validation
        print("  📊 Running comprehensive statistical validation...")
        results = validator.validate_risk_scores(
            risk_scores=predicted_scores,
            true_values=true_scores,
            features=features
        )
        
        # Test results
        assert results.confidence_intervals is not None, "Confidence intervals should be calculated"
        assert results.p_values is not None, "P-values should be calculated"
        assert results.effect_sizes is not None, "Effect sizes should be calculated"
        assert results.model_performance is not None, "Model performance should be calculated"
        assert results.data_quality is not None, "Data quality should be assessed"
        
        print("  ✅ Confidence intervals calculated:", len(results.confidence_intervals))
        print("  ✅ P-values calculated:", len(results.p_values))
        print("  ✅ Effect sizes calculated:", len(results.effect_sizes))
        print("  ✅ Model performance metrics:", len(results.model_performance))
        print("  ✅ Data quality metrics:", len(results.data_quality))
        
        # Test statistical summary
        summary = validator.generate_statistical_summary(results)
        assert 'data_quality_score' in summary, "Summary should include data quality score"
        assert 'key_findings' in summary, "Summary should include key findings"
        
        print("  ✅ Statistical summary generated successfully")
        print(f"  📈 Data Quality Score: {summary['data_quality_score']:.1f}%")
        print(f"  🎯 Statistical Confidence: {summary['statistical_confidence']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing statistical validation engine: {e}")
        logger.error(f"Statistical validation engine test failed: {e}", exc_info=True)
        return False

def test_technical_visualizations():
    """Test the technical visualization engine"""
    
    print("\n📊 Testing Technical Visualization Engine...")
    
    try:
        from scripts.statistical_validation import (
            StatisticalValidationEngine, 
            TechnicalVisualizationEngine
        )
        
        # Generate test data
        np.random.seed(42)
        n_samples = 500
        
        risk_scores = np.random.beta(2, 5, n_samples)
        true_values = risk_scores + np.random.normal(0, 0.1, n_samples)
        
        # Run validation to get results
        validator = StatisticalValidationEngine()
        results = validator.validate_risk_scores(risk_scores, true_values)
        
        # Initialize visualization engine
        viz_engine = TechnicalVisualizationEngine()
        
        # Test statistical summary cards
        print("  📊 Testing statistical summary cards...")
        summary_fig = viz_engine.create_statistical_summary_cards(results)
        assert summary_fig is not None, "Summary cards should be created"
        print("  ✅ Statistical summary cards created")
        
        # Test confidence interval plot
        print("  📈 Testing confidence interval plot...")
        ci_fig = viz_engine.create_confidence_interval_plot(results)
        assert ci_fig is not None, "Confidence interval plot should be created"
        print("  ✅ Confidence interval plot created")
        
        # Test p-value heatmap
        print("  🎯 Testing p-value heatmap...")
        pvalue_fig = viz_engine.create_pvalue_heatmap(results)
        assert pvalue_fig is not None, "P-value heatmap should be created"
        print("  ✅ P-value heatmap created")
        
        # Test model diagnostics
        print("  🔍 Testing model diagnostics plot...")
        diag_fig = viz_engine.create_model_diagnostics_plot(risk_scores, true_values)
        assert diag_fig is not None, "Model diagnostics plot should be created"
        print("  ✅ Model diagnostics plot created")
        
        # Test data quality assessment
        print("  📋 Testing data quality assessment...")
        quality_fig = viz_engine.create_data_quality_assessment(results)
        assert quality_fig is not None, "Data quality assessment should be created"
        print("  ✅ Data quality assessment created")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing technical visualizations: {e}")
        logger.error(f"Technical visualizations test failed: {e}", exc_info=True)
        return False

def test_technical_dashboard():
    """Test the technical dashboard components"""
    
    print("\n🔬 Testing Technical Dashboard...")
    
    try:
        from scripts.statistical_validation import TechnicalDashboard
        
        # Initialize dashboard
        dashboard = TechnicalDashboard()
        
        # Test initialization
        assert dashboard.validation_engine is not None, "Validation engine should be initialized"
        assert dashboard.viz_engine is not None, "Visualization engine should be initialized"
        
        print("  ✅ Technical dashboard initialized successfully")
        print("  ✅ Validation engine available")
        print("  ✅ Visualization engine available")
        
        # Test helper methods
        test_data_quality = {'missing_percentage': 5, 'outliers_percentage': 8}
        quality_score = dashboard._calculate_quality_score(test_data_quality)
        assert 0 <= quality_score <= 100, "Quality score should be between 0 and 100"
        print(f"  ✅ Quality score calculation: {quality_score:.1f}%")
        
        test_p_values = {'test1': 0.03, 'test2': 0.08, 'test3': 0.15}
        confidence_score = dashboard._calculate_confidence_score(test_p_values)
        assert 0 <= confidence_score <= 100, "Confidence score should be between 0 and 100"
        print(f"  ✅ Confidence score calculation: {confidence_score:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing technical dashboard: {e}")
        logger.error(f"Technical dashboard test failed: {e}", exc_info=True)
        return False

def test_integration():
    """Test integration between all components"""
    
    print("\n🔗 Testing Component Integration...")
    
    try:
        from scripts.statistical_validation import (
            StatisticalValidationEngine,
            TechnicalVisualizationEngine, 
            TechnicalDashboard,
            StatisticalResults
        )
        
        # Generate comprehensive test dataset
        np.random.seed(42)
        n_samples = 800
        n_features = 6
        
        # Create realistic banking features
        features = pd.DataFrame({
            'loan_to_value': np.random.beta(2, 3, n_samples) * 100,
            'debt_to_income': np.random.gamma(2, 10, n_samples),
            'credit_score': np.random.normal(650, 100, n_samples),
            'employment_years': np.random.exponential(5, n_samples),
            'property_value': np.random.lognormal(12, 0.5, n_samples),
            'loan_amount': np.random.lognormal(11, 0.7, n_samples)
        })
        
        # Create realistic risk model
        features_norm = (features - features.mean()) / features.std()
        coefficients = np.array([0.3, 0.4, -0.5, -0.2, -0.1, 0.3])
        
        # True risk scores
        linear_combination = features_norm.values @ coefficients
        true_values = 1 / (1 + np.exp(-linear_combination))  # Sigmoid
        
        # Predicted risk scores with realistic error
        prediction_error = np.random.normal(0, 0.15, n_samples)
        risk_scores = 1 / (1 + np.exp(-(linear_combination + prediction_error)))
        
        print(f"  📊 Generated dataset: {n_samples} samples, {n_features} features")
        print(f"  🎯 True risk range: [{np.min(true_values):.3f}, {np.max(true_values):.3f}]")
        print(f"  📈 Predicted risk range: [{np.min(risk_scores):.3f}, {np.max(risk_scores):.3f}]")
        
        # Test full pipeline
        print("  🔄 Running full validation pipeline...")
        
        # 1. Statistical validation
        validator = StatisticalValidationEngine(confidence_level=0.95)
        results = validator.validate_risk_scores(risk_scores, true_values, features)
        
        # 2. Generate visualizations
        viz_engine = TechnicalVisualizationEngine()
        
        # Create all visualizations
        summary_fig = viz_engine.create_statistical_summary_cards(results)
        ci_fig = viz_engine.create_confidence_interval_plot(results)
        pvalue_fig = viz_engine.create_pvalue_heatmap(results)
        diag_fig = viz_engine.create_model_diagnostics_plot(risk_scores, true_values)
        quality_fig = viz_engine.create_data_quality_assessment(results)
        
        # 3. Generate comprehensive summary
        summary = validator.generate_statistical_summary(results)
        
        print("  ✅ Full pipeline completed successfully")
        print(f"  📊 Data Quality Score: {summary['data_quality_score']:.1f}%")
        print(f"  🎯 Statistical Confidence: {summary['statistical_confidence']}")
        print(f"  📈 Model R²: {results.model_performance.get('r2_score', 0):.3f}")
        print(f"  🔍 Key Findings: {len(summary['key_findings'])}")
        print(f"  💡 Recommendations: {len(summary['recommendations'])}")
        
        # Test dashboard integration
        dashboard = TechnicalDashboard()
        print("  ✅ Dashboard integration ready")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error in integration test: {e}")
        logger.error(f"Integration test failed: {e}", exc_info=True)
        return False

def run_performance_benchmark():
    """Run performance benchmark for statistical validation"""
    
    print("\n⚡ Running Performance Benchmark...")
    
    try:
        import time
        from scripts.statistical_validation import StatisticalValidationEngine
        
        # Test different data sizes
        test_sizes = [100, 500, 1000, 2000]
        
        for n_samples in test_sizes:
            print(f"  📊 Testing with {n_samples} samples...")
            
            # Generate test data
            np.random.seed(42)
            risk_scores = np.random.beta(2, 5, n_samples)
            true_values = risk_scores + np.random.normal(0, 0.1, n_samples)
            features = pd.DataFrame(np.random.randn(n_samples, 5))
            
            # Time the validation
            start_time = time.time()
            
            validator = StatisticalValidationEngine()
            results = validator.validate_risk_scores(risk_scores, true_values, features)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"    ⏱️  Processing time: {duration:.2f} seconds")
            print(f"    📈 Samples per second: {n_samples/duration:.0f}")
            
            # Memory efficiency check
            import sys
            memory_usage = sys.getsizeof(results) / 1024  # KB
            print(f"    💾 Memory usage: {memory_usage:.1f} KB")
        
        print("  ✅ Performance benchmark completed")
        return True
        
    except Exception as e:
        print(f"  ❌ Error in performance benchmark: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🚀 Starting Statistical Validation Component Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run individual component tests
    test_results.append(("Statistical Validation Engine", test_statistical_validation_engine()))
    test_results.append(("Technical Visualizations", test_technical_visualizations()))
    test_results.append(("Technical Dashboard", test_technical_dashboard()))
    test_results.append(("Component Integration", test_integration()))
    test_results.append(("Performance Benchmark", run_performance_benchmark()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Statistical validation components are ready for production.")
        print("\n🚀 Ready to launch technical dashboard:")
        print("   streamlit run technical_dashboard_launcher.py --server.port 8510")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)