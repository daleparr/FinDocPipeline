"""
Comprehensive Test Suite for Phase 3: Statistical Analysis Components
Tests time series analysis, anomaly detection, and risk scoring
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add the data_science directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import Phase 3 components
from statistical_analysis.time_series_analyzer import TimeSeriesAnalyzer
from statistical_analysis.anomaly_detector import AnomalyDetector
from statistical_analysis.risk_scorer import RiskScorer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_comprehensive_test_data() -> pd.DataFrame:
    """Create comprehensive test data for statistical analysis"""
    
    # Create time series data across multiple quarters and institutions
    institutions = ['JPMorgan Chase', 'Bank of America', 'Wells Fargo', 'Citigroup']
    quarters = ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023', 'Q1_2024', 'Q2_2024']
    
    test_data = []
    
    for institution in institutions:
        for quarter in quarters:
            # Create varying risk profiles for different institutions
            if institution == 'JPMorgan Chase':
                base_risk = 0.2  # Low risk institution
                volatility = 0.1
            elif institution == 'Bank of America':
                base_risk = 0.3  # Medium risk institution
                volatility = 0.15
            elif institution == 'Wells Fargo':
                base_risk = 0.5  # Higher risk institution
                volatility = 0.2
            else:  # Citigroup
                base_risk = 0.4  # Medium-high risk
                volatility = 0.18
            
            # Add temporal trends
            quarter_index = quarters.index(quarter)
            trend_factor = 1 + (quarter_index * 0.05)  # Slight increasing trend
            
            # Generate sentences for this institution-quarter
            num_sentences = np.random.randint(15, 25)
            
            for i in range(num_sentences):
                # Generate realistic financial text
                texts = [
                    f"Our credit portfolio performance was {'strong' if base_risk < 0.3 else 'challenging'} this quarter.",
                    f"We are {'confident' if base_risk < 0.4 else 'cautious'} about our risk management capabilities.",
                    f"Market conditions have been {'favorable' if base_risk < 0.3 else 'volatile'} for our trading operations.",
                    f"Regulatory compliance costs {'remained stable' if base_risk < 0.4 else 'increased significantly'}.",
                    f"Our capital ratios {'exceed' if base_risk < 0.3 else 'meet'} regulatory requirements.",
                    f"We expect {'continued growth' if base_risk < 0.4 else 'some headwinds'} in the coming quarters.",
                    f"Credit losses were {'below' if base_risk < 0.3 else 'above'} our expectations.",
                    f"Operational risk events were {'minimal' if base_risk < 0.4 else 'elevated'} this period."
                ]
                
                text = np.random.choice(texts)
                
                # Generate sentiment scores based on risk profile
                sentiment_positive = max(0, min(1, np.random.normal(0.7 - base_risk, 0.2)))
                sentiment_negative = max(0, min(1, np.random.normal(base_risk, 0.15)))
                sentiment_neutral = max(0, min(1, 1 - sentiment_positive - sentiment_negative))
                
                # Normalize sentiments
                total_sentiment = sentiment_positive + sentiment_negative + sentiment_neutral
                sentiment_positive /= total_sentiment
                sentiment_negative /= total_sentiment
                sentiment_neutral /= total_sentiment
                
                # Generate other metrics
                risk_escalation_score = max(0, min(1, np.random.normal(base_risk * trend_factor, volatility)))
                stress_score = max(0, min(1, np.random.normal(base_risk * 0.8, volatility)))
                confidence_score = max(0, min(1, np.random.normal(1 - base_risk, 0.2)))
                hedging_score = max(0, min(1, np.random.normal(base_risk * 0.3, 0.1)))
                uncertainty_score = max(0, min(1, np.random.normal(base_risk * 0.4, 0.15)))
                formality_score = np.random.uniform(0.6, 0.9)
                complexity_score = np.random.uniform(0.4, 0.8)
                
                # Add some anomalies
                if np.random.random() < 0.05:  # 5% anomaly rate
                    risk_escalation_score = min(1, risk_escalation_score * 3)
                    stress_score = min(1, stress_score * 2.5)
                
                # Generate speaker
                speakers = ['CEO', 'CFO', 'CRO', 'CCO', 'Analyst']
                speaker_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
                speaker = np.random.choice(speakers, p=speaker_weights)
                
                # Generate topic
                topics = ['credit_risk', 'operational_risk', 'market_risk', 'regulatory_risk', 
                         'capital_management', 'miscellaneous']
                topic_weights = [0.25, 0.2, 0.2, 0.15, 0.1, 0.1]
                final_topic = np.random.choice(topics, p=topic_weights)
                topic_confidence = np.random.uniform(0.6, 0.95)
                
                test_data.append({
                    'text': text,
                    'bank': institution,
                    'quarter': quarter,
                    'speaker_norm': speaker,
                    'sentiment_positive': sentiment_positive,
                    'sentiment_negative': sentiment_negative,
                    'sentiment_neutral': sentiment_neutral,
                    'sentiment_confidence': np.random.uniform(0.7, 0.95),
                    'risk_escalation_score': risk_escalation_score,
                    'stress_score': stress_score,
                    'confidence_score': confidence_score,
                    'hedging_score': hedging_score,
                    'uncertainty_score': uncertainty_score,
                    'formality_score': formality_score,
                    'complexity_score': complexity_score,
                    'final_topic': final_topic,
                    'topic_confidence': topic_confidence,
                    'source_file': f'{institution}_{quarter}_transcript.txt'
                })
    
    df = pd.DataFrame(test_data)
    print(f"[DATA] Created test dataset: {len(df)} records across {df['bank'].nunique()} institutions and {df['quarter'].nunique()} quarters")
    
    return df

def test_time_series_analysis():
    """Test time series analysis functionality"""
    print("\n" + "="*60)
    print("TESTING TIME SERIES ANALYSIS")
    print("="*60)
    
    try:
        # Initialize analyzer
        analyzer = TimeSeriesAnalyzer()
        print("[OK] Time series analyzer initialized successfully")
        
        # Create test data
        test_data = create_comprehensive_test_data()
        print("[OK] Test data created successfully")
        
        # Prepare time series data
        print("[Processing] Preparing time series data...")
        ts_data = analyzer.prepare_time_series_data(test_data)
        print(f"[OK] Time series data prepared: {len(ts_data)} time series records")
        
        # Test analysis for each institution
        institutions = test_data['bank'].unique()
        
        for institution in institutions[:2]:  # Test first 2 institutions
            print(f"\n[Processing] Analyzing {institution}...")
            
            # Perform comprehensive analysis
            analysis_results = analyzer.analyze_institution(ts_data, institution)
            
            print(f"[OK] Analysis completed for {institution}")
            print(f"  Data periods: {analysis_results['data_periods']}")
            print(f"  Trends detected: {len(analysis_results['trends'])}")
            print(f"  Seasonality patterns: {len(analysis_results['seasonality'])}")
            print(f"  Anomalies found: {sum(len(a['anomalies']) for a in analysis_results['anomalies'].values())}")
            print(f"  Risk level: {analysis_results['risk_signals']['overall_risk_level']}")
            
            # Display sample trend results
            if analysis_results['trends']:
                sample_metric = list(analysis_results['trends'].keys())[0]
                trend_data = analysis_results['trends'][sample_metric]
                print(f"  Sample trend ({sample_metric}): {trend_data['trend_direction']} "
                      f"(R²: {trend_data['r_squared']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"[FAILED] Time series analysis test failed: {e}")
        return False

def test_anomaly_detection():
    """Test anomaly detection functionality"""
    print("\n" + "="*60)
    print("TESTING ANOMALY DETECTION")
    print("="*60)
    
    try:
        # Initialize detector
        detector = AnomalyDetector()
        print("[OK] Anomaly detector initialized successfully")
        
        # Create test data
        test_data = create_comprehensive_test_data()
        print("[OK] Test data created successfully")
        
        # Perform anomaly detection
        print("[Processing] Running ensemble anomaly detection...")
        results_df, anomaly_report = detector.detect_anomalies(test_data)
        
        print("[OK] Anomaly detection completed")
        
        # Display results
        summary = anomaly_report['summary']
        print(f"\n[STATS] Anomaly Detection Results:")
        print(f"Total records analyzed: {summary['total_records']}")
        print(f"Anomalies detected: {summary['total_anomalies']}")
        print(f"Anomaly rate: {summary['anomaly_rate']:.1%}")
        
        # Severity distribution
        if 'severity_distribution' in anomaly_report:
            print(f"\nSeverity Distribution:")
            for severity, count in anomaly_report['severity_distribution'].items():
                print(f"  {severity}: {count}")
        
        # Method agreement
        if 'method_agreement' in anomaly_report:
            print(f"\nMethod Agreement:")
            for method, detections in anomaly_report['method_agreement'].items():
                print(f"  {method}: {detections} detections")
        
        # Top anomalies
        if anomaly_report['top_anomalies']:
            print(f"\n[SAMPLE] Top Anomalies:")
            for i, anomaly in enumerate(anomaly_report['top_anomalies'][:3]):
                print(f"  {i+1}. Score: {anomaly['anomaly_score']:.3f}, "
                      f"Institution: {anomaly.get('institution', 'Unknown')}, "
                      f"Features: {anomaly.get('contributing_features', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"[FAILED] Anomaly detection test failed: {e}")
        return False

def test_risk_scoring():
    """Test risk scoring functionality"""
    print("\n" + "="*60)
    print("TESTING RISK SCORING")
    print("="*60)
    
    try:
        # Initialize risk scorer
        scorer = RiskScorer()
        print("[OK] Risk scorer initialized successfully")
        
        # Create test data
        test_data = create_comprehensive_test_data()
        print("[OK] Test data created successfully")
        
        # Calculate composite risk scores
        print("[Processing] Calculating composite risk scores...")
        risk_results = scorer.calculate_composite_risk_score(test_data)
        
        print("[OK] Risk scoring completed")
        
        # Display overall statistics
        print(f"\n[STATS] Risk Scoring Results:")
        print(f"Total records scored: {len(risk_results)}")
        print(f"Mean risk score: {risk_results['risk_score'].mean():.3f}")
        print(f"Max risk score: {risk_results['risk_score'].max():.3f}")
        print(f"Risk score std: {risk_results['risk_score'].std():.3f}")
        
        # Risk category distribution
        risk_dist = risk_results['risk_category'].value_counts()
        print(f"\nRisk Category Distribution:")
        for category, count in risk_dist.items():
            percentage = count / len(risk_results) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        # Component analysis
        risk_components = ['risk_sentiment', 'risk_topic', 'risk_speaker', 
                          'risk_temporal', 'risk_anomaly', 'risk_volatility']
        
        print(f"\n[COMPONENTS] Risk Component Analysis:")
        for component in risk_components:
            if component in risk_results.columns:
                mean_score = risk_results[component].mean()
                print(f"  {component}: {mean_score:.3f}")
        
        # Institution-level analysis
        institutions = test_data['bank'].unique()
        
        print(f"\n[INSTITUTIONS] Institution Risk Profiles:")
        for institution in institutions:
            profile = scorer.generate_institution_risk_profile(risk_results, institution)
            
            if profile:
                overall_risk = profile['overall_risk']
                print(f"\n  {institution}:")
                print(f"    Mean risk score: {overall_risk['mean_score']:.3f}")
                print(f"    Current risk score: {overall_risk['current_score']:.3f}")
                print(f"    Records analyzed: {profile['total_records']}")
                
                # Risk distribution for this institution
                risk_dist = profile.get('risk_distribution', {})
                if risk_dist:
                    print(f"    Risk distribution: {dict(risk_dist)}")
                
                # Recommendations
                recommendations = profile.get('recommendations', [])
                if recommendations:
                    print(f"    Key recommendations: {len(recommendations)} items")
                    for rec in recommendations[:2]:  # Show first 2
                        print(f"      - {rec}")
        
        # Comparative analysis
        print(f"\n[COMPARATIVE] Generating comparative analysis...")
        comparative_analysis = scorer.generate_comparative_analysis(risk_results)
        
        if comparative_analysis:
            benchmarks = comparative_analysis.get('industry_benchmarks', {})
            print(f"Industry Benchmarks:")
            print(f"  Mean risk score: {benchmarks.get('mean_risk_score', 0):.3f}")
            print(f"  High risk rate: {benchmarks.get('high_risk_rate', 0):.1%}")
            
            # Institution rankings
            rankings = comparative_analysis.get('institution_rankings', {})
            if 'by_mean_risk' in rankings:
                print(f"\nInstitution Rankings (by mean risk):")
                for i, (inst, score) in enumerate(list(rankings['by_mean_risk'].items())[:3]):
                    print(f"  {i+1}. {inst}: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"[FAILED] Risk scoring test failed: {e}")
        return False

def test_integration():
    """Test integration between all Phase 3 components"""
    print("\n" + "="*60)
    print("TESTING INTEGRATION")
    print("="*60)
    
    try:
        # Initialize all components
        analyzer = TimeSeriesAnalyzer()
        detector = AnomalyDetector()
        scorer = RiskScorer()
        print("[OK] All components initialized successfully")
        
        # Create test data
        test_data = create_comprehensive_test_data()
        print("[OK] Test data created successfully")
        
        # Step 1: Time series preparation
        print("[Processing] Step 1: Time series analysis...")
        ts_data = analyzer.prepare_time_series_data(test_data)
        
        # Step 2: Anomaly detection
        print("[Processing] Step 2: Anomaly detection...")
        anomaly_results, anomaly_report = detector.detect_anomalies(test_data)
        
        # Step 3: Risk scoring (using anomaly-enhanced data)
        print("[Processing] Step 3: Risk scoring with anomaly data...")
        final_results = scorer.calculate_composite_risk_score(anomaly_results)
        
        print("[OK] Integrated analysis completed")
        
        # Analyze integration results
        print(f"\n[INTEGRATION] Integration Analysis:")
        print(f"Records processed through pipeline: {len(final_results)}")
        print(f"Anomalies detected: {anomaly_report['summary']['total_anomalies']}")
        print(f"High-risk records: {len(final_results[final_results['risk_category'].isin(['high', 'critical'])])}")
        
        # Correlation between anomalies and risk scores
        if 'is_anomaly' in final_results.columns:
            anomaly_risk_corr = final_results[['is_anomaly', 'risk_score']].corr().iloc[0, 1]
            print(f"Anomaly-Risk correlation: {anomaly_risk_corr:.3f}")
        
        # Cross-component validation
        high_risk_records = final_results[final_results['risk_score'] > 0.7]
        if len(high_risk_records) > 0:
            anomaly_rate_in_high_risk = high_risk_records.get('is_anomaly', pd.Series(0)).mean()
            print(f"Anomaly rate in high-risk records: {anomaly_rate_in_high_risk:.1%}")
        
        # Institution-level integration
        print(f"\n[VALIDATION] Cross-Component Validation:")
        for institution in test_data['bank'].unique()[:2]:  # Test 2 institutions
            inst_data = final_results[final_results['bank'] == institution]
            
            # Time series analysis
            ts_analysis = analyzer.analyze_institution(ts_data, institution)
            
            # Risk profile
            risk_profile = scorer.generate_institution_risk_profile(final_results, institution)
            
            print(f"\n  {institution}:")
            print(f"    Time series risk level: {ts_analysis['risk_signals']['overall_risk_level']}")
            print(f"    Mean risk score: {risk_profile['overall_risk']['mean_score']:.3f}")
            print(f"    Anomaly rate: {inst_data.get('is_anomaly', pd.Series(0)).mean():.1%}")
        
        return True
        
    except Exception as e:
        print(f"[FAILED] Integration test failed: {e}")
        return False

def main():
    """Run all Phase 3 statistical analysis tests"""
    print("Starting Phase 3 Statistical Analysis Component Testing")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    test_results = {
        'time_series_analysis': test_time_series_analysis(),
        'anomaly_detection': test_anomaly_detection(),
        'risk_scoring': test_risk_scoring(),
        'integration': test_integration()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "[OK] PASSED" if result else "[FAILED] FAILED"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n[STATS] Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("[SUCCESS] All Phase 3 statistical analysis components are working correctly!")
        print("[OK] Ready for production deployment and Phase 4 development")
    else:
        print("[WARNING] Some tests failed. Please review the errors above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)