"""
Simplified Phase 4 Dashboard Testing

This script tests the core Phase 4 components with simplified dependencies
and proper encoding handling for Windows systems.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add project paths
sys.path.append(str(Path(__file__).parent))

def test_stakeholder_translator_simple():
    """Test stakeholder translator with simplified output"""
    print("\n" + "="*50)
    print("TESTING STAKEHOLDER TRANSLATOR")
    print("="*50)
    
    try:
        from scripts.business_intelligence.stakeholder_translator import StakeholderTranslator
        
        # Initialize translator
        translator = StakeholderTranslator()
        print("[OK] Stakeholder translator initialized")
        
        # Test data
        sample_results = {
            'composite_risk_score': 0.42,
            'anomaly_detection': {'total_anomalies': 5},
            'time_series': {'trend_direction': 'improving'},
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
                }
            },
            'risk_drivers': [
                {
                    'topic': 'regulatory_compliance',
                    'severity': 'medium',
                    'description': 'Increased regulatory discussions'
                }
            ]
        }
        
        # Test risk score translation
        print("[Processing] Testing risk score translation...")
        risk_classification = translator.translate_risk_score(sample_results)
        print(f"[OK] Risk classification: {risk_classification['classification']}")
        print(f"[OK] Risk score: {risk_classification['score']}/10")
        print(f"[OK] Risk color: {risk_classification['color']}")
        
        # Test topic translation
        print("[Processing] Testing topic translation...")
        business_topics = translator.translate_topics_to_business_language(sample_results)
        print(f"[OK] Translated {len(business_topics)} topics")
        
        for topic in business_topics:
            print(f"  - {topic['percentage']}% {topic['risk_level']} risk")
        
        # Test recommendations
        print("[Processing] Testing recommendations...")
        recommendations = translator.generate_stakeholder_recommendations(sample_results)
        immediate = len(recommendations['immediate_attention'])
        watch = len(recommendations['watch_closely'])
        positive = len(recommendations['positive_indicators'])
        print(f"[OK] Recommendations: {immediate} immediate, {watch} watch, {positive} positive")
        
        # Test executive summary
        print("[Processing] Testing executive summary...")
        executive_summary = translator.generate_executive_summary("JPMorgan Chase", sample_results)
        print(f"[OK] Executive summary generated ({len(executive_summary)} chars)")
        
        print("[SUCCESS] Stakeholder translator tests passed!")
        return True
        
    except Exception as e:
        print(f"[FAILED] Stakeholder translator test failed: {e}")
        return False

def test_business_logic():
    """Test core business logic without complex dependencies"""
    print("\n" + "="*50)
    print("TESTING BUSINESS LOGIC")
    print("="*50)
    
    try:
        from scripts.business_intelligence.stakeholder_translator import StakeholderTranslator
        
        translator = StakeholderTranslator()
        
        # Test risk level classification
        print("[Processing] Testing risk level logic...")
        
        test_scores = [0.2, 0.5, 0.8]
        expected_levels = ['LOW RISK', 'MEDIUM RISK', 'HIGH RISK']
        
        for score, expected in zip(test_scores, expected_levels):
            result = translator.translate_risk_score({'composite_risk_score': score})
            actual = result['classification']
            
            if actual == expected:
                print(f"[OK] Score {score} -> {actual}")
            else:
                print(f"[ERROR] Score {score} -> {actual}, expected {expected}")
                return False
        
        # Test topic mapping
        print("[Processing] Testing topic mapping...")
        
        sample_topics = {
            'topic_analysis': {
                'financial_performance': {'percentage': 40, 'average_sentiment': 0.8},
                'regulatory_compliance': {'percentage': 25, 'average_sentiment': 0.3}
            }
        }
        
        business_topics = translator.translate_topics_to_business_language(sample_topics)
        
        if len(business_topics) == 2:
            print("[OK] Topic mapping working correctly")
        else:
            print(f"[ERROR] Expected 2 topics, got {len(business_topics)}")
            return False
        
        # Test recommendation generation
        print("[Processing] Testing recommendation logic...")
        
        test_drivers = [
            {'topic': 'regulatory_compliance', 'severity': 'high', 'description': 'Test high severity'},
            {'topic': 'financial_performance', 'severity': 'low', 'description': 'Test low severity'}
        ]
        
        recommendations = translator.generate_stakeholder_recommendations({'risk_drivers': test_drivers})
        
        if len(recommendations['immediate_attention']) > 0:
            print("[OK] High severity items in immediate attention")
        else:
            print("[ERROR] High severity items not properly categorized")
            return False
        
        if len(recommendations['positive_indicators']) > 0:
            print("[OK] Low severity items in positive indicators")
        else:
            print("[ERROR] Low severity items not properly categorized")
            return False
        
        print("[SUCCESS] Business logic tests passed!")
        return True
        
    except Exception as e:
        print(f"[FAILED] Business logic test failed: {e}")
        return False

def test_data_processing():
    """Test data processing and formatting"""
    print("\n" + "="*50)
    print("TESTING DATA PROCESSING")
    print("="*50)
    
    try:
        from scripts.business_intelligence.stakeholder_translator import StakeholderTranslator
        
        translator = StakeholderTranslator()
        
        # Test with missing data
        print("[Processing] Testing missing data handling...")
        
        incomplete_results = {
            'composite_risk_score': 0.5
            # Missing other fields
        }
        
        risk_info = translator.translate_risk_score(incomplete_results)
        
        if 'classification' in risk_info and 'score' in risk_info:
            print("[OK] Missing data handled gracefully")
        else:
            print("[ERROR] Missing data not handled properly")
            return False
        
        # Test with edge cases
        print("[Processing] Testing edge cases...")
        
        edge_cases = [
            {'composite_risk_score': 0.0},  # Minimum score
            {'composite_risk_score': 1.0},  # Maximum score
            {'composite_risk_score': 0.5}   # Middle score
        ]
        
        for case in edge_cases:
            result = translator.translate_risk_score(case)
            if result['score'] >= 0 and result['score'] <= 10:
                print(f"[OK] Edge case score {case['composite_risk_score']} handled")
            else:
                print(f"[ERROR] Edge case score {case['composite_risk_score']} failed")
                return False
        
        # Test sentiment processing
        print("[Processing] Testing sentiment processing...")
        
        sentiment_data = {
            'positive_percentage': 70,
            'neutral_percentage': 20,
            'negative_percentage': 10,
            'trend': 'stable'
        }
        
        sentiment_summary = translator.create_sentiment_summary(sentiment_data)
        
        if 'overall_sentiment' in sentiment_summary:
            print("[OK] Sentiment processing working")
        else:
            print("[ERROR] Sentiment processing failed")
            return False
        
        print("[SUCCESS] Data processing tests passed!")
        return True
        
    except Exception as e:
        print(f"[FAILED] Data processing test failed: {e}")
        return False

def test_output_format():
    """Test output formatting and structure"""
    print("\n" + "="*50)
    print("TESTING OUTPUT FORMAT")
    print("="*50)
    
    try:
        from scripts.business_intelligence.stakeholder_translator import StakeholderTranslator
        
        translator = StakeholderTranslator()
        
        # Test complete workflow output
        print("[Processing] Testing complete output structure...")
        
        sample_results = {
            'composite_risk_score': 0.6,
            'anomaly_detection': {'total_anomalies': 7},
            'time_series': {'trend_direction': 'declining'},
            'topic_analysis': {
                'financial_performance': {'percentage': 30, 'average_sentiment': 0.6},
                'regulatory_compliance': {'percentage': 25, 'average_sentiment': 0.4},
                'technology_digital': {'percentage': 20, 'average_sentiment': 0.7}
            },
            'risk_drivers': [
                {'topic': 'regulatory_compliance', 'severity': 'medium', 'description': 'Test'},
                {'topic': 'financial_performance', 'severity': 'low', 'description': 'Test'}
            ]
        }
        
        # Generate all outputs
        risk_info = translator.translate_risk_score(sample_results)
        topics = translator.translate_topics_to_business_language(sample_results)
        recommendations = translator.generate_stakeholder_recommendations(sample_results)
        sentiment = translator.create_sentiment_summary({'positive_percentage': 65, 'negative_percentage': 15, 'neutral_percentage': 20, 'trend': 'stable'})
        executive = translator.generate_executive_summary("Test Bank", sample_results)
        
        # Validate structure
        required_risk_fields = ['classification', 'score', 'message', 'color']
        for field in required_risk_fields:
            if field in risk_info:
                print(f"[OK] Risk info has {field}")
            else:
                print(f"[ERROR] Risk info missing {field}")
                return False
        
        if len(topics) > 0:
            print(f"[OK] Generated {len(topics)} topic insights")
        else:
            print("[ERROR] No topic insights generated")
            return False
        
        rec_categories = ['immediate_attention', 'watch_closely', 'positive_indicators']
        for category in rec_categories:
            if category in recommendations:
                print(f"[OK] Recommendations has {category}")
            else:
                print(f"[ERROR] Recommendations missing {category}")
                return False
        
        if len(executive) > 50:
            print(f"[OK] Executive summary generated ({len(executive)} chars)")
        else:
            print("[ERROR] Executive summary too short")
            return False
        
        print("[SUCCESS] Output format tests passed!")
        return True
        
    except Exception as e:
        print(f"[FAILED] Output format test failed: {e}")
        return False

def test_integration_workflow():
    """Test complete integration workflow"""
    print("\n" + "="*50)
    print("TESTING INTEGRATION WORKFLOW")
    print("="*50)
    
    try:
        from scripts.business_intelligence.stakeholder_translator import StakeholderTranslator
        
        translator = StakeholderTranslator()
        
        # Simulate complete stakeholder workflow
        print("[Processing] Simulating complete workflow...")
        
        # Step 1: Mock document processing results
        institution = "Wells Fargo"
        mock_analysis = {
            'composite_risk_score': 0.72,
            'anomaly_detection': {'total_anomalies': 9},
            'time_series': {'trend_direction': 'declining'},
            'topic_analysis': {
                'financial_performance': {'percentage': 28, 'average_sentiment': 0.5},
                'regulatory_compliance': {'percentage': 32, 'average_sentiment': 0.3},
                'technology_digital': {'percentage': 15, 'average_sentiment': 0.6},
                'market_conditions': {'percentage': 15, 'average_sentiment': 0.4},
                'operations_strategy': {'percentage': 10, 'average_sentiment': 0.7}
            },
            'risk_drivers': [
                {'topic': 'regulatory_compliance', 'severity': 'high', 'description': 'Significant regulatory concerns'},
                {'topic': 'market_conditions', 'severity': 'medium', 'description': 'Market volatility impact'},
                {'topic': 'operations_strategy', 'severity': 'low', 'description': 'Strong operational performance'}
            ]
        }
        
        # Step 2: Generate stakeholder insights
        stakeholder_output = {
            'institution': institution,
            'risk_classification': translator.translate_risk_score(mock_analysis),
            'topic_insights': translator.translate_topics_to_business_language(mock_analysis),
            'recommendations': translator.generate_stakeholder_recommendations(mock_analysis),
            'sentiment_summary': translator.create_sentiment_summary({
                'positive_percentage': 45, 'negative_percentage': 35, 'neutral_percentage': 20, 'trend': 'declining'
            }),
            'executive_summary': translator.generate_executive_summary(institution, mock_analysis)
        }
        
        # Step 3: Validate complete output
        print(f"[OK] Institution: {stakeholder_output['institution']}")
        print(f"[OK] Risk Level: {stakeholder_output['risk_classification']['classification']}")
        print(f"[OK] Risk Score: {stakeholder_output['risk_classification']['score']}/10")
        print(f"[OK] Topics Analyzed: {len(stakeholder_output['topic_insights'])}")
        
        total_recommendations = (
            len(stakeholder_output['recommendations']['immediate_attention']) +
            len(stakeholder_output['recommendations']['watch_closely']) +
            len(stakeholder_output['recommendations']['positive_indicators'])
        )
        print(f"[OK] Total Recommendations: {total_recommendations}")
        
        print(f"[OK] Sentiment: {stakeholder_output['sentiment_summary']['overall_sentiment']}")
        print(f"[OK] Executive Summary: {len(stakeholder_output['executive_summary'])} characters")
        
        # Step 4: Validate business value
        if stakeholder_output['risk_classification']['classification'] == 'HIGH RISK':
            print("[OK] High risk properly identified")
        
        if len(stakeholder_output['recommendations']['immediate_attention']) > 0:
            print("[OK] Immediate actions identified for high risk")
        
        if 'regulatory' in stakeholder_output['executive_summary'].lower():
            print("[OK] Key risk drivers mentioned in summary")
        
        print("[SUCCESS] Integration workflow test passed!")
        return True
        
    except Exception as e:
        print(f"[FAILED] Integration workflow test failed: {e}")
        return False

def main():
    """Run simplified Phase 4 tests"""
    print("Starting Simplified Phase 4 Dashboard Testing")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test results tracking
    test_results = {}
    
    # Run tests
    test_results['stakeholder_translator'] = test_stakeholder_translator_simple()
    test_results['business_logic'] = test_business_logic()
    test_results['data_processing'] = test_data_processing()
    test_results['output_format'] = test_output_format()
    test_results['integration_workflow'] = test_integration_workflow()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        icon = "[OK]" if result else "[FAILED]"
        print(f"{icon} {status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n[STATS] Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("[SUCCESS] Phase 4 core components working correctly!")
        print("[READY] Stakeholder dashboard core functionality validated!")
    else:
        print(f"[WARNING] {total_tests - passed_tests} tests failed.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)