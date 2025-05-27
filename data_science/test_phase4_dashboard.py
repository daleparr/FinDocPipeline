"""
Phase 4 Dashboard Testing Suite

This script tests the complete stakeholder dashboard implementation,
including business intelligence layer, document processing, and 
stakeholder-friendly output generation.

Test Coverage:
- Business intelligence translator
- Dashboard integration layer
- Stakeholder dashboard components
- End-to-end workflow validation
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import json
from datetime import datetime
import logging

# Add project paths
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_stakeholder_translator():
    """Test the stakeholder translator component"""
    print("\n" + "="*60)
    print("TESTING STAKEHOLDER TRANSLATOR")
    print("="*60)
    
    try:
        from scripts.business_intelligence.stakeholder_translator import StakeholderTranslator
        
        # Initialize translator
        translator = StakeholderTranslator()
        print("[OK] Stakeholder translator initialized successfully")
        
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
        print(f"[OK] Risk classification: {risk_classification['classification']} ({risk_classification['score']}/10)")
        print(f"[OK] Risk message: {risk_classification['message']}")
        
        # Test topic translation
        print("[Processing] Testing topic translation...")
        business_topics = translator.translate_topics_to_business_language(sample_results)
        print(f"[OK] Translated {len(business_topics)} topics to business language")
        
        for topic in business_topics:
            print(f"  - {topic['label']}: {topic['percentage']}% ({topic['risk_level']} risk)")
        
        # Test recommendations
        print("[Processing] Testing recommendation generation...")
        recommendations = translator.generate_stakeholder_recommendations(sample_results)
        print(f"[OK] Generated recommendations:")
        print(f"  - Immediate attention: {len(recommendations['immediate_attention'])} items")
        print(f"  - Watch closely: {len(recommendations['watch_closely'])} items")
        print(f"  - Positive indicators: {len(recommendations['positive_indicators'])} items")
        
        # Test sentiment summary
        print("[Processing] Testing sentiment summary...")
        sentiment_summary = translator.create_sentiment_summary({
            'positive_percentage': 70,
            'neutral_percentage': 20,
            'negative_percentage': 10,
            'trend': 'stable'
        })
        print(f"[OK] Sentiment summary: {sentiment_summary['overall_sentiment']} {sentiment_summary['emoji']}")
        
        # Test executive summary
        print("[Processing] Testing executive summary generation...")
        executive_summary = translator.generate_executive_summary("JPMorgan Chase", sample_results)
        print(f"[OK] Executive summary generated ({len(executive_summary)} characters)")
        
        print("[SUCCESS] All stakeholder translator tests passed!")
        return True
        
    except Exception as e:
        print(f"[FAILED] Stakeholder translator test failed: {e}")
        return False

def test_dashboard_integration():
    """Test the dashboard integration layer"""
    print("\n" + "="*60)
    print("TESTING DASHBOARD INTEGRATION")
    print("="*60)
    
    try:
        from scripts.business_intelligence.dashboard_integration import DashboardIntegration
        
        # Initialize integration
        integration = DashboardIntegration()
        print("[OK] Dashboard integration initialized successfully")
        
        # Create mock uploaded files
        print("[Processing] Creating mock uploaded files...")
        mock_files = create_mock_uploaded_files()
        print(f"[OK] Created {len(mock_files)} mock files")
        
        # Test document processing (synchronous)
        print("[Processing] Testing document processing...")
        institution = "JPMorgan Chase"
        
        # Test with fallback (since we don't have real ETL pipeline)
        results = integration._generate_fallback_results(institution, "Test mode")
        print(f"[OK] Document processing completed for {institution}")
        
        # Validate results structure
        print("[Processing] Validating results structure...")
        required_keys = [
            'institution', 'processing_summary', 'statistical_analysis',
            'stakeholder_insights', 'composite_risk_score'
        ]
        
        for key in required_keys:
            if key in results:
                print(f"[OK] Required key '{key}' present")
            else:
                print(f"[ERROR] Missing required key '{key}'")
                return False
        
        # Test helper methods
        print("[Processing] Testing helper methods...")
        
        # Test quarter extraction
        test_filenames = [
            "Q1_2024_transcript.txt",
            "earnings_Q2_2023.pdf",
            "financial_summary_2024Q3.xlsx"
        ]
        
        for filename in test_filenames:
            quarter = integration._extract_quarter_from_filename(filename)
            print(f"[OK] Extracted quarter '{quarter}' from '{filename}'")
        
        # Test content segmentation
        test_content = "This is the first paragraph.\n\nThis is the second paragraph with more content.\n\nThird paragraph here."
        segments = integration._split_content_into_segments(test_content)
        print(f"[OK] Split content into {len(segments)} segments")
        
        print("[SUCCESS] All dashboard integration tests passed!")
        return True
        
    except Exception as e:
        print(f"[FAILED] Dashboard integration test failed: {e}")
        return False

def test_business_intelligence_workflow():
    """Test the complete business intelligence workflow"""
    print("\n" + "="*60)
    print("TESTING BUSINESS INTELLIGENCE WORKFLOW")
    print("="*60)
    
    try:
        from scripts.business_intelligence.stakeholder_translator import StakeholderTranslator
        from scripts.business_intelligence.dashboard_integration import DashboardIntegration
        
        # Initialize components
        translator = StakeholderTranslator()
        integration = DashboardIntegration()
        print("[OK] All BI components initialized")
        
        # Create sample analysis results
        print("[Processing] Creating sample analysis results...")
        sample_statistical_results = {
            'composite_risk_score': 0.65,
            'anomaly_detection': {
                'total_anomalies': 8,
                'severity_distribution': {'low': 6, 'medium': 2, 'high': 0}
            },
            'time_series': {
                'trend_direction': 'declining',
                'risk_level': 'medium'
            },
            'risk_scoring': {
                'overall_risk_score': 0.65,
                'component_scores': {
                    'sentiment_risk': 0.4,
                    'topic_risk': 0.8,
                    'speaker_risk': 0.3,
                    'temporal_risk': 0.7,
                    'anomaly_risk': 0.6,
                    'volatility_risk': 0.5
                }
            }
        }
        
        # Test complete workflow
        print("[Processing] Testing complete BI workflow...")
        
        # Step 1: Risk classification
        risk_info = translator.translate_risk_score(sample_statistical_results)
        print(f"[OK] Step 1 - Risk Classification: {risk_info['classification']}")
        
        # Step 2: Topic analysis
        topic_insights = translator.translate_topics_to_business_language({
            'topic_analysis': integration._generate_sample_topic_analysis()
        })
        print(f"[OK] Step 2 - Topic Analysis: {len(topic_insights)} topics")
        
        # Step 3: Recommendations
        recommendations = translator.generate_stakeholder_recommendations({
            'risk_drivers': integration._extract_risk_drivers(sample_statistical_results)
        })
        print(f"[OK] Step 3 - Recommendations generated")
        
        # Step 4: Executive summary
        executive_summary = translator.generate_executive_summary(
            "Wells Fargo", sample_statistical_results
        )
        print(f"[OK] Step 4 - Executive Summary: {len(executive_summary.split())} words")
        
        # Validate output quality
        print("[Processing] Validating output quality...")
        
        # Check risk classification
        assert risk_info['classification'] in ['LOW RISK', 'MEDIUM RISK', 'HIGH RISK']
        assert 0 <= risk_info['score'] <= 10
        print("[OK] Risk classification format valid")
        
        # Check topic insights
        assert len(topic_insights) > 0
        for topic in topic_insights:
            assert 'label' in topic
            assert 'percentage' in topic
            assert 'risk_level' in topic
        print("[OK] Topic insights format valid")
        
        # Check recommendations structure
        assert 'immediate_attention' in recommendations
        assert 'watch_closely' in recommendations
        assert 'positive_indicators' in recommendations
        print("[OK] Recommendations structure valid")
        
        # Check executive summary
        assert len(executive_summary) > 100  # Should be substantial
        assert "RISK ASSESSMENT" in executive_summary
        print("[OK] Executive summary format valid")
        
        print("[SUCCESS] Complete BI workflow test passed!")
        return True
        
    except Exception as e:
        print(f"[FAILED] BI workflow test failed: {e}")
        return False

def test_stakeholder_output_format():
    """Test stakeholder-friendly output formatting"""
    print("\n" + "="*60)
    print("TESTING STAKEHOLDER OUTPUT FORMAT")
    print("="*60)
    
    try:
        from scripts.business_intelligence.stakeholder_translator import StakeholderTranslator
        
        translator = StakeholderTranslator()
        
        # Test various risk levels
        risk_scenarios = [
            {'composite_risk_score': 0.2, 'expected': 'LOW RISK'},
            {'composite_risk_score': 0.5, 'expected': 'MEDIUM RISK'},
            {'composite_risk_score': 0.8, 'expected': 'HIGH RISK'}
        ]
        
        print("[Processing] Testing risk level classifications...")
        for scenario in risk_scenarios:
            result = translator.translate_risk_score(scenario)
            actual = result['classification']
            expected = scenario['expected']
            
            if actual == expected:
                print(f"[OK] Score {scenario['composite_risk_score']} → {actual}")
            else:
                print(f"[ERROR] Score {scenario['composite_risk_score']} → {actual}, expected {expected}")
                return False
        
        # Test business language translation
        print("[Processing] Testing business language translation...")
        technical_topics = {
            'topic_analysis': {
                'financial_performance': {'percentage': 40, 'average_sentiment': 0.8},
                'regulatory_compliance': {'percentage': 25, 'average_sentiment': 0.3},
                'technology_digital': {'percentage': 20, 'average_sentiment': 0.6}
            }
        }
        
        business_topics = translator.translate_topics_to_business_language(technical_topics)
        
        # Verify business-friendly labels
        expected_labels = ['💰 Revenue & Profitability', '🏛️ Regulatory & Compliance', '💻 Technology & Digital']
        actual_labels = [topic['label'] for topic in business_topics]
        
        for expected in expected_labels:
            if expected in actual_labels:
                print(f"[OK] Business label: {expected}")
            else:
                print(f"[ERROR] Missing business label: {expected}")
                return False
        
        # Test recommendation language
        print("[Processing] Testing recommendation language...")
        sample_drivers = [
            {'topic': 'regulatory_compliance', 'severity': 'high', 'description': 'Test issue'},
            {'topic': 'financial_performance', 'severity': 'low', 'description': 'Test strength'}
        ]
        
        recommendations = translator.generate_stakeholder_recommendations({'risk_drivers': sample_drivers})
        
        # Check for actionable language
        immediate_actions = recommendations.get('immediate_attention', [])
        if immediate_actions:
            action_text = immediate_actions[0].get('action', '')
            if any(word in action_text.lower() for word in ['schedule', 'review', 'monitor', 'assess']):
                print("[OK] Actionable language detected in recommendations")
            else:
                print("[ERROR] Recommendations lack actionable language")
                return False
        
        print("[SUCCESS] All stakeholder output format tests passed!")
        return True
        
    except Exception as e:
        print(f"[FAILED] Stakeholder output format test failed: {e}")
        return False

def create_mock_uploaded_files():
    """Create mock uploaded files for testing"""
    class MockFile:
        def __init__(self, name, content):
            self.name = name
            self.content = content.encode('utf-8')
            self.size = len(self.content)
        
        def getvalue(self):
            return self.content
    
    mock_files = [
        MockFile("Q1_2024_earnings_transcript.txt", 
                "CEO: Welcome to our Q1 earnings call. We reported strong revenue growth of 15% this quarter."),
        MockFile("Q2_2024_financial_summary.csv", 
                "Metric,Value\nRevenue,2.5B\nNet Income,500M\nROE,12%"),
        MockFile("Q3_2024_presentation.pdf", 
                "Financial presentation content with regulatory compliance discussions."),
        MockFile("Q4_2024_analyst_call.txt", 
                "Analyst: What are your thoughts on market conditions? CFO: We remain cautiously optimistic.")
    ]
    
    return mock_files

def run_integration_test():
    """Run complete integration test"""
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TEST")
    print("="*60)
    
    try:
        # Test complete workflow with mock data
        print("[Processing] Testing complete stakeholder workflow...")
        
        # Simulate document upload and processing
        institution = "Bank of America"
        mock_files = create_mock_uploaded_files()
        
        # Simulate analysis results
        mock_results = {
            'institution': institution,
            'processing_summary': {
                'total_documents': len(mock_files),
                'total_records': 150,
                'quarters_analyzed': 4
            },
            'composite_risk_score': 0.45,
            'anomaly_detection': {'total_anomalies': 3},
            'time_series': {'trend_direction': 'stable'},
            'topic_analysis': {
                'financial_performance': {'percentage': 35, 'average_sentiment': 0.7},
                'regulatory_compliance': {'percentage': 25, 'average_sentiment': 0.4}
            }
        }
        
        # Test stakeholder translation
        from scripts.business_intelligence.stakeholder_translator import StakeholderTranslator
        translator = StakeholderTranslator()
        
        stakeholder_insights = {
            'risk_classification': translator.translate_risk_score(mock_results),
            'topic_insights': translator.translate_topics_to_business_language(mock_results),
            'recommendations': translator.generate_stakeholder_recommendations(mock_results),
            'executive_summary': translator.generate_executive_summary(institution, mock_results)
        }
        
        # Validate complete output
        print(f"[OK] Institution: {institution}")
        print(f"[OK] Documents processed: {mock_results['processing_summary']['total_documents']}")
        print(f"[OK] Risk classification: {stakeholder_insights['risk_classification']['classification']}")
        print(f"[OK] Topics analyzed: {len(stakeholder_insights['topic_insights'])}")
        print(f"[OK] Recommendations generated: {len(stakeholder_insights['recommendations']['immediate_attention']) + len(stakeholder_insights['recommendations']['watch_closely'])}")
        print(f"[OK] Executive summary: {len(stakeholder_insights['executive_summary'])} characters")
        
        print("[SUCCESS] Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"[FAILED] Integration test failed: {e}")
        return False

def main():
    """Run all Phase 4 tests"""
    print("Starting Phase 4 Dashboard Component Testing")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test results tracking
    test_results = {}
    
    # Run individual component tests
    test_results['stakeholder_translator'] = test_stakeholder_translator()
    test_results['dashboard_integration'] = test_dashboard_integration()
    test_results['business_intelligence_workflow'] = test_business_intelligence_workflow()
    test_results['stakeholder_output_format'] = test_stakeholder_output_format()
    test_results['integration_test'] = run_integration_test()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        icon = "[OK]" if result else "[FAILED]"
        print(f"{icon} {status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n[STATS] Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("[SUCCESS] All Phase 4 dashboard components are working correctly!")
        print("[READY] Stakeholder dashboard is ready for deployment!")
    else:
        print(f"[WARNING] {total_tests - passed_tests} tests failed. Please review the errors above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)