"""
Test script for Phase 2 NLP components: Topic Modeling and Sentiment Analysis
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
from datetime import datetime

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent))

# Import our new components
from sentiment_analysis.finbert_analyzer import FinBERTAnalyzer
from topic_modeling.hybrid_topic_engine import HybridTopicEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_test_data() -> pd.DataFrame:
    """Create comprehensive test data for validation"""
    
    test_sentences = [
        # Credit Risk Examples
        "Our loan loss provisions increased significantly this quarter due to deteriorating credit quality in the commercial portfolio.",
        "We are seeing higher delinquency rates in our consumer lending business, particularly in credit cards.",
        "The allowance for credit losses was raised by $500 million to reflect current economic conditions.",
        
        # Operational Risk Examples  
        "We experienced a cyber security incident that temporarily disrupted our online banking services.",
        "Compliance costs continue to rise as we strengthen our regulatory oversight and control frameworks.",
        "The operational risk team identified several process failures that need immediate attention.",
        
        # Market Risk Examples
        "Trading revenues declined due to increased market volatility and reduced client activity.",
        "Our value-at-risk measures have increased reflecting higher correlation across asset classes.",
        "Interest rate exposure remains well within our established risk limits and stress test scenarios.",
        
        # Regulatory Risk Examples
        "We successfully completed the annual stress test with capital ratios well above regulatory minimums.",
        "Basel III implementation continues to impact our capital planning and business strategy.",
        "The Federal Reserve examination resulted in several recommendations for improvement.",
        
        # Positive Sentiment Examples
        "We are confident in our strong capital position and robust risk management framework.",
        "The bank delivered solid performance with improving credit metrics across all business lines.",
        "Our digital transformation initiatives are strengthening our competitive position.",
        
        # Negative Sentiment Examples
        "We are concerned about the potential impact of economic uncertainty on our loan portfolio.",
        "Credit conditions are challenging and we expect continued pressure on margins.",
        "The regulatory environment remains complex and costly to navigate.",
        
        # Neutral/Mixed Examples
        "We expect moderate growth in lending volumes over the next quarter.",
        "Interest rates may impact our net interest margin depending on the Fed's policy decisions.",
        "The bank is evaluating various strategic options to enhance shareholder value.",
        
        # Emergent Topic Examples (should not match seed topics)
        "Our ESG initiatives include sustainable finance products and carbon footprint reduction.",
        "Artificial intelligence and machine learning are transforming our customer experience.",
        "We are expanding our presence in emerging markets through strategic partnerships."
    ]
    
    # Create speakers with different roles
    speakers = [
        'CEO', 'CFO', 'CRO',  # Credit risk (3)
        'CRO', 'CCO', 'CRO',  # Operational risk (3)
        'CFO', 'CRO', 'CFO',  # Market risk (3)
        'CFO', 'CRO', 'CEO',  # Regulatory risk (3)
        'CEO', 'CFO', 'CEO',  # Positive (3)
        'CRO', 'CFO', 'CEO',  # Negative (3)
        'CFO', 'CEO', 'CFO',  # Neutral (3)
        'CEO', 'CTO', 'CEO'   # Emergent (3)
    ]
    
    # Create test DataFrame
    test_data = pd.DataFrame({
        'text': test_sentences,
        'speaker_norm': speakers,
        'bank': ['TestBank'] * len(test_sentences),
        'quarter': ['Q1_2025'] * len(test_sentences),
        'source_file': ['test_transcript.txt'] * len(test_sentences),
        'sentence_id': [f'sent_{i:03d}' for i in range(len(test_sentences))]
    })
    
    return test_data

def test_sentiment_analysis():
    """Test the FinBERT sentiment analyzer"""
    print("\n" + "="*60)
    print("TESTING SENTIMENT ANALYSIS")
    print("="*60)
    
    try:
        # Initialize analyzer
        analyzer = FinBERTAnalyzer()
        print("[OK] FinBERT analyzer initialized successfully")
        
        # Create test data
        test_data = create_test_data()
        print(f"[OK] Test data created: {len(test_data)} sentences")
        
        # Analyze sentiment
        print("[Processing] Running sentiment analysis...")
        result_df = analyzer.analyze_dataframe(test_data)
        print("[OK] Sentiment analysis completed")
        
        # Display results
        print(f"\n[STATS] Sentiment Analysis Results:")
        print(f"Total sentences processed: {len(result_df)}")
        
        # Sentiment distribution
        sentiment_dist = result_df['sentiment_label'].value_counts()
        print(f"\nSentiment Distribution:")
        for sentiment, count in sentiment_dist.items():
            print(f"  {sentiment}: {count} ({count/len(result_df)*100:.1f}%)")
        
        # Average scores
        print(f"\nAverage Scores:")
        print(f"  Confidence: {result_df['sentiment_confidence'].mean():.3f}")
        print(f"  Hedging: {result_df['hedging_score'].mean():.3f}")
        print(f"  Uncertainty: {result_df['uncertainty_score'].mean():.3f}")
        print(f"  Risk Escalation: {result_df['risk_escalation_score'].mean():.3f}")
        
        # Sample results
        print(f"\n[SAMPLE] Sample Results:")
        sample_cols = ['text', 'sentiment_label', 'sentiment_confidence', 'hedging_score', 'risk_escalation_score']
        for i in range(min(3, len(result_df))):
            print(f"\nSentence {i+1}:")
            print(f"  Text: {result_df.iloc[i]['text'][:80]}...")
            print(f"  Sentiment: {result_df.iloc[i]['sentiment_label']} (conf: {result_df.iloc[i]['sentiment_confidence']:.3f})")
            print(f"  Hedging: {result_df.iloc[i]['hedging_score']:.3f}")
            print(f"  Risk Escalation: {result_df.iloc[i]['risk_escalation_score']:.3f}")
        
        # Test speaker aggregation (simplified)
        print(f"\n[SPEAKERS] Testing Speaker Aggregation:")
        speaker_summary = result_df.groupby('speaker_norm').agg({
            'sentiment_confidence': 'mean',
            'sentiment_label': 'count',
            'risk_escalation_score': 'mean'
        }).round(3)
        speaker_summary.columns = ['Avg_Confidence', 'Sentence_Count', 'Avg_Risk_Escalation']
        print(f"Speakers analyzed: {len(speaker_summary)}")
        for speaker, row in speaker_summary.iterrows():
            print(f"  {speaker}: {row['Sentence_Count']} sentences, "
                  f"avg confidence: {row['Avg_Confidence']:.3f}")
        
        return result_df
        
    except Exception as e:
        print(f"[FAILED] Sentiment analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_topic_modeling():
    """Test the hybrid topic modeling engine"""
    print("\n" + "="*60)
    print("TESTING TOPIC MODELING")
    print("="*60)
    
    try:
        # Initialize engine
        engine = HybridTopicEngine()
        print("[OK] Hybrid topic engine initialized successfully")
        
        # Create test data
        test_data = create_test_data()
        print(f"[OK] Test data created: {len(test_data)} sentences")
        
        # Process topics
        print("[Processing] Running topic modeling...")
        result_df, metadata = engine.process_quarter_data('TestBank', 'Q1_2025', test_data)
        print("[OK] Topic modeling completed")
        
        # Display results
        print(f"\n[STATS] Topic Modeling Results:")
        print(f"Total sentences processed: {len(result_df)}")
        
        # Topic distribution
        topic_dist = result_df['final_topic'].value_counts()
        print(f"\nTopic Distribution:")
        for topic, count in topic_dist.items():
            print(f"  {topic}: {count} ({count/len(result_df)*100:.1f}%)")
        
        # Topic type distribution
        type_dist = result_df['topic_type'].value_counts()
        print(f"\nTopic Type Distribution:")
        for topic_type, count in type_dist.items():
            print(f"  {topic_type}: {count} ({count/len(result_df)*100:.1f}%)")
        
        # Average confidence
        print(f"\nAverage topic confidence: {result_df['topic_confidence'].mean():.3f}")
        
        # Processing statistics
        print(f"\n[METRICS] Processing Statistics:")
        stats = metadata['statistics']
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Sample results
        print(f"\n[SAMPLE] Sample Topic Assignments:")
        for i in range(min(5, len(result_df))):
            row = result_df.iloc[i]
            print(f"\nSentence {i+1}:")
            print(f"  Text: {row['text'][:80]}...")
            print(f"  Final Topic: {row['final_topic']}")
            print(f"  Topic Type: {row['topic_type']}")
            print(f"  Confidence: {row['topic_confidence']:.3f}")
        
        # Seed topic performance
        seed_topics = result_df[result_df['topic_type'] == 'seed']
        if len(seed_topics) > 0:
            print(f"\n[TARGET] Seed Topic Performance:")
            seed_dist = seed_topics['final_topic'].value_counts()
            for topic, count in seed_dist.items():
                avg_conf = seed_topics[seed_topics['final_topic'] == topic]['topic_confidence'].mean()
                print(f"  {topic}: {count} sentences (avg conf: {avg_conf:.3f})")
        
        return result_df, metadata
        
    except Exception as e:
        print(f"[FAILED] Topic modeling test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_integration():
    """Test integration of sentiment and topic analysis"""
    print("\n" + "="*60)
    print("TESTING INTEGRATION")
    print("="*60)
    
    try:
        # Initialize components
        analyzer = FinBERTAnalyzer()
        engine = HybridTopicEngine()
        
        # Create test data
        test_data = create_test_data()
        
        # Run topic modeling
        print("[Processing] Running integrated analysis...")
        topic_result, topic_metadata = engine.process_quarter_data('TestBank', 'Q1_2025', test_data)
        
        # Run sentiment analysis
        final_result = analyzer.analyze_dataframe(topic_result)
        
        print("[OK] Integrated analysis completed")
        
        # Combined analysis
        print(f"\n[STATS] Integrated Results:")
        print(f"Total sentences: {len(final_result)}")
        
        # Sentiment by topic
        print(f"\n[SENTIMENT] Sentiment by Topic:")
        topic_sentiment = final_result.groupby('final_topic')['sentiment_label'].value_counts().unstack(fill_value=0)
        print(topic_sentiment)
        
        # Risk analysis
        print(f"\n[WARNING]  Risk Analysis:")
        risk_topics = ['credit_risk', 'operational_risk', 'market_risk', 'regulatory_risk']
        risk_data = final_result[final_result['final_topic'].isin(risk_topics)]
        
        if len(risk_data) > 0:
            print(f"Risk-related sentences: {len(risk_data)}")
            print(f"Average risk escalation score: {risk_data['risk_escalation_score'].mean():.3f}")
            print(f"Average uncertainty score: {risk_data['uncertainty_score'].mean():.3f}")
            
            # High-risk sentences
            high_risk = risk_data[
                (risk_data['sentiment_label'] == 'negative') & 
                (risk_data['risk_escalation_score'] > 0.05)
            ]
            print(f"High-risk sentences: {len(high_risk)}")
        
        # Speaker analysis
        print(f"\n[SPEAKERS] Speaker Risk Profile:")
        speaker_risk = final_result.groupby('speaker_norm').agg({
            'risk_escalation_score': 'mean',
            'sentiment_label': lambda x: (x == 'negative').sum() / len(x),
            'uncertainty_score': 'mean'
        }).round(3)
        speaker_risk.columns = ['Avg_Risk_Escalation', 'Negative_Sentiment_Rate', 'Avg_Uncertainty']
        print(speaker_risk)
        
        return final_result
        
    except Exception as e:
        print(f"[FAILED] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all Phase 2 tests"""
    print("Starting Phase 2 NLP Component Testing")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Sentiment Analysis
    sentiment_result = test_sentiment_analysis()
    
    # Test 2: Topic Modeling  
    topic_result, topic_metadata = test_topic_modeling()
    
    # Test 3: Integration
    integrated_result = test_integration()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    tests_passed = 0
    total_tests = 3
    
    if sentiment_result is not None:
        print("[OK] Sentiment Analysis: PASSED")
        tests_passed += 1
    else:
        print("[FAILED] Sentiment Analysis: FAILED")
    
    if topic_result is not None:
        print("[OK] Topic Modeling: PASSED")
        tests_passed += 1
    else:
        print("[FAILED] Topic Modeling: FAILED")
    
    if integrated_result is not None:
        print("[OK] Integration: PASSED")
        tests_passed += 1
    else:
        print("[FAILED] Integration: FAILED")
    
    print(f"\n[STATS] Overall Result: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("[SUCCESS] All Phase 2 NLP components are working correctly!")
        print("[OK] Ready to proceed to Phase 3: Statistical Analysis")
    else:
        print("[WARNING]  Some tests failed. Please review the errors above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)