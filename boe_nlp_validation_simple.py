#!/usr/bin/env python3
"""
Bank of England Mosaic Lens - Simple NLP Validation Script
==========================================================

Simplified validation script for Topic Modeling and Sentiment Analysis
that works with basic Python libraries for Bank of England code scrutiny.

This script demonstrates the NLP workflow and validation methodology
without requiring complex dependencies.

Usage:
    python boe_nlp_validation_simple.py

Output:
    - NLP workflow demonstration
    - Validation metrics and results
    - Code transparency for audit
"""

import re
import time
import math
from datetime import datetime
from collections import Counter, defaultdict

print("=" * 80)
print("BANK OF ENGLAND MOSAIC LENS - SIMPLE NLP VALIDATION")
print("=" * 80)
print(f"Validation started at: {datetime.now()}")
print("Purpose: Code scrutiny and NLP workflow demonstration")
print("Using: Core Python libraries only (no external dependencies)")
print("=" * 80)

class SimpleNLPValidator:
    """
    Simplified NLP validation for Bank of England code scrutiny.
    
    This implementation uses only core Python libraries to demonstrate:
    1. Text preprocessing methodology
    2. Basic topic extraction through keyword analysis
    3. Rule-based sentiment analysis
    4. Contradiction detection logic
    5. Performance metrics calculation
    """
    
    def __init__(self):
        """Initialize the validator with core components."""
        self.validation_results = {}
        self.performance_metrics = {}
        
        # Financial domain lexicon for sentiment analysis
        self.financial_sentiment_lexicon = {
            # Positive terms
            'strong': 0.6, 'robust': 0.6, 'solid': 0.4, 'excellent': 0.8,
            'outstanding': 0.7, 'growth': 0.5, 'improved': 0.5, 'stable': 0.3,
            'resilient': 0.5, 'conservative': 0.2, 'prudent': 0.3, 'healthy': 0.4,
            'enhanced': 0.4, 'efficient': 0.4, 'profitable': 0.6, 'successful': 0.6,
            
            # Negative terms
            'challenging': -0.5, 'volatile': -0.4, 'weak': -0.5, 'deteriorating': -0.7,
            'impacted': -0.3, 'disruptions': -0.5, 'turbulence': -0.6, 'losses': -0.6,
            'declined': -0.5, 'decreased': -0.4, 'pressure': -0.3, 'uncertainty': -0.4,
            'risk': -0.2, 'concern': -0.4, 'difficult': -0.5, 'adverse': -0.6,
            
            # Neutral/context-dependent terms
            'maintained': 0.1, 'continued': 0.1, 'remains': 0.0, 'compliance': 0.2,
            'regulatory': 0.0, 'capital': 0.1, 'provisions': -0.1, 'monitoring': 0.1
        }
        
        # Topic keywords for classification
        self.topic_keywords = {
            'regulatory_compliance': [
                'regulatory', 'compliance', 'capital', 'basel', 'stress', 'test',
                'requirements', 'standards', 'supervisory', 'authorities', 'cet1',
                'buffers', 'framework', 'guidelines', 'ratios'
            ],
            'financial_performance': [
                'revenue', 'profit', 'earnings', 'growth', 'margin', 'performance',
                'roe', 'returns', 'profitability', 'efficiency', 'cost', 'income',
                'diversification', 'business', 'results', 'quarterly'
            ],
            'credit_risk': [
                'credit', 'loan', 'npl', 'provision', 'default', 'underwriting',
                'portfolio', 'asset', 'quality', 'impairment', 'coverage',
                'exposures', 'losses', 'disciplined', 'conservative'
            ],
            'operational_risk': [
                'operational', 'technology', 'cyber', 'security', 'process',
                'automation', 'digital', 'efficiency', 'resilience', 'disruptions',
                'systems', 'capabilities', 'monitoring', 'infrastructure'
            ],
            'market_risk': [
                'market', 'trading', 'volatility', 'var', 'hedging', 'currency',
                'interest', 'rate', 'foreign', 'exchange', 'risk', 'management',
                'exposure', 'limits', 'appetite', 'conditions'
            ]
        }
        
        # Common stop words
        self.stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'a', 'an', 'we', 'our', 'us', 'bank', 'banking', 'financial',
            'institution', 'company', 'business', 'quarter', 'year'
        }
    
    def generate_validation_dataset(self):
        """Generate comprehensive validation dataset with known ground truth."""
        print("\n Generating validation dataset...")
        
        # Financial documents with known topics and sentiments
        documents = [
            # Regulatory compliance (positive sentiment)
            "Our capital ratios remain strong with CET1 at 12.5%, well above regulatory requirements. We continue to meet all Basel III standards.",
            "Regulatory compliance remains a top priority. We have enhanced our risk management framework and strengthened our capital position.",
            "The bank maintains robust capital buffers and liquidity positions. Our stress testing results demonstrate resilience under adverse scenarios.",
            
            # Financial performance (positive sentiment)
            "Revenue growth of 8% year-over-year driven by strong net interest margin expansion. Operating efficiency improved significantly.",
            "Solid financial performance with ROE of 12.3% and strong profit margins. Revenue diversification continues to support sustainable growth.",
            "Excellent quarterly results with earnings per share up 15%. Our diversified business model continues to deliver consistent returns.",
            
            # Credit risk (mixed sentiment)
            "Credit quality remains stable with NPL ratio at 1.8%. Provision coverage is adequate and we maintain conservative underwriting standards.",
            "Our loan portfolio shows strong performance with low default rates. Credit provisions have been increased prudently given economic uncertainty.",
            "Challenging economic conditions have led to higher provisions, but our diversified portfolio and strong risk management provide resilience.",
            
            # Operational risk (mixed sentiment)
            "Operational efficiency initiatives delivered cost savings of $50M. Technology investments continue to enhance our digital capabilities.",
            "Cyber security remains a priority with enhanced monitoring and threat detection capabilities. Operational risk framework has been strengthened.",
            "Technology disruptions caused temporary service interruptions but our business continuity plans ensured minimal customer impact.",
            
            # Market risk (negative sentiment)
            "Trading revenue was impacted by market volatility but risk management kept losses within acceptable limits. VaR models performed well.",
            "Volatile market conditions resulted in trading losses, but our risk management framework prevented significant exposure to tail risks.",
            "Market turbulence created challenging conditions for our trading business, with revenues declining due to reduced client activity."
        ]
        
        # Ground truth labels
        topic_labels = [
            'regulatory_compliance', 'regulatory_compliance', 'regulatory_compliance',
            'financial_performance', 'financial_performance', 'financial_performance',
            'credit_risk', 'credit_risk', 'credit_risk',
            'operational_risk', 'operational_risk', 'operational_risk',
            'market_risk', 'market_risk', 'market_risk'
        ]
        
        sentiment_labels = [
            'positive', 'positive', 'positive',
            'positive', 'positive', 'positive',
            'positive', 'positive', 'negative',
            'positive', 'positive', 'negative',
            'neutral', 'negative', 'negative'
        ]
        
        print(f"    Generated {len(documents)} documents")
        print(f"    Topics: {set(topic_labels)}")
        print(f"    Sentiment distribution: {Counter(sentiment_labels)}")
        
        return documents, topic_labels, sentiment_labels
    
    def preprocess_text(self, text):
        """Preprocess text using basic Python string operations."""
        # Convert to lowercase
        text = text.lower()
        
        # Extract words (alphanumeric only)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        
        # Remove stop words
        words = [word for word in words if word not in self.stop_words]
        
        return words
    
    def classify_topic(self, text):
        """Classify document topic using keyword matching."""
        words = self.preprocess_text(text)
        word_set = set(words)
        
        topic_scores = {}
        
        for topic, keywords in self.topic_keywords.items():
            # Calculate overlap between document words and topic keywords
            overlap = len(word_set.intersection(set(keywords)))
            # Normalize by topic keyword count
            score = overlap / len(keywords) if keywords else 0
            topic_scores[topic] = score
        
        # Return topic with highest score
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            confidence = topic_scores[best_topic]
            return best_topic, confidence, topic_scores
        else:
            return 'unknown', 0.0, {}
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using financial domain lexicon."""
        words = self.preprocess_text(text)
        
        sentiment_scores = []
        matched_words = []
        
        for word in words:
            if word in self.financial_sentiment_lexicon:
                score = self.financial_sentiment_lexicon[word]
                sentiment_scores.append(score)
                matched_words.append((word, score))
        
        if sentiment_scores:
            # Calculate compound sentiment score
            compound_score = sum(sentiment_scores) / len(sentiment_scores)
            
            # Apply financial context weighting
            financial_keywords = ['capital', 'risk', 'regulatory', 'compliance', 'credit']
            financial_weight = sum(1 for keyword in financial_keywords if keyword in text.lower())
            
            if financial_weight > 0:
                # Amplify sentiment in financial context
                compound_score *= (1 + financial_weight * 0.1)
                compound_score = max(-1, min(1, compound_score))  # Clip to [-1, 1]
        else:
            compound_score = 0.0
        
        # Classify sentiment
        if compound_score >= 0.05:
            classification = 'positive'
        elif compound_score <= -0.05:
            classification = 'negative'
        else:
            classification = 'neutral'
        
        return {
            'compound': compound_score,
            'classification': classification,
            'matched_words': matched_words,
            'word_count': len(matched_words)
        }
    
    def detect_contradictions(self, text):
        """Detect contradictions between positive language and negative indicators."""
        sentences = re.split(r'[.!?]+', text)
        
        if len(sentences) < 2:
            return {'has_contradiction': False, 'confidence': 0.0, 'details': 'Insufficient sentences'}
        
        sentence_sentiments = []
        
        for sentence in sentences:
            if sentence.strip():
                sentiment = self.analyze_sentiment(sentence)
                sentence_sentiments.append(sentiment['compound'])
        
        if len(sentence_sentiments) >= 2:
            # Check for opposing sentiments
            max_sentiment = max(sentence_sentiments)
            min_sentiment = min(sentence_sentiments)
            
            # Detect contradiction (strong positive and strong negative)
            sentiment_diff = max_sentiment - min_sentiment
            has_contradiction = (max_sentiment > 0.3 and min_sentiment < -0.3) or sentiment_diff > 0.6
            
            confidence = min(sentiment_diff, 1.0)
            
            return {
                'has_contradiction': has_contradiction,
                'confidence': confidence,
                'sentiment_range': (min_sentiment, max_sentiment),
                'sentence_count': len(sentence_sentiments)
            }
        
        return {'has_contradiction': False, 'confidence': 0.0, 'details': 'Insufficient analysis'}
    
    def validate_topic_classification(self, documents, true_labels):
        """Validate topic classification accuracy."""
        print("\n Validating Topic Classification...")
        start_time = time.time()
        
        predictions = []
        confidences = []
        
        for doc in documents:
            topic, confidence, scores = self.classify_topic(doc)
            predictions.append(topic)
            confidences.append(confidence)
        
        # Calculate accuracy
        correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
        accuracy = correct / len(true_labels)
        
        # Calculate per-topic performance
        topic_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for true, pred in zip(true_labels, predictions):
            topic_performance[true]['total'] += 1
            if true == pred:
                topic_performance[true]['correct'] += 1
        
        processing_time = time.time() - start_time
        
        results = {
            'accuracy': accuracy,
            'predictions': predictions,
            'confidences': confidences,
            'topic_performance': dict(topic_performance),
            'processing_time': processing_time
        }
        
        # Print results
        print(f"    Topic Classification Results:")
        print(f"   Overall Accuracy: {accuracy:.3f} ({correct}/{len(true_labels)})")
        print(f"   Average Confidence: {sum(confidences)/len(confidences):.3f}")
        print(f"   Processing Time: {processing_time:.3f} seconds")
        
        print(f"    Per-Topic Performance:")
        for topic, perf in topic_performance.items():
            topic_acc = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
            print(f"      {topic}: {topic_acc:.3f} ({perf['correct']}/{perf['total']})")
        
        threshold_status = " PASS" if accuracy >= 0.70 else " FAIL"
        print(f"   Status (>=70%): {threshold_status}")
        
        self.validation_results['topic_classification'] = results
        return results
    
    def validate_sentiment_analysis(self, documents, true_labels):
        """Validate sentiment analysis accuracy."""
        print("\n Validating Sentiment Analysis...")
        start_time = time.time()
        
        predictions = []
        scores = []
        
        for doc in documents:
            sentiment = self.analyze_sentiment(doc)
            predictions.append(sentiment['classification'])
            scores.append(sentiment['compound'])
        
        # Calculate accuracy
        correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
        accuracy = correct / len(true_labels)
        
        # Calculate sentiment distribution
        pred_distribution = Counter(predictions)
        true_distribution = Counter(true_labels)
        
        processing_time = time.time() - start_time
        
        results = {
            'accuracy': accuracy,
            'predictions': predictions,
            'scores': scores,
            'pred_distribution': dict(pred_distribution),
            'true_distribution': dict(true_distribution),
            'processing_time': processing_time
        }
        
        # Print results
        print(f"    Sentiment Analysis Results:")
        print(f"   Overall Accuracy: {accuracy:.3f} ({correct}/{len(true_labels)})")
        print(f"   Average Score Magnitude: {sum(abs(s) for s in scores)/len(scores):.3f}")
        print(f"   Processing Time: {processing_time:.3f} seconds")
        
        print(f"    Prediction Distribution: {pred_distribution}")
        print(f"    True Distribution: {true_distribution}")
        
        threshold_status = " PASS" if accuracy >= 0.80 else " FAIL"
        print(f"   Status (>=80%): {threshold_status}")
        
        self.validation_results['sentiment_analysis'] = results
        return results
    
    def validate_contradiction_detection(self, documents):
        """Validate contradiction detection capability."""
        print("\n Validating Contradiction Detection...")
        
        # Test cases with known contradictions
        test_cases = [
            {
                'text': "We maintain strong credit quality and robust risk management. However, NPL ratios have increased significantly to 8.5% this quarter.",
                'expected': True
            },
            {
                'text': "Excellent operational performance with enhanced efficiency. Technology disruptions caused major service outages affecting customers.",
                'expected': True
            },
            {
                'text': "Capital ratios remain strong at 12.5% CET1. We continue to meet all regulatory requirements comfortably.",
                'expected': False
            },
            {
                'text': "Challenging market conditions impacted trading revenue. Risk management frameworks performed as expected during volatility.",
                'expected': False
            }
        ]
        
        correct_detections = 0
        results_details = []
        
        for case in test_cases:
            result = self.detect_contradictions(case['text'])
            detected = result['has_contradiction']
            expected = case['expected']
            
            if detected == expected:
                correct_detections += 1
            
            results_details.append({
                'text': case['text'][:60] + "...",
                'expected': expected,
                'detected': detected,
                'confidence': result['confidence'],
                'correct': detected == expected
            })
        
        accuracy = correct_detections / len(test_cases)
        
        results = {
            'accuracy': accuracy,
            'correct_detections': correct_detections,
            'total_cases': len(test_cases),
            'details': results_details
        }
        
        # Print results
        print(f"    Contradiction Detection Results:")
        print(f"   Accuracy: {accuracy:.3f} ({correct_detections}/{len(test_cases)})")
        
        for detail in results_details:
            status = "" if detail['correct'] else ""
            print(f"   {status} {detail['text']} (confidence: {detail['confidence']:.2f})")
        
        threshold_status = " PASS" if accuracy >= 0.75 else " FAIL"
        print(f"   Status (>=75%): {threshold_status}")
        
        self.validation_results['contradiction_detection'] = results
        return results
    
    def generate_comprehensive_report(self):
        """Generate comprehensive validation report."""
        print("\n" + "=" * 80)
        print(" COMPREHENSIVE NLP VALIDATION REPORT")
        print("=" * 80)
        
        # Extract key metrics
        topic_accuracy = self.validation_results['topic_classification']['accuracy']
        sentiment_accuracy = self.validation_results['sentiment_analysis']['accuracy']
        contradiction_accuracy = self.validation_results['contradiction_detection']['accuracy']
        
        print(f"\n VALIDATION SUMMARY:")
        print(f"   Topic Classification Accuracy: {topic_accuracy:.3f} (threshold >=0.70)")
        print(f"   Sentiment Analysis Accuracy: {sentiment_accuracy:.3f} (threshold >=0.80)")
        print(f"   Contradiction Detection Accuracy: {contradiction_accuracy:.3f} (threshold >=0.75)")
        
        # Pass/fail assessment
        topic_pass = topic_accuracy >= 0.70
        sentiment_pass = sentiment_accuracy >= 0.80
        contradiction_pass = contradiction_accuracy >= 0.75
        
        overall_pass = topic_pass and sentiment_pass and contradiction_pass
        
        print(f"\n COMPONENT ASSESSMENT:")
        print(f"   Topic Classification: {' PASS' if topic_pass else ' FAIL'}")
        print(f"   Sentiment Analysis: {' PASS' if sentiment_pass else ' FAIL'}")
        print(f"   Contradiction Detection: {' PASS' if contradiction_pass else ' FAIL'}")
        print(f"   Overall NLP System: {' APPROVED' if overall_pass else ' NEEDS IMPROVEMENT'}")
        
        # Performance summary
        total_processing_time = (
            self.validation_results['topic_classification']['processing_time'] +
            self.validation_results['sentiment_analysis']['processing_time']
        )
        
        print(f"\n PERFORMANCE SUMMARY:")
        print(f"   Total Processing Time: {total_processing_time:.3f} seconds")
        print(f"   Documents Processed: 15")
        print(f"   Average Time per Document: {total_processing_time/15:.3f} seconds")
        
        # Bank of England compliance assessment
        print(f"\n BANK OF ENGLAND COMPLIANCE:")
        if overall_pass:
            print("    NLP system meets validation criteria for supervisory use")
            print("    Code is transparent and auditable")
            print("    Methodology is documented and reproducible")
            print("    Ready for deployment with monitoring")
        else:
            print("    System requires improvement before supervisory deployment")
            print("    Address failing components and revalidate")
            print("    Consider additional training data or methodology refinement")
        
        return {
            'overall_pass': overall_pass,
            'topic_accuracy': topic_accuracy,
            'sentiment_accuracy': sentiment_accuracy,
            'contradiction_accuracy': contradiction_accuracy,
            'processing_time': total_processing_time
        }

def main():
    """Main validation execution function."""
    print(" Starting NLP validation with core Python libraries...")
    
    # Initialize validator
    validator = SimpleNLPValidator()
    
    # Generate validation dataset
    documents, topic_labels, sentiment_labels = validator.generate_validation_dataset()
    
    # Run validations
    topic_results = validator.validate_topic_classification(documents, topic_labels)
    sentiment_results = validator.validate_sentiment_analysis(documents, sentiment_labels)
    contradiction_results = validator.validate_contradiction_detection(documents)
    
    # Generate comprehensive report
    final_report = validator.generate_comprehensive_report()
    
    print("\n" + "=" * 80)
    print(" NLP VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Validation completed at: {datetime.now()}")
    print("Results demonstrate NLP workflow transparency for Bank of England review.")
    print("All code is auditable and uses only core Python libraries.")
    
    return validator, final_report

if __name__ == "__main__":
    validator, report = main()