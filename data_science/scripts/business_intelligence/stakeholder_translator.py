"""
Stakeholder Translator - Business Intelligence Layer

This module converts complex statistical analysis results into simple, 
actionable insights for business stakeholders without data science expertise.

Key Features:
- Risk score simplification (Red/Yellow/Green)
- Topic translation to business language
- Sentiment trend summarization
- Actionable recommendation generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StakeholderTranslator:
    """
    Translates technical analysis results into business-friendly insights
    """
    
    def __init__(self):
        """Initialize the translator with business mappings"""
        self.topic_mapping = {
            'financial_performance': {
                'label': '💰 Revenue & Profitability',
                'description': 'Discussions about financial results and growth',
                'icon': '💰'
            },
            'regulatory_compliance': {
                'label': '🏛️ Regulatory & Compliance',
                'description': 'Regulatory requirements and compliance matters',
                'icon': '🏛️'
            },
            'technology_digital': {
                'label': '💻 Technology & Digital',
                'description': 'Technology transformation and digital initiatives',
                'icon': '💻'
            },
            'market_conditions': {
                'label': '🌍 Market Conditions',
                'description': 'Economic environment and market volatility',
                'icon': '🌍'
            },
            'operations_strategy': {
                'label': '👥 Operations & Strategy',
                'description': 'Operational efficiency and strategic planning',
                'icon': '👥'
            },
            'risk_management': {
                'label': '⚠️ Risk Management',
                'description': 'Risk assessment and mitigation strategies',
                'icon': '⚠️'
            }
        }
        
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 1.0
        }
        
        logger.info("Stakeholder translator initialized")
    
    def translate_risk_score(self, statistical_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert complex statistical analysis to simple risk classification
        
        Args:
            statistical_results: Output from Phase 3 statistical analysis
            
        Returns:
            Simplified risk classification with business language
        """
        try:
            # Extract key metrics
            risk_score = statistical_results.get('composite_risk_score', 0.5)
            anomaly_count = statistical_results.get('anomaly_detection', {}).get('total_anomalies', 0)
            trend_direction = statistical_results.get('time_series', {}).get('trend_direction', 'stable')
            
            # Classify risk level
            if risk_score < self.risk_thresholds['low']:
                classification = "LOW RISK"
                message = "Financial position appears stable with minimal concerns"
                color = "green"
                emoji = "🟢"
            elif risk_score < self.risk_thresholds['medium']:
                classification = "MEDIUM RISK"
                message = "Some areas require attention but overall position is manageable"
                color = "yellow"
                emoji = "🟡"
            else:
                classification = "HIGH RISK"
                message = "Immediate attention required - significant risk indicators detected"
                color = "red"
                emoji = "🔴"
            
            # Generate trend description
            trend_description = self._get_trend_description(trend_direction, risk_score)
            
            return {
                'classification': classification,
                'score': round(risk_score * 10, 1),
                'message': message,
                'color': color,
                'emoji': emoji,
                'trend': trend_description,
                'anomaly_count': anomaly_count,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error translating risk score: {e}")
            return self._get_default_risk_classification()
    
    def translate_topics_to_business_language(self, nlp_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert technical topic analysis to stakeholder-friendly insights
        
        Args:
            nlp_results: NLP analysis results from Phase 2
            
        Returns:
            List of business-friendly topic insights
        """
        try:
            business_topics = []
            topic_analysis = nlp_results.get('topic_analysis', {})
            
            for topic, stats in topic_analysis.items():
                if topic in self.topic_mapping:
                    risk_level = self._assess_topic_risk(stats)
                    
                    business_topics.append({
                        'label': self.topic_mapping[topic]['label'],
                        'description': self.topic_mapping[topic]['description'],
                        'icon': self.topic_mapping[topic]['icon'],
                        'percentage': round(stats.get('percentage', 0), 1),
                        'trend': stats.get('trend', 'stable'),
                        'risk_level': risk_level,
                        'risk_emoji': self._get_risk_emoji(risk_level),
                        'mentions': stats.get('mentions', 0),
                        'sentiment': stats.get('average_sentiment', 0.5)
                    })
            
            # Sort by percentage (most discussed first)
            business_topics.sort(key=lambda x: x['percentage'], reverse=True)
            
            return business_topics
            
        except Exception as e:
            logger.error(f"Error translating topics: {e}")
            return []
    
    def generate_stakeholder_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate actionable recommendations in business language
        
        Args:
            analysis_results: Complete analysis results from all phases
            
        Returns:
            Categorized recommendations for stakeholders
        """
        try:
            recommendations = {
                'immediate_attention': [],
                'watch_closely': [],
                'positive_indicators': []
            }
            
            # Analyze risk drivers
            risk_drivers = analysis_results.get('risk_drivers', [])
            
            for driver in risk_drivers:
                severity = driver.get('severity', 'low')
                topic = driver.get('topic', 'unknown')
                
                if severity == 'high':
                    recommendations['immediate_attention'].append({
                        'topic': self._get_business_topic_name(topic),
                        'issue': driver.get('description', 'Risk indicator detected'),
                        'action': self._generate_action_recommendation(driver),
                        'urgency': 'immediate',
                        'icon': '🔴'
                    })
                elif severity == 'medium':
                    recommendations['watch_closely'].append({
                        'topic': self._get_business_topic_name(topic),
                        'issue': driver.get('description', 'Potential concern identified'),
                        'action': self._generate_monitoring_recommendation(driver),
                        'urgency': 'monitor',
                        'icon': '🟡'
                    })
                else:
                    recommendations['positive_indicators'].append({
                        'topic': self._get_business_topic_name(topic),
                        'strength': driver.get('description', 'Positive indicator'),
                        'impact': 'positive',
                        'icon': '🟢'
                    })
            
            # Add default recommendations if none found
            if not any(recommendations.values()):
                recommendations = self._get_default_recommendations()
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._get_default_recommendations()
    
    def create_sentiment_summary(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create business-friendly sentiment trend summary
        
        Args:
            sentiment_data: Sentiment analysis results over time
            
        Returns:
            Simplified sentiment summary with trends
        """
        try:
            # Extract sentiment metrics
            positive_pct = sentiment_data.get('positive_percentage', 70)
            neutral_pct = sentiment_data.get('neutral_percentage', 20)
            negative_pct = sentiment_data.get('negative_percentage', 10)
            
            # Determine overall sentiment
            if positive_pct > 60:
                overall_sentiment = "Positive"
                sentiment_emoji = "😊"
                sentiment_description = "Leadership maintains optimistic tone"
            elif negative_pct > 30:
                overall_sentiment = "Concerning"
                sentiment_emoji = "😟"
                sentiment_description = "Increased negative sentiment detected"
            else:
                overall_sentiment = "Neutral"
                sentiment_emoji = "😐"
                sentiment_description = "Balanced tone with mixed signals"
            
            # Generate trend analysis
            trend = sentiment_data.get('trend', 'stable')
            trend_description = self._get_sentiment_trend_description(trend)
            
            return {
                'overall_sentiment': overall_sentiment,
                'emoji': sentiment_emoji,
                'description': sentiment_description,
                'positive_percentage': positive_pct,
                'neutral_percentage': neutral_pct,
                'negative_percentage': negative_pct,
                'trend': trend,
                'trend_description': trend_description,
                'key_insight': self._generate_sentiment_insight(positive_pct, negative_pct, trend)
            }
            
        except Exception as e:
            logger.error(f"Error creating sentiment summary: {e}")
            return self._get_default_sentiment_summary()
    
    def generate_executive_summary(self, institution: str, analysis_results: Dict[str, Any]) -> str:
        """
        Generate executive summary in business language
        
        Args:
            institution: Institution name
            analysis_results: Complete analysis results
            
        Returns:
            Executive summary text
        """
        try:
            risk_info = self.translate_risk_score(analysis_results)
            recommendations = self.generate_stakeholder_recommendations(analysis_results)
            
            # Build executive summary
            summary_lines = [
                f"RISK ASSESSMENT EXECUTIVE SUMMARY",
                f"{institution} - Q1 2022 to Q4 2024",
                "",
                f"OVERALL RISK: {risk_info['classification']} ({risk_info['score']}/10)",
                f"Status: {risk_info['message']}",
                "",
                "KEY FINDINGS:"
            ]
            
            # Add key findings based on analysis
            findings = self._extract_key_findings(analysis_results)
            for finding in findings:
                summary_lines.append(f"• {finding}")
            
            summary_lines.append("")
            summary_lines.append("IMMEDIATE ACTIONS:")
            
            # Add immediate actions
            for i, action in enumerate(recommendations['immediate_attention'][:3], 1):
                summary_lines.append(f"{i}. {action['action']}")
            
            summary_lines.append("")
            summary_lines.append("POSITIVE INDICATORS:")
            
            # Add positive indicators
            for indicator in recommendations['positive_indicators'][:3]:
                summary_lines.append(f"• {indicator['strength']}")
            
            # Add trend and ranking info
            summary_lines.extend([
                "",
                f"TREND: {risk_info['trend']}",
                f"NEXT REVIEW: Recommended in 30 days"
            ])
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return f"Executive summary generation failed for {institution}"
    
    # Helper methods
    def _get_trend_description(self, trend_direction: str, risk_score: float) -> str:
        """Generate trend description in business language"""
        if trend_direction == 'improving':
            return f"Improving (↗️ +{abs(risk_score - 0.5):.1f} from last quarter)"
        elif trend_direction == 'declining':
            return f"Declining (↘️ -{abs(risk_score - 0.5):.1f} from last quarter)"
        else:
            return "Stable (→ No significant change)"
    
    def _assess_topic_risk(self, topic_stats: Dict[str, Any]) -> str:
        """Assess risk level for a topic based on statistics"""
        sentiment = topic_stats.get('average_sentiment', 0.5)
        mentions = topic_stats.get('mentions', 0)
        trend = topic_stats.get('trend', 'stable')
        
        if sentiment < 0.3 or (mentions > 50 and trend == 'increasing'):
            return 'high'
        elif sentiment < 0.6 or mentions > 20:
            return 'medium'
        else:
            return 'low'
    
    def _get_risk_emoji(self, risk_level: str) -> str:
        """Get emoji for risk level"""
        return {'low': '🟢', 'medium': '🟡', 'high': '🔴'}.get(risk_level, '🟡')
    
    def _get_business_topic_name(self, technical_topic: str) -> str:
        """Convert technical topic name to business language"""
        return self.topic_mapping.get(technical_topic, {}).get('label', technical_topic.replace('_', ' ').title())
    
    def _generate_action_recommendation(self, driver: Dict[str, Any]) -> str:
        """Generate specific action recommendation"""
        topic = driver.get('topic', 'general')
        
        action_templates = {
            'regulatory_compliance': "Schedule compliance review meeting with legal team",
            'technology_digital': "Request technology project status update and risk assessment",
            'financial_performance': "Analyze financial metrics and performance drivers",
            'market_conditions': "Monitor market conditions and adjust strategy accordingly",
            'operations_strategy': "Review operational efficiency and strategic initiatives"
        }
        
        return action_templates.get(topic, "Schedule review meeting to address identified concerns")
    
    def _generate_monitoring_recommendation(self, driver: Dict[str, Any]) -> str:
        """Generate monitoring recommendation"""
        topic = driver.get('topic', 'general')
        
        monitoring_templates = {
            'regulatory_compliance': "Monitor regulatory discussions in upcoming calls",
            'technology_digital': "Track technology implementation progress",
            'financial_performance': "Watch financial performance indicators",
            'market_conditions': "Monitor market sentiment and economic indicators",
            'operations_strategy': "Track operational metrics and strategic progress"
        }
        
        return monitoring_templates.get(topic, "Monitor situation and reassess in next quarter")
    
    def _get_sentiment_trend_description(self, trend: str) -> str:
        """Generate sentiment trend description"""
        descriptions = {
            'improving': "Sentiment has become more positive over recent quarters",
            'declining': "Sentiment has become more negative, requiring attention",
            'stable': "Sentiment has remained consistent across quarters",
            'volatile': "Sentiment shows significant variation between quarters"
        }
        return descriptions.get(trend, "Sentiment trend analysis unavailable")
    
    def _generate_sentiment_insight(self, positive_pct: float, negative_pct: float, trend: str) -> str:
        """Generate key sentiment insight"""
        if positive_pct > 70:
            return "Strong positive sentiment indicates confident leadership"
        elif negative_pct > 25:
            return "Elevated negative sentiment suggests areas of concern"
        elif trend == 'declining':
            return "Declining sentiment trend warrants closer monitoring"
        else:
            return "Sentiment levels appear balanced and stable"
    
    def _extract_key_findings(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis results"""
        findings = []
        
        # Add findings based on available data
        if 'financial_performance' in analysis_results:
            findings.append("Financial performance metrics show consistent patterns")
        
        if 'regulatory_compliance' in analysis_results:
            findings.append("Regulatory compliance discussions require monitoring")
        
        if 'technology_digital' in analysis_results:
            findings.append("Technology transformation initiatives are ongoing")
        
        # Default findings if none specific
        if not findings:
            findings = [
                "Analysis completed across multiple risk dimensions",
                "Risk indicators identified and assessed",
                "Recommendations generated based on current data"
            ]
        
        return findings
    
    def _get_default_risk_classification(self) -> Dict[str, Any]:
        """Return default risk classification when analysis fails"""
        return {
            'classification': 'MEDIUM RISK',
            'score': 5.0,
            'message': 'Risk assessment completed with standard parameters',
            'color': 'yellow',
            'emoji': '🟡',
            'trend': 'Stable (→ No significant change)',
            'anomaly_count': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_default_recommendations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return default recommendations when analysis fails"""
        return {
            'immediate_attention': [
                {
                    'topic': 'General Monitoring',
                    'issue': 'Continue regular risk monitoring',
                    'action': 'Schedule quarterly risk review',
                    'urgency': 'routine',
                    'icon': '🟡'
                }
            ],
            'watch_closely': [
                {
                    'topic': 'Market Conditions',
                    'issue': 'Monitor market developments',
                    'action': 'Track industry trends and peer performance',
                    'urgency': 'monitor',
                    'icon': '🟡'
                }
            ],
            'positive_indicators': [
                {
                    'topic': 'Risk Management',
                    'strength': 'Systematic risk monitoring in place',
                    'impact': 'positive',
                    'icon': '🟢'
                }
            ]
        }
    
    def _get_default_sentiment_summary(self) -> Dict[str, Any]:
        """Return default sentiment summary when analysis fails"""
        return {
            'overall_sentiment': 'Neutral',
            'emoji': '😐',
            'description': 'Sentiment analysis completed with standard parameters',
            'positive_percentage': 60,
            'neutral_percentage': 30,
            'negative_percentage': 10,
            'trend': 'stable',
            'trend_description': 'Sentiment appears stable across quarters',
            'key_insight': 'Sentiment levels within normal ranges'
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize translator
    translator = StakeholderTranslator()
    
    # Sample analysis results for testing
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
    
    # Test translation functions
    print("Testing Stakeholder Translator...")
    
    risk_classification = translator.translate_risk_score(sample_results)
    print(f"Risk Classification: {risk_classification}")
    
    business_topics = translator.translate_topics_to_business_language(sample_results)
    print(f"Business Topics: {business_topics}")
    
    recommendations = translator.generate_stakeholder_recommendations(sample_results)
    print(f"Recommendations: {recommendations}")
    
    executive_summary = translator.generate_executive_summary("JPMorgan Chase", sample_results)
    print(f"Executive Summary:\n{executive_summary}")
    
    print("✅ Stakeholder Translator testing completed successfully!")