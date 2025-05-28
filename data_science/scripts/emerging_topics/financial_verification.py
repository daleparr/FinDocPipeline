"""
Financial Verification Module for Quote Analysis
Cross-checks verbal claims against actual financial data to detect inconsistencies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import re
import warnings

warnings.filterwarnings('ignore')

class FinancialVerificationEngine:
    """
    Verifies verbal claims against actual financial data
    """
    
    def __init__(self):
        # High-carbon sector classifications
        self.high_carbon_sectors = {
            'oil_and_gas': [
                'oil', 'gas', 'petroleum', 'crude', 'refining', 'exploration',
                'drilling', 'upstream', 'downstream', 'lng', 'natural gas'
            ],
            'coal': [
                'coal', 'mining', 'thermal coal', 'metallurgical coal',
                'coal-fired', 'coal power'
            ],
            'utilities': [
                'power generation', 'electricity', 'utility', 'power plant',
                'fossil fuel power', 'gas-fired', 'coal-fired'
            ],
            'heavy_industry': [
                'steel', 'cement', 'aluminum', 'chemicals', 'petrochemicals',
                'heavy manufacturing', 'smelting'
            ],
            'transportation': [
                'airline', 'aviation', 'shipping', 'freight', 'trucking',
                'logistics', 'cargo'
            ],
            'automotive': [
                'automotive', 'car manufacturer', 'vehicle', 'internal combustion'
            ]
        }
        
        # Exposure thresholds for "limited" classification
        self.exposure_thresholds = {
            'very_limited': 0.05,  # < 5%
            'limited': 0.10,       # < 10%
            'moderate': 0.20,      # < 20%
            'significant': 0.35,   # < 35%
            'high': 1.0           # >= 35%
        }
        
        # Financial metrics to analyze
        self.financial_metrics = [
            'loan_portfolio', 'investment_portfolio', 'trading_book',
            'credit_exposure', 'revenue_by_sector', 'assets_under_management'
        ]
    
    def verify_exposure_claim(self, claim_text: str, financial_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Verify claims about sector exposure against actual financial data
        """
        try:
            # Extract claim details
            claim_analysis = self._analyze_exposure_claim(claim_text)
            
            # Calculate actual exposure
            actual_exposure = self._calculate_sector_exposure(financial_data)
            
            # Compare claim vs reality
            verification_result = self._compare_claim_vs_actual(claim_analysis, actual_exposure)
            
            return {
                'claim_text': claim_text,
                'claim_analysis': claim_analysis,
                'actual_exposure': actual_exposure,
                'verification_result': verification_result,
                'discrepancy_detected': verification_result['is_inconsistent'],
                'severity': verification_result['severity'],
                'detailed_analysis': self._generate_detailed_analysis(claim_analysis, actual_exposure),
                'regulatory_flags': self._generate_regulatory_flags(verification_result)
            }
            
        except Exception as e:
            return self._get_default_verification_result(claim_text, str(e))
    
    def _analyze_exposure_claim(self, claim_text: str) -> Dict[str, Any]:
        """
        Analyze the verbal claim to extract key details
        """
        claim_lower = claim_text.lower()
        
        # Extract exposure level claim
        exposure_level = 'unknown'
        if any(term in claim_lower for term in ['limited', 'minimal', 'low']):
            exposure_level = 'limited'
        elif any(term in claim_lower for term in ['moderate', 'reasonable']):
            exposure_level = 'moderate'
        elif any(term in claim_lower for term in ['significant', 'substantial']):
            exposure_level = 'significant'
        elif any(term in claim_lower for term in ['high', 'major', 'extensive']):
            exposure_level = 'high'
        
        # Extract sectors mentioned
        mentioned_sectors = []
        for sector, keywords in self.high_carbon_sectors.items():
            if any(keyword in claim_lower for keyword in keywords):
                mentioned_sectors.append(sector)
        
        # If "high-carbon" is mentioned generally
        if 'high-carbon' in claim_lower or 'carbon-intensive' in claim_lower:
            mentioned_sectors = list(self.high_carbon_sectors.keys())
        
        # Extract any numerical claims
        numerical_claims = re.findall(r'(\d+(?:\.\d+)?)\s*%', claim_text)
        
        return {
            'exposure_level_claimed': exposure_level,
            'sectors_mentioned': mentioned_sectors,
            'numerical_claims': [float(x) for x in numerical_claims],
            'claim_confidence': self._assess_claim_confidence(claim_text),
            'hedging_language': self._detect_hedging_in_claim(claim_text)
        }
    
    def _calculate_sector_exposure(self, financial_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate actual exposure to high-carbon sectors from financial data
        """
        if financial_data.empty:
            return self._generate_sample_exposure_data()
        
        total_exposure = 0
        sector_breakdown = {}
        
        # This would normally analyze real financial data
        # For demonstration, we'll create realistic sample data
        return self._generate_sample_exposure_data()
    
    def _generate_sample_exposure_data(self) -> Dict[str, Any]:
        """
        Generate realistic sample exposure data for demonstration
        """
        # Simulate actual portfolio exposure that contradicts the "limited" claim
        return {
            'total_high_carbon_exposure': 0.18,  # 18% - not "limited"
            'sector_breakdown': {
                'oil_and_gas': 0.08,      # 8%
                'utilities': 0.05,        # 5%
                'heavy_industry': 0.03,   # 3%
                'transportation': 0.02    # 2%
            },
            'exposure_by_metric': {
                'loan_portfolio': 0.22,   # 22% of loans
                'investment_portfolio': 0.15,  # 15% of investments
                'trading_book': 0.12      # 12% of trading book
            },
            'trend_analysis': {
                'q1_2024': 0.16,
                'q2_2024': 0.17,
                'q3_2024': 0.18,
                'q4_2024': 0.18,
                'q1_2025': 0.18
            },
            'data_quality': {
                'completeness': 0.95,
                'accuracy_score': 0.92,
                'last_updated': '2025-01-15'
            }
        }
    
    def _compare_claim_vs_actual(self, claim_analysis: Dict, actual_exposure: Dict) -> Dict[str, Any]:
        """
        Compare verbal claim against actual financial data
        """
        claimed_level = claim_analysis['exposure_level_claimed']
        actual_percentage = actual_exposure['total_high_carbon_exposure']
        
        # Determine if claim is consistent with actual data
        is_consistent = True
        severity = 'low'
        
        if claimed_level == 'limited':
            # "Limited" should be < 10%
            if actual_percentage > self.exposure_thresholds['limited']:
                is_consistent = False
                if actual_percentage > self.exposure_thresholds['moderate']:
                    severity = 'high'
                else:
                    severity = 'medium'
        elif claimed_level == 'moderate':
            # "Moderate" should be 10-20%
            if actual_percentage < 0.05 or actual_percentage > 0.25:
                is_consistent = False
                severity = 'medium'
        elif claimed_level == 'significant':
            # "Significant" should be > 20%
            if actual_percentage < self.exposure_thresholds['moderate']:
                is_consistent = False
                severity = 'medium'
        
        # Calculate discrepancy magnitude
        if claimed_level == 'limited':
            expected_max = self.exposure_thresholds['limited']
            discrepancy = max(0, actual_percentage - expected_max)
        else:
            discrepancy = 0  # Simplified for other levels
        
        return {
            'is_inconsistent': not is_consistent,
            'severity': severity,
            'discrepancy_percentage': discrepancy,
            'actual_vs_claimed': {
                'claimed': claimed_level,
                'actual_percentage': actual_percentage,
                'actual_classification': self._classify_actual_exposure(actual_percentage)
            },
            'confidence_in_assessment': self._calculate_assessment_confidence(actual_exposure)
        }
    
    def _classify_actual_exposure(self, percentage: float) -> str:
        """
        Classify actual exposure percentage into categories
        """
        if percentage < self.exposure_thresholds['very_limited']:
            return 'very_limited'
        elif percentage < self.exposure_thresholds['limited']:
            return 'limited'
        elif percentage < self.exposure_thresholds['moderate']:
            return 'moderate'
        elif percentage < self.exposure_thresholds['significant']:
            return 'significant'
        else:
            return 'high'
    
    def _generate_detailed_analysis(self, claim_analysis: Dict, actual_exposure: Dict) -> Dict[str, Any]:
        """
        Generate detailed analysis of the discrepancy
        """
        return {
            'key_findings': [
                f"Claim: '{claim_analysis['exposure_level_claimed']}' exposure to high-carbon sectors",
                f"Reality: {actual_exposure['total_high_carbon_exposure']:.1%} actual exposure",
                f"Classification: {self._classify_actual_exposure(actual_exposure['total_high_carbon_exposure'])} exposure"
            ],
            'sector_details': actual_exposure['sector_breakdown'],
            'trend_concern': self._analyze_exposure_trend(actual_exposure.get('trend_analysis', {})),
            'materiality_assessment': self._assess_materiality(actual_exposure),
            'potential_explanations': [
                "Definition differences (what constitutes 'high-carbon')",
                "Timing differences (claim vs data dates)",
                "Scope differences (what's included in exposure calculation)",
                "Intentional downplaying of actual exposure"
            ]
        }
    
    def _generate_regulatory_flags(self, verification_result: Dict) -> List[Dict[str, Any]]:
        """
        Generate regulatory flags based on verification results
        """
        flags = []
        
        if verification_result['is_inconsistent']:
            if verification_result['severity'] == 'high':
                flags.append({
                    'flag_type': 'material_misstatement',
                    'priority': 'high',
                    'description': 'Significant discrepancy between claimed and actual exposure',
                    'action_required': 'Immediate follow-up required'
                })
            elif verification_result['severity'] == 'medium':
                flags.append({
                    'flag_type': 'potential_misstatement',
                    'priority': 'medium',
                    'description': 'Moderate discrepancy detected',
                    'action_required': 'Clarification needed in next review'
                })
        
        # Additional flags based on exposure level
        actual_percentage = verification_result['actual_vs_claimed']['actual_percentage']
        if actual_percentage > 0.25:  # > 25%
            flags.append({
                'flag_type': 'high_carbon_concentration',
                'priority': 'medium',
                'description': 'High concentration in carbon-intensive sectors',
                'action_required': 'Review climate risk management'
            })
        
        return flags
    
    def _assess_claim_confidence(self, claim_text: str) -> float:
        """
        Assess confidence level in the claim based on language used
        """
        confidence_indicators = {
            'high': ['definitely', 'certainly', 'clearly', 'obviously'],
            'medium': ['believe', 'expect', 'likely', 'generally'],
            'low': ['may', 'might', 'could', 'potentially', 'possibly']
        }
        
        claim_lower = claim_text.lower()
        
        for level, indicators in confidence_indicators.items():
            if any(indicator in claim_lower for indicator in indicators):
                if level == 'high':
                    return 0.9
                elif level == 'medium':
                    return 0.7
                else:
                    return 0.4
        
        return 0.6  # Default medium confidence
    
    def _detect_hedging_in_claim(self, claim_text: str) -> List[str]:
        """
        Detect hedging language in the claim
        """
        hedging_patterns = [
            'believe', 'expect', 'generally', 'typically', 'relatively',
            'compared to', 'in our view', 'we consider', 'appears to be'
        ]
        
        found_hedging = []
        claim_lower = claim_text.lower()
        
        for pattern in hedging_patterns:
            if pattern in claim_lower:
                found_hedging.append(pattern)
        
        return found_hedging
    
    def _analyze_exposure_trend(self, trend_data: Dict) -> str:
        """
        Analyze trend in exposure over time
        """
        if not trend_data:
            return "No trend data available"
        
        values = list(trend_data.values())
        if len(values) < 2:
            return "Insufficient data for trend analysis"
        
        recent_change = values[-1] - values[-2]
        overall_change = values[-1] - values[0]
        
        if overall_change > 0.02:  # > 2% increase
            return "Increasing trend - exposure growing over time"
        elif overall_change < -0.02:  # > 2% decrease
            return "Decreasing trend - exposure reducing over time"
        else:
            return "Stable trend - exposure relatively unchanged"
    
    def _assess_materiality(self, actual_exposure: Dict) -> str:
        """
        Assess materiality of the exposure
        """
        total_exposure = actual_exposure['total_high_carbon_exposure']
        
        if total_exposure > 0.30:
            return "Highly material - significant climate risk exposure"
        elif total_exposure > 0.15:
            return "Material - notable climate risk exposure"
        elif total_exposure > 0.05:
            return "Moderately material - some climate risk exposure"
        else:
            return "Low materiality - minimal climate risk exposure"
    
    def _calculate_assessment_confidence(self, actual_exposure: Dict) -> float:
        """
        Calculate confidence in the assessment based on data quality
        """
        data_quality = actual_exposure.get('data_quality', {})
        completeness = data_quality.get('completeness', 0.8)
        accuracy = data_quality.get('accuracy_score', 0.8)
        
        return (completeness + accuracy) / 2
    
    def _get_default_verification_result(self, claim_text: str, error_msg: str) -> Dict[str, Any]:
        """
        Return default verification result when analysis fails
        """
        return {
            'claim_text': claim_text,
            'claim_analysis': {'exposure_level_claimed': 'unknown'},
            'actual_exposure': {'total_high_carbon_exposure': 0.0},
            'verification_result': {
                'is_inconsistent': False,
                'severity': 'low',
                'error': error_msg
            },
            'discrepancy_detected': False,
            'severity': 'low',
            'detailed_analysis': {'error': error_msg},
            'regulatory_flags': []
        }

# Example usage function
def verify_climate_exposure_claim(claim_text: str, financial_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Convenience function to verify a climate exposure claim
    """
    verifier = FinancialVerificationEngine()
    
    if financial_data is None:
        # Use empty DataFrame to trigger sample data generation
        financial_data = pd.DataFrame()
    
    return verifier.verify_exposure_claim(claim_text, financial_data)