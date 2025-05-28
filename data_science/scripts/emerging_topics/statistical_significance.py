"""
Statistical Significance Testing for Emerging Topics Analysis
Provides comprehensive statistical validation for trend detection
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

class StatisticalSignificanceTester:
    """
    Comprehensive statistical testing for emerging topics analysis
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.confidence_level = 1 - alpha
    
    def test_frequency_change(self, recent_counts: List[int], 
                            historical_counts: List[int]) -> Dict[str, Any]:
        """
        Test statistical significance of frequency changes using Mann-Whitney U test
        """
        try:
            if len(recent_counts) < 2 or len(historical_counts) < 2:
                return self._get_default_test_result()
            
            statistic, p_value = stats.mannwhitneyu(
                recent_counts, historical_counts, alternative='two-sided'
            )
            
            effect_size = self._calculate_effect_size_mannwhitney(
                recent_counts, historical_counts
            )
            
            return {
                'test_name': 'Mann-Whitney U Test',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_significant': p_value < self.alpha,
                'effect_size': effect_size,
                'interpretation': self._interpret_effect_size(effect_size)
            }
            
        except Exception as e:
            return self._get_default_test_result()
    
    def test_sentiment_change(self, recent_sentiment: List[float], 
                            historical_sentiment: List[float]) -> Dict[str, Any]:
        """
        Test statistical significance of sentiment changes using t-test
        """
        try:
            if len(recent_sentiment) < 2 or len(historical_sentiment) < 2:
                return self._get_default_test_result()
            
            # Remove NaN values
            recent_clean = [x for x in recent_sentiment if not np.isnan(x)]
            historical_clean = [x for x in historical_sentiment if not np.isnan(x)]
            
            if len(recent_clean) < 2 or len(historical_clean) < 2:
                return self._get_default_test_result()
            
            statistic, p_value = stats.ttest_ind(recent_clean, historical_clean)
            
            # Calculate Cohen's d for effect size
            effect_size = self._calculate_cohens_d(recent_clean, historical_clean)
            
            return {
                'test_name': 'Independent t-test',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_significant': p_value < self.alpha,
                'effect_size': effect_size,
                'interpretation': self._interpret_effect_size(effect_size)
            }
            
        except Exception as e:
            return self._get_default_test_result()
    
    def test_speaker_pattern_change(self, recent_speakers: List[str], 
                                  historical_speakers: List[str]) -> Dict[str, Any]:
        """
        Test statistical significance of speaker pattern changes using Chi-square test
        """
        try:
            # Create contingency table
            all_speakers = list(set(recent_speakers + historical_speakers))
            
            if len(all_speakers) < 2:
                return self._get_default_test_result()
            
            recent_counts = [recent_speakers.count(speaker) for speaker in all_speakers]
            historical_counts = [historical_speakers.count(speaker) for speaker in all_speakers]
            
            # Create contingency table
            contingency_table = np.array([recent_counts, historical_counts])
            
            # Ensure minimum expected frequencies
            if np.any(contingency_table.sum(axis=0) < 5):
                return self._get_default_test_result()
            
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Calculate Cramér's V for effect size
            effect_size = self._calculate_cramers_v(chi2, contingency_table)
            
            return {
                'test_name': 'Chi-square test',
                'statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'is_significant': p_value < self.alpha,
                'effect_size': effect_size,
                'interpretation': self._interpret_effect_size(effect_size)
            }
            
        except Exception as e:
            return self._get_default_test_result()
    
    def multiple_testing_correction(self, p_values: List[float], 
                                  method: str = 'bonferroni') -> Dict[str, Any]:
        """
        Apply multiple testing correction
        """
        try:
            p_values_array = np.array(p_values)
            
            if method.lower() == 'bonferroni':
                corrected_p_values = p_values_array * len(p_values)
                corrected_p_values = np.minimum(corrected_p_values, 1.0)
            elif method.lower() == 'holm':
                # Holm-Bonferroni method
                sorted_indices = np.argsort(p_values_array)
                corrected_p_values = np.zeros_like(p_values_array)
                
                for i, idx in enumerate(sorted_indices):
                    correction_factor = len(p_values) - i
                    corrected_p_values[idx] = min(
                        p_values_array[idx] * correction_factor, 1.0
                    )
            else:
                corrected_p_values = p_values_array
            
            return {
                'method': method,
                'original_p_values': p_values,
                'corrected_p_values': corrected_p_values.tolist(),
                'significant_after_correction': (corrected_p_values < self.alpha).tolist(),
                'alpha_level': self.alpha
            }
            
        except Exception as e:
            return {
                'method': method,
                'original_p_values': p_values,
                'corrected_p_values': p_values,
                'significant_after_correction': [p < self.alpha for p in p_values],
                'alpha_level': self.alpha
            }
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate Cohen's d effect size
        """
        try:
            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            
            if pooled_std == 0:
                return 0.0
            
            cohens_d = (mean1 - mean2) / pooled_std
            return float(cohens_d)
            
        except Exception:
            return 0.0
    
    def _calculate_effect_size_mannwhitney(self, group1: List[int], group2: List[int]) -> float:
        """
        Calculate effect size for Mann-Whitney U test (rank-biserial correlation)
        """
        try:
            n1, n2 = len(group1), len(group2)
            U1, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            
            # Rank-biserial correlation
            r = 1 - (2 * U1) / (n1 * n2)
            return float(abs(r))
            
        except Exception:
            return 0.0
    
    def _calculate_cramers_v(self, chi2: float, contingency_table: np.ndarray) -> float:
        """
        Calculate Cramér's V effect size for Chi-square test
        """
        try:
            n = contingency_table.sum()
            min_dim = min(contingency_table.shape) - 1
            
            if n == 0 or min_dim == 0:
                return 0.0
            
            cramers_v = np.sqrt(chi2 / (n * min_dim))
            return float(cramers_v)
            
        except Exception:
            return 0.0
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """
        Interpret effect size magnitude
        """
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return 'Small effect'
        elif abs_effect < 0.5:
            return 'Medium effect'
        elif abs_effect < 0.8:
            return 'Large effect'
        else:
            return 'Very large effect'
    
    def _get_default_test_result(self) -> Dict[str, Any]:
        """
        Return default test result when testing fails
        """
        return {
            'test_name': 'Statistical Test',
            'statistic': 0.0,
            'p_value': 1.0,
            'is_significant': False,
            'effect_size': 0.0,
            'interpretation': 'No effect'
        }