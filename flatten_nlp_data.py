#!/usr/bin/env python3
"""
NLP Data Flattening and Cleansing

This script addresses the nesting challenges by providing multiple cleansing options:
1. Flattened columns for each data type
2. Simplified text-based representations
3. Separate normalized tables for complex data
4. NLP-optimized formats
"""

import pandas as pd
import json
import re
from typing import Dict, List, Any
from collections import defaultdict

class NLPDataFlattener:
    """Flatten nested JSON data for NLP workflows."""
    
    def __init__(self):
        """Initialize the data flattener."""
        self.entity_type_mapping = {
            'PERSON': 'person',
            'ORG': 'organization', 
            'MONEY': 'monetary_amount',
            'PERCENT': 'percentage',
            'DATE': 'date',
            'CARDINAL': 'number',
            'GPE': 'location',
            'ORDINAL': 'ordinal'
        }
    
    def flatten_named_entities(self, entities_json: str) -> Dict[str, Any]:
        """Flatten named entities into separate columns."""
        if not entities_json or entities_json == '':
            return {
                'entities_text': '',
                'entity_count': 0,
                'has_person': False,
                'has_organization': False,
                'has_money': False,
                'has_date': False,
                'person_entities': '',
                'org_entities': '',
                'money_entities': '',
                'date_entities': '',
                'all_entities': ''
            }
        
        try:
            entities_dict = json.loads(entities_json)
        except:
            return self.flatten_named_entities('')
        
        # Extract entities by type
        persons = entities_dict.get('PERSON', [])
        orgs = entities_dict.get('ORG', [])
        money = entities_dict.get('MONEY', [])
        dates = entities_dict.get('DATE', [])
        
        # Collect all entities
        all_entities = []
        for entity_type, entities in entities_dict.items():
            all_entities.extend(entities)
        
        return {
            'entities_text': ' | '.join(all_entities),
            'entity_count': len(all_entities),
            'has_person': len(persons) > 0,
            'has_organization': len(orgs) > 0,
            'has_money': len(money) > 0,
            'has_date': len(dates) > 0,
            'person_entities': ' | '.join(persons),
            'org_entities': ' | '.join(orgs),
            'money_entities': ' | '.join(money),
            'date_entities': ' | '.join(dates),
            'all_entities': ' | '.join(all_entities)
        }
    
    def flatten_financial_terms(self, terms_json: str) -> Dict[str, Any]:
        """Flatten financial terms into separate columns."""
        if not terms_json or terms_json == '':
            return {
                'financial_terms_text': '',
                'financial_term_count': 0,
                'has_metrics': False,
                'has_instruments': False,
                'has_segments': False,
                'has_currencies': False,
                'metrics_terms': '',
                'instruments_terms': '',
                'segments_terms': '',
                'currencies_terms': '',
                'all_financial_terms': ''
            }
        
        try:
            terms_dict = json.loads(terms_json)
        except:
            return self.flatten_financial_terms('')
        
        # Extract terms by category
        metrics = terms_dict.get('metrics', [])
        instruments = terms_dict.get('financial_instruments', [])
        segments = terms_dict.get('business_segments', [])
        currencies = terms_dict.get('currencies', [])
        
        # Collect all terms
        all_terms = []
        for category, terms in terms_dict.items():
            all_terms.extend(terms)
        
        return {
            'financial_terms_text': ' | '.join(all_terms),
            'financial_term_count': len(all_terms),
            'has_metrics': len(metrics) > 0,
            'has_instruments': len(instruments) > 0,
            'has_segments': len(segments) > 0,
            'has_currencies': len(currencies) > 0,
            'metrics_terms': ' | '.join(metrics),
            'instruments_terms': ' | '.join(instruments),
            'segments_terms': ' | '.join(segments),
            'currencies_terms': ' | '.join(currencies),
            'all_financial_terms': ' | '.join(all_terms)
        }
    
    def flatten_financial_figures(self, figures_json: str) -> Dict[str, Any]:
        """Flatten financial figures into separate columns."""
        if not figures_json or figures_json == '':
            return {
                'financial_figures_text': '',
                'figure_count': 0,
                'has_billions': False,
                'has_millions': False,
                'has_percentages': False,
                'billion_figures': '',
                'million_figures': '',
                'percentage_figures': '',
                'all_figures': ''
            }
        
        try:
            figures_list = json.loads(figures_json)
        except:
            return self.flatten_financial_figures('')
        
        if not isinstance(figures_list, list):
            return self.flatten_financial_figures('')
        
        # Categorize figures
        billions = []
        millions = []
        percentages = []
        all_figures = []
        
        for figure in figures_list:
            if isinstance(figure, dict):
                value = figure.get('value', '') or ''
                unit = figure.get('unit', '') or ''
                unit = str(unit).lower()
                
                figure_text = f"{value} {unit}".strip()
                all_figures.append(figure_text)
                
                if 'billion' in unit or 'bn' in unit:
                    billions.append(figure_text)
                elif 'million' in unit or 'mn' in unit:
                    millions.append(figure_text)
                elif '%' in str(value):
                    percentages.append(figure_text)
        
        return {
            'financial_figures_text': ' | '.join(all_figures),
            'figure_count': len(all_figures),
            'has_billions': len(billions) > 0,
            'has_millions': len(millions) > 0,
            'has_percentages': len(percentages) > 0,
            'billion_figures': ' | '.join(billions),
            'million_figures': ' | '.join(millions),
            'percentage_figures': ' | '.join(percentages),
            'all_figures': ' | '.join(all_figures)
        }
    
    def create_nlp_features(self, row: pd.Series) -> Dict[str, Any]:
        """Create NLP-optimized feature columns."""
        features = {}
        
        # Text-based features
        text = str(row.get('text', ''))
        
        # Length features
        features['char_count'] = len(text)
        features['word_count_calculated'] = len(text.split())
        features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
        
        # Content type features
        features['is_financial_content'] = row.get('financial_term_count', 0) > 0
        features['is_entity_rich'] = row.get('entity_count', 0) > 2
        features['is_numeric_heavy'] = row.get('figure_count', 0) > 0
        
        # Speaker features
        features['is_management'] = row.get('speaker_category_enhanced', '') in ['CEO', 'CFO']
        features['is_analyst'] = row.get('speaker_category_enhanced', '') == 'Analyst'
        features['is_named_speaker'] = row.get('speaker_norm', 'UNKNOWN') != 'UNKNOWN'
        
        # Content complexity
        features['complexity_score'] = (
            (row.get('entity_count', 0) * 0.3) +
            (row.get('financial_term_count', 0) * 0.4) +
            (row.get('figure_count', 0) * 0.3)
        )
        
        return features

def create_flattened_nlp_dataset():
    """Create multiple flattened datasets for different NLP workflows."""
    
    print("🔧 FLATTENING NESTED DATA FOR NLP WORKFLOWS")
    print("=" * 60)
    
    # Load enhanced data
    df = pd.read_csv('processed_data_enhanced.csv')
    print(f"📊 Processing {len(df)} records with nested data...")
    
    # Initialize flattener
    flattener = NLPDataFlattener()
    
    # Create flattened dataset
    flattened_df = df.copy()
    
    print("🏷️ Flattening named entities...")
    entity_features = []
    for idx, row in df.iterrows():
        entity_flat = flattener.flatten_named_entities(row.get('named_entities_enhanced', ''))
        entity_features.append(entity_flat)
        
        if idx % 500 == 0:
            print(f"   Processed {idx}/{len(df)} records...")
    
    # Add entity features to dataframe
    entity_df = pd.DataFrame(entity_features)
    for col in entity_df.columns:
        flattened_df[col] = entity_df[col]
    
    print("💰 Flattening financial terms...")
    financial_features = []
    for idx, row in df.iterrows():
        financial_flat = flattener.flatten_financial_terms(row.get('financial_terms', ''))
        financial_features.append(financial_flat)
        
        if idx % 500 == 0:
            print(f"   Processed {idx}/{len(df)} records...")
    
    # Add financial features to dataframe
    financial_df = pd.DataFrame(financial_features)
    for col in financial_df.columns:
        flattened_df[col] = financial_df[col]
    
    print("📊 Flattening financial figures...")
    figure_features = []
    for idx, row in df.iterrows():
        figure_flat = flattener.flatten_financial_figures(row.get('financial_figures', ''))
        figure_features.append(figure_flat)
        
        if idx % 500 == 0:
            print(f"   Processed {idx}/{len(df)} records...")
    
    # Add figure features to dataframe
    figure_df = pd.DataFrame(figure_features)
    for col in figure_df.columns:
        flattened_df[col] = figure_df[col]
    
    print("🚀 Creating NLP-optimized features...")
    nlp_features = []
    for idx, row in flattened_df.iterrows():
        nlp_flat = flattener.create_nlp_features(row)
        nlp_features.append(nlp_flat)
        
        if idx % 500 == 0:
            print(f"   Processed {idx}/{len(flattened_df)} records...")
    
    # Add NLP features to dataframe
    nlp_df = pd.DataFrame(nlp_features)
    for col in nlp_df.columns:
        flattened_df[col] = nlp_df[col]
    
    # Save complete flattened dataset
    flattened_file = 'processed_data_flattened_nlp.csv'
    flattened_df.to_csv(flattened_file, index=False, encoding='utf-8-sig')
    
    # Create specialized NLP datasets
    print("📋 Creating specialized NLP datasets...")
    
    # 1. Core NLP dataset (essential fields only)
    core_nlp_columns = [
        'Review_ID', 'source_file', 'quarter_period', 'sentence_id',
        'speaker_norm', 'speaker_role', 'speaker_category_enhanced',
        'text', 'word_count', 'char_count', 'complexity_score',
        'is_management', 'is_analyst', 'is_financial_content',
        'entities_text', 'financial_terms_text', 'financial_figures_text'
    ]
    
    # Add Review_ID if not present
    if 'Review_ID' not in flattened_df.columns:
        flattened_df.insert(0, 'Review_ID', range(1, len(flattened_df) + 1))
    
    core_nlp_df = flattened_df[[col for col in core_nlp_columns if col in flattened_df.columns]].copy()
    core_nlp_file = 'nlp_core_dataset.csv'
    core_nlp_df.to_csv(core_nlp_file, index=False, encoding='utf-8-sig')
    
    # 2. Financial analysis dataset
    financial_columns = [
        'Review_ID', 'source_file', 'quarter_period', 'sentence_id',
        'speaker_category_enhanced', 'text', 'is_financial_content',
        'financial_term_count', 'figure_count', 'has_metrics', 'has_instruments',
        'metrics_terms', 'instruments_terms', 'all_financial_terms',
        'billion_figures', 'million_figures', 'percentage_figures'
    ]
    
    financial_nlp_df = flattened_df[[col for col in financial_columns if col in flattened_df.columns]].copy()
    # Filter to financial content only
    financial_nlp_df = financial_nlp_df[financial_nlp_df['is_financial_content'] == True]
    financial_nlp_file = 'nlp_financial_dataset.csv'
    financial_nlp_df.to_csv(financial_nlp_file, index=False, encoding='utf-8-sig')
    
    # 3. Speaker analysis dataset
    speaker_columns = [
        'Review_ID', 'source_file', 'quarter_period', 'sentence_id',
        'speaker_norm', 'speaker_role', 'speaker_category_enhanced',
        'is_management', 'is_analyst', 'text', 'word_count',
        'entities_text', 'person_entities', 'org_entities'
    ]
    
    speaker_nlp_df = flattened_df[[col for col in speaker_columns if col in flattened_df.columns]].copy()
    # Filter to named speakers only
    speaker_nlp_df = speaker_nlp_df[speaker_nlp_df['speaker_norm'] != 'UNKNOWN']
    speaker_nlp_file = 'nlp_speaker_dataset.csv'
    speaker_nlp_df.to_csv(speaker_nlp_file, index=False, encoding='utf-8-sig')
    
    # 4. Entity-rich dataset
    entity_columns = [
        'Review_ID', 'source_file', 'quarter_period', 'sentence_id',
        'text', 'entity_count', 'is_entity_rich',
        'has_person', 'has_organization', 'has_money', 'has_date',
        'person_entities', 'org_entities', 'money_entities', 'date_entities'
    ]
    
    entity_nlp_df = flattened_df[[col for col in entity_columns if col in flattened_df.columns]].copy()
    # Filter to entity-rich content only
    entity_nlp_df = entity_nlp_df[entity_nlp_df['entity_count'] > 0]
    entity_nlp_file = 'nlp_entity_dataset.csv'
    entity_nlp_df.to_csv(entity_nlp_file, index=False, encoding='utf-8-sig')
    
    # Generate summary report
    print("📊 Generating flattening summary...")
    
    summary_file = 'nlp_flattening_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("NLP DATA FLATTENING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("FLATTENING APPROACH:\n")
        f.write("-" * 30 + "\n")
        f.write("✅ JSON structures converted to flat columns\n")
        f.write("✅ Boolean flags for presence detection\n")
        f.write("✅ Text concatenation for simple access\n")
        f.write("✅ Categorical separation by type\n")
        f.write("✅ NLP-optimized feature engineering\n\n")
        
        f.write("DATASETS CREATED:\n")
        f.write("-" * 30 + "\n")
        f.write(f"1. {flattened_file} - Complete flattened dataset ({len(flattened_df)} records)\n")
        f.write(f"2. {core_nlp_file} - Core NLP essentials ({len(core_nlp_df)} records)\n")
        f.write(f"3. {financial_nlp_file} - Financial analysis focus ({len(financial_nlp_df)} records)\n")
        f.write(f"4. {speaker_nlp_file} - Speaker analysis focus ({len(speaker_nlp_df)} records)\n")
        f.write(f"5. {entity_nlp_file} - Entity-rich content ({len(entity_nlp_df)} records)\n\n")
        
        f.write("FLATTENED FEATURES:\n")
        f.write("-" * 30 + "\n")
        f.write("Named Entities:\n")
        f.write("  - entities_text (pipe-separated)\n")
        f.write("  - entity_count, has_person, has_organization\n")
        f.write("  - person_entities, org_entities, money_entities\n\n")
        
        f.write("Financial Terms:\n")
        f.write("  - financial_terms_text (pipe-separated)\n")
        f.write("  - financial_term_count, has_metrics, has_instruments\n")
        f.write("  - metrics_terms, instruments_terms, segments_terms\n\n")
        
        f.write("Financial Figures:\n")
        f.write("  - financial_figures_text (pipe-separated)\n")
        f.write("  - figure_count, has_billions, has_millions\n")
        f.write("  - billion_figures, million_figures, percentage_figures\n\n")
        
        f.write("NLP Features:\n")
        f.write("  - complexity_score, is_financial_content\n")
        f.write("  - is_management, is_analyst, is_entity_rich\n")
        f.write("  - char_count, word_count_calculated\n\n")
        
        f.write("USAGE RECOMMENDATIONS:\n")
        f.write("-" * 30 + "\n")
        f.write("• Use core_dataset for general NLP tasks\n")
        f.write("• Use financial_dataset for domain-specific analysis\n")
        f.write("• Use speaker_dataset for attribution analysis\n")
        f.write("• Use entity_dataset for information extraction\n")
        f.write("• Boolean flags enable easy filtering\n")
        f.write("• Text fields support direct NLP processing\n")
    
    # Print completion summary
    print(f"\n✅ NLP DATA FLATTENING COMPLETE")
    print("=" * 60)
    print(f"📄 Complete flattened: {flattened_file}")
    print(f"   • {len(flattened_df):,} records with {len(flattened_df.columns)} flat columns")
    
    print(f"\n📋 Specialized NLP datasets:")
    print(f"   • Core NLP: {core_nlp_file} ({len(core_nlp_df):,} records)")
    print(f"   • Financial: {financial_nlp_file} ({len(financial_nlp_df):,} records)")
    print(f"   • Speaker: {speaker_nlp_file} ({len(speaker_nlp_df):,} records)")
    print(f"   • Entity-rich: {entity_nlp_file} ({len(entity_nlp_df):,} records)")
    
    print(f"\n📊 Summary: {summary_file}")
    print(f"   • Detailed flattening approach and usage guide")
    
    print(f"\n🎯 NLP WORKFLOW BENEFITS:")
    print(f"   • No more JSON parsing required")
    print(f"   • Direct column access for all features")
    print(f"   • Boolean flags for easy filtering")
    print(f"   • Specialized datasets for focused analysis")
    
    return flattened_df

if __name__ == "__main__":
    flattened_data = create_flattened_nlp_dataset()
    print(f"\n🎉 NLP data flattening complete - Ready for analysis!")